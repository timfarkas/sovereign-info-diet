#!/usr/bin/env python3
"""X/Twitter ingestion for the AI digest, via twitterapi.io.

Shape mirrors reddit_scraper.py: construct, call a scrape method, get a list of
normalized dicts, save to extracts/.

WHY twitterapi.io and not nitter or X's own API: see KYRO-9. Short version --
nitter was cease-and-desisted out of existence in Aug 2026, X's own API is
$0.005/post-read with no free tier, twitterapi.io is $0.00015/tweet.

ASSUMPTIONS, stated so they fail loudly rather than silently:
  1. TWITTER_IO_API_KEY exists in .env. Missing key raises at construction.
  2. The account universe is whoever X_SEED_ACCOUNTS follow. That list is
     cached and only refreshed every X_ACCOUNT_LIST_TTL_DAYS, because it moves
     far more slowly than the tweets do.
  3. twitterapi.io free tier allows 1 request / 5s. We pace to that whether or
     not we're on free tier -- being fast here buys us nothing.
  4. `since:` in X search is DATE granularity, not timestamp. So every run
     re-sees tweets it already has; dedupe against the seen-ids store is what
     makes the run idempotent, not the query.
  5. X search silently returns ZERO results for an over-long query instead of
     erroring. Measured 2026-09-22: 22 OR'd from: terms (466 chars) works,
     24 terms (514 chars) returns 0 for accounts that demonstrably posted. So
     batches are packed by QUERY LENGTH, not by a fixed account count, and a
     run where suspiciously many batches come back empty is flagged degraded.
  6. A run that fetches SOME batches is worth more than nothing. Batches are
     written to disk as they complete and a re-run resumes from them, so
     twitterapi.io going down mid-run costs us the remainder, not the whole.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv(os.getenv("DOTENV_PATH") or None)

API_BASE = "https://api.twitterapi.io"

# domains we refuse to surface in the digest -- Tim blocks these on his devices
# on purpose, so a link to one is a dead link AND a temptation. Both bad.
BLOCKED_LINK_DOMAINS = (
    "x.com", "twitter.com", "t.co", "mobile.twitter.com",
    "reddit.com", "redd.it", "redditmedia.com",
)


class XFeedDown(RuntimeError):
    """twitterapi.io gave us nothing usable at all. Loud on purpose."""


def is_blocked_link(url: str) -> bool:
    """True for links we must not put in the digest."""
    if not url:
        return True
    host = url.split("//", 1)[-1].split("/", 1)[0].split("@")[-1].split(":")[0].lower()
    host = host[4:] if host.startswith("www.") else host
    return any(host == d or host.endswith("." + d) for d in BLOCKED_LINK_DOMAINS)


class TwitterIOClient:
    """Thin twitterapi.io client with QPS pacing and retry.

    Records every failure it survives so the pipeline can tell Tim the feed is
    sick even when the run technically succeeded.
    """

    def __init__(self, api_key: str = None, min_interval: float = 5.5,
                 max_retries: int = 4, timeout: int = 60):
        self.api_key = api_key or os.getenv("TWITTER_IO_API_KEY")
        if not self.api_key:
            raise ValueError("TWITTER_IO_API_KEY not found in .env")
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = self.api_key
        self._last_call = 0.0
        self.errors: List[str] = []
        self.calls = 0

    def _pace(self):
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def get(self, path: str, **params) -> Dict[str, Any]:
        """GET with retry/backoff. Raises XFeedDown after max_retries."""
        delay = self.min_interval
        last = ""
        for attempt in range(1, self.max_retries + 1):
            self._pace()
            try:
                r = self.session.get(API_BASE + path, params=params, timeout=self.timeout)
                self.calls += 1
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError:
                        last = f"HTTP 200 but unparseable body: {r.text[:120]}"
                elif r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}: {r.text[:120]}"
                else:
                    # 4xx that retrying won't fix -- fail immediately
                    msg = f"{path} -> HTTP {r.status_code}: {r.text[:200]}"
                    self.errors.append(msg)
                    raise XFeedDown(msg)
            self.errors.append(f"{path} attempt {attempt}/{self.max_retries}: {last}")
            print(f"  [x] retry {attempt}/{self.max_retries} on {path}: {last}")
            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
        raise XFeedDown(f"{path} failed after {self.max_retries} attempts: {last}")

    def credits(self) -> Optional[int]:
        """Remaining credits, or None if the endpoint didn't answer."""
        try:
            d = self.get("/oapi/my/info")
        except XFeedDown:
            return None
        return int(d.get("recharge_credits", 0)) + int(d.get("total_bonus_credits", 0))


class XScraper:
    def __init__(self, base_dir: str = "extracts", client: TwitterIOClient = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.state_dir = self.base_dir / "x_state"
        self.state_dir.mkdir(exist_ok=True)
        self.accounts_cache = self.base_dir / "x_accounts.json"
        self.seen_path = self.base_dir / "x_seen_ids.json"
        self.health_path = self.base_dir / "x_health.json"
        self.client = client or TwitterIOClient()
        self.notes: List[str] = []      # informational; shown but not alarming
        self.warnings: List[str] = []   # degrades health -- something needs a human

    # --- account universe ----------------------------------------------
    def _fetch_followings(self, handle: str) -> List[Dict[str, Any]]:
        out, cursor = [], ""
        while True:
            d = self.client.get("/twitter/user/followings",
                                userName=handle, pageSize=200, cursor=cursor)
            page = d.get("followings") or []
            out += page
            print(f"  [x] @{handle}: +{len(page)} followings (total {len(out)})")
            if not (d.get("has_next_page") and d.get("next_cursor") and page):
                break
            cursor = d["next_cursor"]
        return out

    def account_list(self, seeds: List[str], ttl_days: int = 7,
                     force: bool = False) -> List[str]:
        """Union of who the seeds follow. Cached; stale cache beats no list."""
        cached = None
        if self.accounts_cache.exists():
            cached = json.loads(self.accounts_cache.read_text())
            age = datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"])
            if not force and age < timedelta(days=ttl_days):
                print(f"  [x] account list from cache ({len(cached['accounts'])} accounts, "
                      f"{age.days}d old)")
                return cached["accounts"]

        accounts: Dict[str, None] = {}
        empty_seeds = []
        try:
            for seed in seeds:
                found = self._fetch_followings(seed)
                if not found:
                    empty_seeds.append(seed)
                for u in found:
                    if u.get("userName"):
                        accounts[u["userName"]] = None
        except XFeedDown as e:
            if cached:
                self.warnings.append(f"account list refresh failed ({e}); using stale cache")
                print(f"  [x] refresh failed, falling back to stale cache: {e}")
                return cached["accounts"]
            raise

        if empty_seeds:
            self.notes.append(f"seed accounts following nobody (or unreadable): "
                              f"{', '.join('@' + s for s in empty_seeds)}")
        if not accounts:
            raise XFeedDown(f"no accounts found across seeds {seeds}")

        names = sorted(accounts)
        self.accounts_cache.write_text(json.dumps(
            {"fetched_at": datetime.now(timezone.utc).isoformat(),
             "seeds": seeds, "accounts": names}, indent=2))
        print(f"  [x] account list refreshed: {len(names)} accounts from {len(seeds)} seeds")
        return names

    # --- tweets ---------------------------------------------------------
    @staticmethod
    def _normalize(t: Dict[str, Any]) -> Dict[str, Any]:
        author = t.get("author") or {}
        links = [u.get("expanded_url") or u.get("url")
                 for u in ((t.get("entities") or {}).get("urls") or [])]
        quoted = t.get("quoted_tweet") or {}
        return {
            "id": str(t.get("id")),
            "handle": author.get("userName"),
            "author_name": author.get("name"),
            "author_followers": author.get("followers"),
            "text": t.get("text") or "",
            "created_at": t.get("createdAt"),
            "likes": t.get("likeCount"),
            "retweets": t.get("retweetCount"),
            "views": t.get("viewCount"),
            "replies": t.get("replyCount"),
            "is_reply": bool(t.get("isReply")),
            "quoted_text": (quoted.get("text") or "")[:400] if quoted else "",
            "external_links": [u for u in links if u and not is_blocked_link(u)],
        }

    def _batch_query(self, batch: List[str], since_date: str) -> str:
        return ("(" + " OR ".join(f"from:{u}" for u in batch) + ")"
                f" -filter:replies since:{since_date}")

    def make_batches(self, accounts: List[str], since_date: str,
                     max_query_chars: int = 400,
                     max_accounts_per_batch: int = 20) -> List[List[str]]:
        """Pack accounts into batches that stay well under X's query-length cliff.

        Sized by characters rather than count because handle lengths vary by 3x;
        a fixed count of long handles is exactly how you fall off the cliff and
        get a silent zero.
        """
        batches, current = [], []
        for acct in accounts:
            trial = current + [acct]
            if current and (len(self._batch_query(trial, since_date)) > max_query_chars
                            or len(trial) > max_accounts_per_batch):
                batches.append(current)
                current = [acct]
            else:
                current = trial
        if current:
            batches.append(current)
        return batches

    def fetch_batch(self, batch: List[str], since_date: str,
                    max_pages: int = 5) -> List[Dict[str, Any]]:
        tweets, cursor = [], ""
        for _ in range(max_pages):
            d = self.client.get("/twitter/tweet/advanced_search",
                                query=self._batch_query(batch, since_date),
                                queryType="Latest", cursor=cursor)
            page = d.get("tweets") or []
            tweets += page
            if not (d.get("has_next_page") and d.get("next_cursor") and page):
                break
            cursor = d["next_cursor"]
        return [self._normalize(t) for t in tweets]

    # --- seen-id store --------------------------------------------------
    def _load_seen(self) -> set:
        if not self.seen_path.exists():
            return set()
        return set(json.loads(self.seen_path.read_text()).get("ids", []))

    def _save_seen(self, seen: set, keep: int = 20000):
        ids = sorted(seen)[-keep:]   # tweet ids are monotonic, so this keeps the newest
        self.seen_path.write_text(json.dumps({"ids": ids}))

    # --- the run --------------------------------------------------------
    def scrape(self, seeds: List[str], since_days: int = 1,
               max_query_chars: int = 400, max_accounts_per_batch: int = 20,
               max_tweets: int = 1500, max_accounts: int = None,
               ttl_days: int = 7, run_id: str = None) -> List[Dict[str, Any]]:
        """One ingestion run. Resumable, deduped, and always writes health."""
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        run_dir = self.state_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%Y-%m-%d")

        accounts = self.account_list(seeds, ttl_days=ttl_days)
        if max_accounts:
            accounts = accounts[:max_accounts]
        batches = self.make_batches(accounts, since_date,
                                    max_query_chars=max_query_chars,
                                    max_accounts_per_batch=max_accounts_per_batch)

        print(f"  [x] {len(accounts)} accounts in {len(batches)} batches, since {since_date}")
        fetched, failed, resumed, attempted, empty, capped = [], 0, 0, 0, 0, False
        for n, batch in enumerate(batches):
            cache = run_dir / f"batch_{n:03d}.json"
            if cache.exists():
                fetched += json.loads(cache.read_text())
                resumed += 1
                attempted += 1
                continue
            if len(fetched) >= max_tweets:
                capped = True
                print(f"  [x] tweet cap {max_tweets} hit at batch {n}/{len(batches)}, stopping")
                break
            attempted += 1
            try:
                got = self.fetch_batch(batch, since_date)
            except XFeedDown as e:
                failed += 1
                print(f"  [x] batch {n} FAILED (kept going): {e}")
                continue
            cache.write_text(json.dumps(got, indent=1))
            if not got:
                empty += 1
            fetched += got
            print(f"  [x] batch {n + 1}/{len(batches)}: {len(got)} tweets "
                  f"(running total {len(fetched)})")

        if resumed:
            print(f"  [x] resumed {resumed} batches from {run_dir}")
        live = attempted - resumed - failed
        if live >= 4 and empty > live * 0.6:
            # the signature of the silent over-long-query failure, not of a
            # quiet news day. Worth a human look before trusting the digest.
            self.warnings.append(
                f"{empty}/{live} freshly-fetched batches returned zero tweets -- "
                f"suspiciously many; check X search query length/syntax")

        seen = self._load_seen()
        fresh, by_id = [], {}
        for t in fetched:
            if t["id"] in seen or t["id"] in by_id:
                continue
            by_id[t["id"]] = t
            fresh.append(t)
        fresh.sort(key=lambda t: t["id"], reverse=True)

        self._write_health(attempted=attempted, failed=failed, tweets=len(fresh),
                           accounts=len(accounts), capped=capped)
        if attempted and failed == attempted:
            raise XFeedDown(f"all {attempted} batches failed -- twitterapi.io looks down")

        self._save_seen(seen | set(by_id))
        return fresh

    def _write_health(self, attempted: int, failed: int, tweets: int,
                      accounts: int, capped: bool):
        if attempted and failed == attempted:
            status = "down"
        elif failed or self.warnings:
            status = "degraded"
        else:
            status = "healthy"
        payload = {
            "status": status,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "batches_attempted": attempted,
            "batches_failed": failed,
            "accounts": accounts,
            "tweets_new": tweets,
            "tweet_cap_hit": capped,
            "api_calls": self.client.calls,
            "notes": self.notes + self.warnings,
            "errors": self.client.errors[-10:],
        }
        self.health_path.write_text(json.dumps(payload, indent=2))
        print(f"  [x] health: {status} "
              f"({failed}/{attempted} batches failed, {tweets} new tweets)")

    def save_to_json(self, data: List[Dict], filename: str = None) -> str:
        filename = filename or f"x_data_{datetime.now():%Y%m%d_%H%M%S}.json"
        path = self.base_dir / filename
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  [x] saved {len(data)} tweets to {path}")
        return str(path)


def main() -> int:
    from config import (X_SEED_ACCOUNTS, X_MAX_QUERY_CHARS, X_MAX_ACCOUNTS_PER_BATCH,
                        X_MAX_TWEETS_PER_RUN, X_MAX_ACCOUNTS,
                        X_ACCOUNT_LIST_TTL_DAYS, TIME_HORIZON_DAYS)
    print("=== X ingestion (twitterapi.io) ===")
    scraper = XScraper()
    credits = scraper.client.credits()
    print(f"  [x] credits remaining: {credits if credits is not None else 'unknown'}")
    try:
        tweets = scraper.scrape(
            seeds=X_SEED_ACCOUNTS, since_days=TIME_HORIZON_DAYS,
            max_query_chars=X_MAX_QUERY_CHARS,
            max_accounts_per_batch=X_MAX_ACCOUNTS_PER_BATCH,
            max_tweets=X_MAX_TWEETS_PER_RUN,
            max_accounts=X_MAX_ACCOUNTS, ttl_days=X_ACCOUNT_LIST_TTL_DAYS)
    except XFeedDown as e:
        # health file is written by scrape() for partial failures; for a total
        # wipeout before any batch ran, write it here so the digest still says so.
        if not scraper.health_path.exists():
            scraper._write_health(attempted=0, failed=0, tweets=0, accounts=0, capped=False)
        health = json.loads(scraper.health_path.read_text())
        health.update({"status": "down", "errors": (health.get("errors") or []) + [str(e)]})
        scraper.health_path.write_text(json.dumps(health, indent=2))
        print(f"  [x] X INGESTION DOWN: {e}")
        return 1
    if not tweets:
        # Don't write an empty dump: it would shadow a good one from earlier the
        # same day and turn a re-run into a silently emptied digest. No new
        # tweets is a fact about the poll, not a reason to discard the window.
        print("  [x] no NEW tweets since the last run -- leaving the previous "
              "x_data_*.json in place")
        return 0
    scraper.save_to_json(tweets)
    print(f"  [x] {len(tweets)} new tweets from "
          f"{len({t['handle'] for t in tweets})} accounts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
