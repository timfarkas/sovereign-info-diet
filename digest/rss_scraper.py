#!/usr/bin/env python3
"""The feed leg: what Tim's subscriptions published in the last few days.

TWO PATHS, and the second one is not a fallback -- it is the reason this module
can be trusted.

  1. Readwise Reader (primary). Everything he actually subscribes to already
     lands there, RSS and newsletters both, with the title/author/site/summary
     already normalised and the tracking junk already stripped off the URL. Free
     to us and better parsed than anything we would write.
  2. Direct RSS via feedparser (always on, independent). A curated per-topic
     list of feeds fetched straight from the publisher, so the digest still has
     material if the Readwise key is missing, the account lapses, or v3 changes
     shape. Also how a topic gets sources he has not subscribed to.

Both paths emit the SAME item dict, they are merged and deduplicated by URL, and
everything downstream is blind to which path an item came from.

WHAT "RSS FEEDS I'M SUBSCRIBED TO" MEANS HERE: `location=feed` with category in
`RSS_CATEGORIES`, which is `rss` AND `email`. That second one is a deliberate
reading of the brief rather than a literal one -- measured 2026-09-29, his
geopolitics and markets sources (Money Stuff, ChinaTalk, Noahpinion,
SemiAnalysis, Sentinel) are all `category=email` newsletters forwarded into
Reader, and a literal rss-only filter would have handed the geopolitics digest
Bloomberg and Reuters headlines while dropping every piece of actual analysis.

THE FEEDBACK LOOP, closed here on purpose: his own digests are auto-forwarded
into the Reader feed, so "AI Digest - 2026-09-28" by Meta Minsky is sitting in
the corpus this module reads. Left alone, tomorrow's digest would summarise
yesterday's digest, and the summary of a summary would compound every night.
Excluded at INGEST rather than at routing, so no topic can ever see one however
the routing changes later.

The body text is NOT here. Measured 2026-09-29 against the live API: the v3
list response populates `summary` on 125 of 126 feed items (median 259 chars)
and `content` on ZERO of them. So Readwise gives clean metadata plus an
abstract, and `link_fetcher.py` gets the actual article text from `source_url`
-- the same fetcher, with the same hygiene rules, that the AI digest already
uses on links found in tweets.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests
from dotenv import load_dotenv

# Same line as every other leg: the cron wrapper exports DOTENV_PATH so the repo
# root's single .env is found regardless of which directory the leg runs from.
load_dotenv(os.getenv("DOTENV_PATH") or None)

import topics
from config import (RSS_CATEGORIES, RSS_DIRECT_ENABLED, RSS_EXCLUDE_AUTHORS,
                    RSS_EXCLUDE_TITLE_PREFIXES, RSS_MAX_ITEMS_PER_FEED,
                    RSS_READWISE_ENABLED, RSS_SUMMARY_MAX_CHARS,
                    RSS_USER_AGENT, RSS_WINDOW_DAYS)

READWISE_LIST_URL = "https://readwise.io/api/v3/list/"
EXTRACTS = Path("extracts")


# =============================================================================
# normalisation, shared by both paths
# =============================================================================

def clean_url(url: Optional[str]) -> str:
    """Drop the query and fragment: that is where the campaign tracking lives.

    Feed URLs arrive decorated with utm_*, ?ref=, newsletter click-ids. Keeping
    them would fork one article into five distinct "items" across two paths and
    defeat the dedupe below.
    """
    if not url:
        return ""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        return ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def item(*, id, title, author, site, summary, url, published, word_count,
         category, feed, source) -> Dict:
    """The one item shape. Keyword-only so a field can never land in the wrong slot."""
    return {
        "id": str(id or url or title)[:200],
        "title": (title or "").strip()[:400],
        "author": (author or "").strip()[:200],
        "site": (site or "").strip()[:200],
        "summary": (summary or "").strip()[:RSS_SUMMARY_MAX_CHARS],
        "url": clean_url(url),
        "published": published or "",
        "word_count": word_count,
        "category": category,
        "feed": (feed or "").strip()[:200],
        "source": source,
    }


def is_self_digest(it: Dict) -> bool:
    """One of our own digests, forwarded back into the feed. See module docstring."""
    author = (it.get("author") or "").lower()
    title = (it.get("title") or "").lower()
    if any(a.lower() in author for a in RSS_EXCLUDE_AUTHORS if a):
        return True
    return any(title.startswith(p.lower()) for p in RSS_EXCLUDE_TITLE_PREFIXES if p)


def within_window(it: Dict, days: int, now: datetime = None) -> bool:
    """Published inside the window. An item with NO usable date is KEPT.

    Deliberate: the alternative is silently dropping a publication whose feed
    omits a date, which is a data-loss bug that looks like a quiet week. A
    stale-but-kept item costs a few tokens and the summarizer is told to ignore
    what is not new.
    """
    now = now or datetime.now(timezone.utc)
    when = parse_date(it.get("published"))
    if when is None:
        return True
    return when >= now - timedelta(days=days)


def parse_date(raw) -> Optional[datetime]:
    if not raw:
        return None
    s = str(raw).strip()
    if s.isdigit():                       # epoch, seconds or milliseconds
        n = int(s)
        if n > 10_000_000_000:
            n //= 1000
        return datetime.fromtimestamp(n, tz=timezone.utc)
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def dedupe(items: Iterable[Dict]) -> List[Dict]:
    """By cleaned URL, first wins.

    Readwise runs first, so when the same article arrives on both paths the
    Readwise copy -- better parsed, with a word count -- is the one kept.
    Items with no URL fall back to their title so a newsletter without a web
    version still deduplicates.

    CLEANED, because that is the whole point: the direct path sees a publisher's
    bare permalink and the Readwise copy of the same article arrives decorated
    with utm_source, so keying on the raw URL lets the pair straight through.

    The title is part of the key as well, and that is a deliberate asymmetry.
    Stripping the query string collapses any site that identifies articles with
    one (`/news.php?id=5104`), and a wrongly-dropped article is invisible --
    nothing in the output says it existed. A wrongly-KEPT duplicate costs a few
    hundred tokens and is plainly visible in the digest. So errors fall that
    way: same URL and same title is a duplicate, same URL and a different title
    is two articles.
    """
    out, seen = [], set()
    for it in items:
        title = (it.get("title") or "").strip().lower()
        url = clean_url(it.get("url"))
        key = f"{url}|{title}" if url else f"title:{title}"
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# =============================================================================
# path 1: Readwise Reader
# =============================================================================

class ReadwiseFeed:
    """LIST-only Reader client. Deliberately NOT the recommender's client.

    `recommender/readwise_client.py` does sync/update/tag against a sqlite
    store and is load-bearing for a job that writes back to Reader. Sharing it
    would couple the digest to the recommender's schema and put digest bugs in
    the blast radius of the thing that tags his library. Forty lines of
    duplication is the cheaper side of that trade.
    """

    MIN_INTERVAL = 3.5     # Reader LIST allows 20/min; this stays under it

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("READWISE_API_KEY") or ""
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Token {self.token}"
        self._last = 0.0

    def _wait(self):
        gap = time.monotonic() - self._last
        if gap < self.MIN_INTERVAL:
            time.sleep(self.MIN_INTERVAL - gap)
        self._last = time.monotonic()

    def _get(self, params: Dict) -> Dict:
        for attempt in range(4):
            self._wait()
            r = self.session.get(READWISE_LIST_URL, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 60)) + 1)
                continue
            if r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("readwise list: gave up after 4 attempts")

    def fetch(self, days: int, categories=RSS_CATEGORIES) -> List[Dict]:
        """Feed items updated in the last `days`, across the given categories.

        `updatedAfter` is the only server-side time filter Reader offers, and for
        a feed item "updated" is when it arrived -- close enough to publication
        that the window is honest. Published-date filtering happens locally.
        """
        if not self.token:
            raise RuntimeError("READWISE_API_KEY is not set")
        after = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        out: List[Dict] = []
        for category in categories:
            cursor, pages = None, 0
            while True:
                params = {"location": "feed", "category": category,
                          "updatedAfter": after}
                if cursor:
                    params["pageCursor"] = cursor
                data = self._get(params)
                batch = data.get("results") or []
                out.extend(self._normalise(d, category) for d in batch)
                pages += 1
                cursor = data.get("nextPageCursor")
                print(f"[rss] readwise {category}: page {pages}, "
                      f"+{len(batch)} (total {len(out)})")
                if not cursor:
                    break
        return out

    @staticmethod
    def _normalise(d: Dict, category: str) -> Dict:
        site = d.get("site_name") or ""
        return item(
            id=d.get("id"),
            title=d.get("title"),
            author=d.get("author"),
            site=site,
            summary=d.get("summary"),
            # source_url is the publisher's URL; `url` is read.readwise.io,
            # which is useless in an email he opens on a phone.
            url=d.get("source_url") or d.get("url"),
            published=d.get("published_date") or d.get("created_at"),
            word_count=d.get("word_count"),
            category=category,
            feed=site or d.get("author") or "",
            source="readwise",
        )


# =============================================================================
# path 2: direct RSS
# =============================================================================

def fetch_feed(url: str, limit: int = RSS_MAX_ITEMS_PER_FEED) -> Tuple[List[Dict], str]:
    """One feed, straight from the publisher. Returns (items, error-or-empty).

    Never raises: one publisher having a bad afternoon must not cost the other
    eleven feeds. The error string is what lands on the status page.
    """
    import feedparser
    try:
        d = feedparser.parse(url, agent=RSS_USER_AGENT)
    except Exception as e:                      # feedparser is usually forgiving
        return [], f"{type(e).__name__}: {e}"

    status = getattr(d, "status", None)
    if status and status >= 400:
        return [], f"HTTP {status}"
    if not d.entries:
        return [], f"no entries (HTTP {status})"

    feed_title = (d.feed.get("title") if d.feed else "") or url
    out = []
    for e in d.entries[:limit]:
        published = ""
        for key in ("published_parsed", "updated_parsed"):
            tm = e.get(key)
            if tm:
                published = datetime(*tm[:6], tzinfo=timezone.utc).isoformat()
                break
        out.append(item(
            id=e.get("id") or e.get("link"),
            title=e.get("title"),
            author=e.get("author") or feed_title,
            site=feed_title,
            summary=strip_tags(e.get("summary") or ""),
            url=e.get("link"),
            published=published,
            word_count=None,
            category="rss",
            feed=feed_title,
            source="rss",
        ))
    return out, ""


def strip_tags(html: str) -> str:
    """Feed summaries arrive as HTML. Text only -- it is going into a prompt."""
    import re
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html,
                  flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'"), ("&hellip;", "...")):
        text = text.replace(a, b)
    return " ".join(text.split())


# =============================================================================
# the leg
# =============================================================================

def collect(days: int = RSS_WINDOW_DAYS, feed_urls: Optional[List[str]] = None
            ) -> Tuple[List[Dict], Dict]:
    """Run both paths, merge, and report health. Returns (items, health).

    Health is separate from the dump for the reason the reddit and X legs keep
    theirs separate: "the feed leg broke" and "nothing was published" both look
    like an empty list, and only this dict can tell them apart tomorrow.
    """
    readwise_items: List[Dict] = []
    readwise_error = ""
    if RSS_READWISE_ENABLED:
        try:
            readwise_items = ReadwiseFeed().fetch(days)
        except Exception as e:
            readwise_error = f"{type(e).__name__}: {e}"
            print(f"[rss] readwise path failed: {readwise_error}")

    direct_items: List[Dict] = []
    feed_errors: Dict[str, str] = {}
    urls = feed_urls if feed_urls is not None else topics.all_feed_urls()
    if RSS_DIRECT_ENABLED:
        for url in urls:
            got, err = fetch_feed(url)
            if err:
                feed_errors[url] = err
                print(f"[rss] {url}: {err}")
            else:
                print(f"[rss] {url}: {len(got)} items")
            direct_items.extend(got)

    raw = readwise_items + direct_items
    self_digests = [i for i in raw if is_self_digest(i)]
    kept = [i for i in raw if not is_self_digest(i)]
    fresh = [i for i in kept if within_window(i, days)]
    items = dedupe(fresh)

    from collections import Counter
    by_feed = Counter(i["feed"] or i["site"] or "(unknown)" for i in items)
    by_topic = {t.key: len(t.select_feed_items(items)) for t in topics.all_topics()}

    # A path that is switched off is not a path that failed -- the difference
    # matters on the status page, so it gets its own status value.
    if not RSS_READWISE_ENABLED:
        readwise_status = "disabled"
    elif readwise_error:
        readwise_status = "down"
    else:
        readwise_status = "healthy" if readwise_items else "empty"

    health = {
        "status": ("down" if not items else
                   "degraded" if (readwise_error or feed_errors) else "healthy"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "items": len(items),
        "readwise": {
            "status": readwise_status,
            "items": len(readwise_items),
            "error": readwise_error,
            "categories": list(RSS_CATEGORIES),
        },
        "direct": {
            "status": ("disabled" if not RSS_DIRECT_ENABLED else
                       "down" if urls and len(feed_errors) == len(urls) else
                       "degraded" if feed_errors else "healthy"),
            "items": len(direct_items),
            "feeds_configured": len(urls),
            "feeds_failed": feed_errors,
        },
        "dropped": {
            "self_digests": len(self_digests),
            "out_of_window": len(kept) - len(fresh),
            "duplicates": len(fresh) - len(items),
        },
        "by_feed": dict(by_feed.most_common()),
        "by_topic": by_topic,
    }
    return items, health


def save(items: List[Dict], health: Dict, base: Path = EXTRACTS) -> str:
    base.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / f"rss_data_{stamp}.json"
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    (base / "rss_health.json").write_text(json.dumps(health, indent=2))
    print(f"[rss] {len(items)} items -> {path}")
    return str(path)


def latest_dump(base: Path = EXTRACTS, max_age_hours: float = 30
                ) -> Tuple[List[Dict], Optional[str]]:
    """Newest rss_data_* dump, if it is fresh enough. Returns ([], None) if not.

    Mirrors the AI digest's own `_latest_fresh`: a summarizer run must never
    quietly re-summarise last week's feed because tonight's leg died.
    """
    dumps = sorted(base.glob("rss_data_*.json"))
    if not dumps:
        return [], None
    newest = dumps[-1]
    age_h = (time.time() - newest.stat().st_mtime) / 3600
    if age_h > max_age_hours:
        print(f"[rss] newest dump {newest.name} is {age_h:.1f}h old "
              f"(max {max_age_hours}h) -- treating the feed leg as absent")
        return [], None
    try:
        return json.loads(newest.read_text()), str(newest)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[rss] {newest} unreadable: {e}")
        return [], None


if __name__ == "__main__":
    days = RSS_WINDOW_DAYS
    for arg in sys.argv[1:]:
        if arg.startswith("--days="):
            days = int(arg.split("=", 1)[1])

    items, health = collect(days=days)
    save(items, health)

    print(f"\n[rss] status={health['status']} items={health['items']} "
          f"(readwise {health['readwise']['items']}, "
          f"direct {health['direct']['items']})")
    print(f"[rss] dropped: {health['dropped']}")
    print("[rss] per topic: " + ", ".join(
        f"{k}={v}" for k, v in health["by_topic"].items()))
    for feed, n in list(health["by_feed"].items())[:15]:
        print(f"       {n:4d}  {feed}")

    # Exit non-zero only when there is nothing at all: run_pipeline.sh treats a
    # failing leg as "carry on without it", and a degraded feed leg is still a
    # usable one.
    if not items:
        sys.exit(1)
