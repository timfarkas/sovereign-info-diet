#!/usr/bin/env python3
"""The orchestrator: one scrape, N digests.

This is what `llm_summarizer.py`'s __main__ used to be, with the AI digest's
hardcoded assumptions lifted into `topics.py`. The shape of a run:

    load the shared corpora once  (X dumps, reddit dumps, feed dump)
      for each topic that is DUE:
        select its slice of each corpus
        fetch the pages its items link to
        summarize with its own prompt and its own budget
        save, mail, record stats
        note that it ran

Properties that matter more than the code:

* **One scrape, not four.** X costs twitterapi.io credits per tweet, so a
  per-topic scrape would multiply the only genuinely expensive leg by four to
  fetch largely the same tweets. The topics select from shared dumps instead.
* **Topics are isolated.** Each one runs inside its own try/except. A prompt
  that trips a content filter, a feed that returns garbage, a mail command that
  fails -- none of it can stop the remaining topics, and the AI digest goes
  FIRST so it is never downstream of a new topic's bug.
* **The AI digest is bit-for-bit the old path.** Same 20h freshness guard, same
  window, same subreddits, same prompt, same filename, same subject line, same
  `digest` stats history. The regression test for this is `test_topics.py`; the
  argument for it is that there is only one code path and the AI topic's row in
  `topics.py` reproduces the old constants.
* **Nothing is mailed on empty.** A topic with no material at all is skipped
  loudly rather than mailed as an empty digest -- but it still writes a stats
  row, because "ran and found nothing" is information and a missing row is not.
"""

import glob
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rss_scraper
import topics
from link_fetcher import LinkFetcher, readable, unreadable
from llm_summarizer import (LLMSummarizer, digest_stats, health_banner,
                            read_health, wall_appendix)

EXTRACTS = Path("extracts")


# =============================================================================
# shared corpora
# =============================================================================

def dumps_within(pattern: str, hours: float) -> List[Path]:
    """Dump files touched in the last `hours`, oldest first."""
    cutoff = time.time() - hours * 3600
    hits = [Path(f) for f in glob.glob(pattern)]
    return sorted((f for f in hits if f.stat().st_mtime >= cutoff),
                  key=lambda f: f.stat().st_mtime)


def load_buckets(pattern: str, hours: float) -> Tuple[List[Tuple[float, Path, List[Dict]]], str]:
    """Every dump inside `hours`, each with its age, oldest first. One disk pass.

    Returned as buckets rather than one flat list because each topic applies its
    OWN lookback afterwards: the AI digest must keep seeing exactly one night of
    tweets even on a run where a 3-day topic pulled three nights off disk. Merge
    them here and that distinction is gone.
    """
    buckets, notes = [], []
    for f in dumps_within(pattern, hours):
        age_h = (time.time() - f.stat().st_mtime) / 3600
        try:
            batch = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as e:
            notes.append(f"{f.name} unreadable ({e})")
            continue
        buckets.append((age_h, f, batch))
        notes.append(f"{f.name} ({age_h:.1f}h, {len(batch)})")
    if not buckets:
        notes.append(f"no dump within {hours:.0f}h")
    return buckets, ", ".join(notes)


def corpus_for(buckets, hours: float, key) -> Tuple[List[Dict], Optional[str]]:
    """One topic's slice of a source: the dumps inside ITS lookback, deduplicated.

    A topic with a lookback of a day or less reads only the newest dump, which is
    what the AI digest has always done. Anything longer reads every dump in its
    window, because a 3-day topic has to see the last three nights and widening
    the X `since:` instead would re-pay twitterapi credits for tweets already on
    disk -- roughly 1.5x the nightly bill for material we own.

    The distinction is not cosmetic. The X scraper keeps a seen-ids store, so two
    dumps from the SAME night are disjoint; unioning them would hand the AI
    digest tweets an earlier run that night already mailed. Across nights the
    dumps are disjoint for the same reason, which is what makes the union safe
    for the 3-day topics.
    """
    mine = [b for b in buckets if b[0] <= hours]
    if not mine:
        return [], None
    if hours <= 24:
        mine = mine[-1:]
    items, seen = [], set()
    for _, _, batch in mine:
        for it in batch:
            k = key(it)
            if k in seen:
                continue
            seen.add(k)
            items.append(it)
    return items, str(mine[-1][1].resolve())


def tweet_key(t: Dict):
    return t.get("id") or t.get("url") or (t.get("handle"), t.get("text", "")[:120])


def post_key(p: Dict):
    return (p.get("subreddit"), p.get("title"))


def load_shared(max_hours: float) -> Dict:
    """Read every dump any selected topic might want, once."""
    x_buckets, x_note = load_buckets("extracts/x_data_*.json", max_hours)
    reddit_buckets, reddit_note = load_buckets("extracts/reddit_data_*.json",
                                               max_hours)
    feed_items, feed_file = rss_scraper.latest_dump(EXTRACTS,
                                                    max_age_hours=max_hours)
    feed_note = (f"{Path(feed_file).name} ({len(feed_items)} items)" if feed_file
                 else f"no fresh rss dump within {max_hours:.0f}h")

    print(f"[corpus] X dumps: {x_note}")
    print(f"[corpus] reddit dumps: {reddit_note}")
    print(f"[corpus] feed: {feed_note}")
    return {
        "x_buckets": x_buckets, "x_note": x_note,
        "reddit_buckets": reddit_buckets, "reddit_note": reddit_note,
        "feed_items": feed_items, "feed_note": feed_note, "feed_file": feed_file,
    }


# =============================================================================
# one topic
# =============================================================================

def run_topic(t: topics.Topic, shared: Dict, *, mail: bool = True) -> Dict:
    """Summarize, save, mail and record one topic. Returns a small result dict.

    Raises nothing the caller has to care about except genuinely unexpected
    programming errors -- `main` catches those per topic so one bad topic cannot
    take the others down.
    """
    tweets_all, x_file = corpus_for(shared["x_buckets"], t.lookback_hours, tweet_key)
    posts_all, reddit_file = corpus_for(shared["reddit_buckets"],
                                        t.lookback_hours, post_key)
    tweets = t.select_tweets(tweets_all)
    posts = t.select_posts(posts_all)

    feed_items = []
    if t.feeds or t.feed_urls:
        # The feed dump covers RSS_WINDOW_DAYS (wider than any topic, so a missed
        # cron does not lose a day); narrow it to this topic's own window first.
        fresh = [i for i in shared["feed_items"]
                 if rss_scraper.within_window(i, t.window_days)]
        feed_items = t.select_feed_items(fresh)

    x_fresh, reddit_fresh = bool(x_file), bool(reddit_file)

    print(f"\n{'=' * 70}\n[{t.key}] {t.name}: {len(feed_items)} feed items, "
          f"{len(tweets)} X posts, {len(posts)} reddit posts\n{'=' * 70}")

    if not (tweets or posts or feed_items):
        print(f"[{t.key}] nothing in any source -- not mailing an empty digest")
        return {"key": t.key, "ok": False, "mailed": False,
                "reason": "no material in any source"}

    from config import LINK_FETCH_ENABLED, LINK_FETCH_MAX_CHARS
    pages, link_stats = [], {}
    if LINK_FETCH_ENABLED:
        # a fetch leg that dies must not take the digest with it
        try:
            fetcher = LinkFetcher()
            pages = fetcher.enrich(tweets, posts, limit=t.link_fetch_max_pages,
                                   max_chars=LINK_FETCH_MAX_CHARS,
                                   feed_items=feed_items)
            link_stats = dict(fetcher.stats)
        except Exception as e:
            print(f"[links] enrichment failed, continuing without it: {e}")

    summarizer = LLMSummarizer()
    summary = summarizer.summarize_posts(posts, tweets, pages, topic=t,
                                         feed_items=feed_items)
    summary += wall_appendix(pages)

    banner = health_banner()
    if dropped := getattr(summarizer, "tweets_dropped", 0):
        banner += (f'<p><strong>⚠ Corpus trimmed to fit the budget</strong> — the '
                   f'{dropped} lowest-engagement X posts of {len(tweets)} '
                   f'were left out of the analysis. Raise this topic\'s '
                   f'cost_ceiling_usd or prune X_SEED_ACCOUNTS.</p>')
    if not x_fresh:
        banner += (f'<p><strong>⚠ No fresh X data</strong> — {shared["x_note"]}.</p>')
    if not reddit_fresh:
        banner += (f'<p><strong>⚠ No fresh Reddit data</strong> — '
                   f'{shared["reddit_note"]}.</p>')
    if (t.feeds or t.feed_urls) and not shared["feed_file"]:
        banner += (f'<p><strong>⚠ No fresh feed data</strong> — '
                   f'{shared["feed_note"]}. This digest is social-only.</p>')

    output_file = summarizer.save_summary(
        summary, len(posts), tweets_analyzed=len(tweets),
        feeds_analyzed=len(feed_items), topic=t, banner=banner,
        footer=(f" — ${summarizer.last_cost:.3f}"
                f", {len(readable(pages))} pages read"
                f", {len(unreadable(pages))} walled"
                if summarizer.last_cost else ""))

    failed = summary.startswith("Failed to generate summary")
    mailed = False
    if mail and not failed:
        # Imported here, not at module scope: send_notification calls
        # load_dotenv() at import time, and a --dry-run must not depend on mail
        # config being present at all.
        from send_notification import send_email
        subject = f"{t.subject} - {datetime.now().strftime('%Y-%m-%d')}"
        mailed = send_email(subject, Path(output_file).read_text())
    elif failed:
        print(f"[{t.key}] the model call failed -- not mailing an error as a digest")

    record_stats(t, posts, tweets, pages, feed_items, summarizer, summary,
                 shared, output_file, link_stats,
                 x_fresh=x_fresh, reddit_fresh=reddit_fresh,
                 x_file=x_file, reddit_file=reddit_file)

    return {"key": t.key, "ok": not failed, "mailed": mailed,
            "cost": summarizer.last_cost, "output": output_file,
            "counts": {"feeds": len(feed_items), "x": len(tweets),
                       "reddit": len(posts), "pages": len(readable(pages))}}


def record_stats(t, posts, tweets, pages, feed_items, summarizer, summary,
                 shared, output_file, link_stats, *, x_fresh, reddit_fresh,
                 x_file, reddit_file) -> None:
    """History row plus a re-render. Never allowed to break a run.

    Same contract the AI digest already had: wrapped, but it prints the
    traceback instead of swallowing it, so a broken page is loud in the log
    rather than invisible.
    """
    try:
        # html_status/ is a repo-root sibling of this folder, not a dependency
        # installed anywhere on sys.path.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from html_status import stats_page, stats_store
        row = digest_stats(
            posts, tweets, pages, summarizer, summary, topic=t,
            feed_items=feed_items,
            x_health=read_health("extracts/x_health.json"),
            reddit_health=read_health("extracts/reddit_health.json"),
            rss_health=read_health("extracts/rss_health.json"),
            x_fresh=x_fresh, reddit_fresh=reddit_fresh,
            rss_fresh=bool(shared["feed_file"]),
            x_note=shared["x_note"], reddit_note=shared["reddit_note"],
            rss_note=shared["feed_note"],
            x_file=x_file, reddit_file=reddit_file,
            rss_file=shared["feed_file"],
            output_file=output_file, link_stats=link_stats,
        )
        stats_store.record(t.stats_kind, row)      # unconditional: cheap history
        written = stats_page.render_digest_page(t.key)   # opt-in: HTML_SERVE_DIR
        if written:
            print(f"[stats] wrote {written}")
        else:
            print("[stats] recorded to data/run_stats -- page rendering is off "
                  "(set HTML_SERVE_DIR in .env to turn it on)")
    except Exception:
        import traceback
        print("[stats] status page failed -- the digest itself is unaffected:")
        traceback.print_exc()


# =============================================================================
# the run
# =============================================================================

def main(argv: List[str] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    only = [a.split("=", 1)[1] for a in argv if a.startswith("--only=")]
    only += [argv[i + 1] for i, a in enumerate(argv)
             if a == "--only" and i + 1 < len(argv)]
    force = "--force" in argv
    mail = "--no-mail" not in argv and "--dry-run" not in argv
    list_only = "--list" in argv

    wanted = [t for t in topics.all_topics() if not only or t.key in only]
    if only and not wanted:
        print(f"[run] no topic matches {only}; "
              f"have {[t.key for t in topics.all_topics()]}")
        return 2

    state = topics.load_state()
    now = datetime.now(timezone.utc)

    if list_only:
        for t in topics.all_topics():
            is_due, why = topics.due(t, state, now)
            print(f"{t.key:12s} {'DUE ' if is_due else 'wait'}  every {t.every_n_days}d"
                  f"  {len(t.subreddits)} subs  {len(t.feed_urls)} feeds  "
                  f"{len(t.feeds)} pubs  -- {why}")
        return 0

    # Read the corpora at the widest lookback any selected topic wants, then let
    # each topic narrow it. One pass over the disk, no matter how many topics.
    max_hours = max((t.lookback_hours for t in wanted), default=24)
    shared = load_shared(max_hours)

    if not (shared["x_buckets"] or shared["reddit_buckets"]
            or shared["feed_items"]):
        print("[run] no input from any source -- refusing to mail empty digests")
        return 1

    results, ran = [], False
    for t in wanted:
        is_due, why = topics.due(t, state, now)
        if not is_due and not force:
            print(f"\n[{t.key}] skipping: {why}")
            continue
        print(f"\n[{t.key}] running: {why}" + (" (forced)" if force and not is_due
                                               else ""))
        ran = True
        try:
            res = run_topic(t, shared, mail=mail)
        except Exception as e:
            # One topic's failure is one topic's failure. Print it in full --
            # this job fails silently into a logfile and the traceback is the
            # only thing that will ever explain a missing digest.
            import traceback
            print(f"[{t.key}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            res = {"key": t.key, "ok": False, "mailed": False, "reason": str(e)}
        results.append(res)
        # A topic only counts as having run when it produced a digest. A failure
        # must leave it due again tomorrow rather than benched for every_n_days.
        state = topics.record_run(state, t, ok=bool(res.get("ok")), now=now)
        topics.save_state(state)

    print(f"\n{'=' * 70}\n[run] summary")
    for r in results:
        counts = r.get("counts") or {}
        bits = " ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {r['key']:12s} ok={r['ok']} mailed={r['mailed']} "
              f"cost={r.get('cost') or 0:.3f} {bits}"
              + (f"  [{r['reason']}]" if r.get("reason") else ""))
    total = sum((r.get("cost") or 0) for r in results)
    print(f"  total model cost this run: ${total:.3f}")

    if not ran:
        print("[run] nothing was due")
        return 0
    # Non-zero only if EVERY topic that ran failed: a partial run still
    # delivered a digest, and run_pipeline.sh should not report that as a
    # failed night.
    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
