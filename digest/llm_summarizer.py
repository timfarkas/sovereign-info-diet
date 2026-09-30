#!/usr/bin/env python3

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

import re

from link_fetcher import LinkFetcher, format_pages, readable, unreadable
from x_scraper import is_blocked_link
import state_notes

load_dotenv(os.getenv("DOTENV_PATH") or None)


BLOCKED_HREF = re.compile(r'<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.S | re.I)


def strip_blocked_links(html: str) -> str:
    """Replace any <a> pointing at a blocked platform with its plain text.

    Belt to the prompt's braces: the model is told not to emit x.com/reddit.com
    links, and this guarantees it. Tim's devices block those domains, so one
    leaking through is a dead link in his inbox.
    """
    def repl(m):
        return m.group(2) if is_blocked_link(m.group(1)) else m.group(0)
    return BLOCKED_HREF.sub(repl, html)


def by_engagement(tweets: List[Dict]) -> List[Dict]:
    return sorted(tweets, key=lambda t: (t.get("likes") or 0) + (t.get("retweets") or 0),
                  reverse=True)


def format_feed_items(items: List[Dict]) -> str:
    """Render subscribed-feed items for the prompt.

    Deliberately flat and labelled rather than JSON: the model reads this as
    prose, and a `link:` line it can copy verbatim is what keeps it from
    inventing URLs. Blocked links are dropped here as well as on the way out --
    a feed item whose source_url is a t.co redirect is not a citable source.
    """
    out = ""
    for i, it in enumerate(items or [], 1):
        out += f"\n{i}. {it.get('title') or '(untitled)'}\n"
        by = " / ".join(x for x in (it.get("site"), it.get("author")) if x)
        if by:
            out += f"   {by}\n"
        when = (it.get("published") or "")[:10]
        if when:
            out += f"   published: {when}\n"
        url = it.get("url") or ""
        if url and not is_blocked_link(url):
            out += f"   link: {url}\n"
        if it.get("summary"):
            out += f"   summary: {it['summary']}\n"
    return out


def format_tweets(tweets: List[Dict], limit: int = None) -> str:
    """Render tweets for the prompt, highest-engagement first."""
    if limit is None:
        try:
            from config import X_MAX_TWEETS_IN_PROMPT as limit
        except ImportError:
            limit = None
    ranked = by_engagement(tweets)[:limit]
    ranked.sort(key=lambda t: t.get("id") or "", reverse=True)
    out = ""
    for i, t in enumerate(ranked, 1):
        eng = f"{t.get('likes') or 0}♥ {t.get('retweets') or 0}RT"
        out += f"\n{i}. @{t.get('handle')} ({t.get('author_name')}) [{eng}] {t.get('created_at')}\n"
        out += f"   {(t.get('text') or '').strip()}\n"
        if t.get("quoted_text"):
            out += f"   quoting: {t['quoted_text'].strip()[:200]}\n"
        if t.get("external_links"):
            out += f"   external links: {', '.join(t['external_links'][:3])}\n"
    return out


def health_banner(path: str = "extracts/x_health.json") -> str:
    """An HTML notice when the X feed is sick, empty string when it is fine."""
    f = Path(path)
    if not f.exists():
        return ('<p><strong>⚠ X ingestion did not run</strong> — no health file at '
                f'{path}. The digest below is Reddit-only.</p>')
    h = json.loads(f.read_text())
    if h.get("status") == "healthy":
        return ""
    bits = [f"status <strong>{h.get('status')}</strong>",
            f"{h.get('batches_failed')}/{h.get('batches_attempted')} batches failed",
            f"{h.get('tweets_new')} new tweets"]
    if h.get("tweet_cap_hit"):
        bits.append("per-run tweet cap hit")
    for n in (h.get("notes") or []):
        bits.append(n)
    err = (h.get("errors") or [])
    tail = f"<br><em>last error: {err[-1][:300]}</em>" if err else ""
    return ("<p><strong>⚠ X ingestion needs a look</strong> — " + "; ".join(bits)
            + f". Checked {h.get('checked_at')}.{tail}</p>")


def usage_breakdown(usage) -> Dict[str, Any]:
    """Token counts and dollars for one completion, as plain data.

    Split out of report_cost so the status page can record the same numbers the
    log prints, without report_cost having to change what it returns.
    """
    from config import SUMMARY_MODEL_PRICE
    inp = getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(usage, "completion_tokens_details", None),
                        "reasoning_tokens", 0) or 0
    p_in, p_cached, p_out = SUMMARY_MODEL_PRICE
    return {
        "input_tokens": inp,
        "cached_tokens": cached,
        "output_tokens": out,
        "reasoning_tokens": reasoning,
        "cost_usd": ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6,
    }


def report_cost(model: str, usage) -> float:
    """Print what this call cost and flag it if it blew the ceiling.

    Tim set the budget per digest, so the digest should say what it cost rather
    than leaving him to reconstruct it from an invoice a month later.
    """
    from config import SUMMARY_COST_CEILING_USD, SUMMARY_SERVICE_TIER
    u = usage_breakdown(usage)
    inp, out = u["input_tokens"], u["output_tokens"]
    cached, reasoning = u["cached_tokens"], u["reasoning_tokens"]
    cost = u["cost_usd"]
    print(f"Cost: ${cost:.4f} on {model} ({SUMMARY_SERVICE_TIER} tier) "
          f"({inp} input [{cached} cached] / {out} output [{reasoning} reasoning] tokens)")
    if cost > SUMMARY_COST_CEILING_USD:
        print(f"⚠ OVER the ${SUMMARY_COST_CEILING_USD:.2f}/digest ceiling by "
              f"${cost - SUMMARY_COST_CEILING_USD:.4f} -- drop to gpt-6-luna, or trim "
              f"LINK_FETCH_MAX_PAGES / LINK_FETCH_MAX_CHARS")
    return cost


def read_health(path: str) -> Dict[str, Any]:
    """A leg's health file, or {} if it never wrote one. Never raises."""
    f = Path(path)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def twitter_spend(x_health) -> Dict[str, Any]:
    """What the X leg cost, in credits and -- if the rate is configured -- dollars.

    The rate is a config knob and flagged unverified on the page, because
    twitterapi.io exposes a balance and never a price. Credits are measured;
    dollars are a conversion.
    """
    from config import TWITTERAPI_CREDIT_SETTLE_SECONDS, TWITTERAPI_CREDITS_PER_USD
    used = (x_health or {}).get("credits_used")
    rate = TWITTERAPI_CREDITS_PER_USD
    return {
        "credits_used": used,
        "credits_remaining": (x_health or {}).get("credits_after"),
        "credits_per_usd": rate,
        "settle_seconds": TWITTERAPI_CREDIT_SETTLE_SECONDS,
        "usd": (used / rate) if (used is not None and rate) else None,
        "rate_verified": False,
    }


def digest_stats(posts, tweets, pages, summarizer, summary, *, x_health=None,
                 reddit_health=None, x_fresh=True, reddit_fresh=True,
                 x_note="", reddit_note="", output_file=None,
                 x_file=None, reddit_file=None,
                 link_stats=None, topic=None, feed_items=None,
                 rss_health=None, rss_fresh=True, rss_note="",
                 rss_file=None) -> Dict[str, Any]:
    """One run's worth of numbers, as a plain dict, for the status page.

    Pure: it reads what the run already computed and returns data. Nothing here
    touches the network, and nothing here is allowed to be the reason a digest
    does not get mailed.

    Every new argument defaults to the AI digest's old behaviour, so a row for
    the AI topic has the same keys and the same values it had before topics
    existed -- which is what lets the existing ai-digest page keep reading its
    existing history without a migration.
    """
    from collections import Counter
    from config import (LINK_FETCH_MAX_PAGES, SORT_BY, SUMMARY_SERVICE_TIER)
    import topics as topics_mod
    topic = topic or topics_mod.topic("ai")
    SUBREDDITS = list(topic.subreddits)
    SUMMARY_COST_CEILING_USD = topic.cost_ceiling_usd
    TIME_HORIZON_DAYS = topic.window_days
    x_health = x_health if x_health is not None else {}
    reddit_health = reddit_health if reddit_health is not None else {}
    rss_health = rss_health if rss_health is not None else {}
    feed_items = feed_items or []
    usage = dict(summarizer.last_usage or {})
    failed = summary.startswith("Failed to generate summary")

    problems = []
    if not x_fresh:
        problems.append(f"no fresh X data ({x_note})")
    if not reddit_fresh:
        problems.append(f"no fresh reddit data ({reddit_note})")
    if topic.feeds and not rss_fresh:
        problems.append(f"no fresh feed data ({rss_note})")
    if rss_health.get("status") not in (None, "healthy"):
        problems.append(f"feed ingestion reported status {rss_health.get('status')}")
    for url, err in (rss_health.get("direct", {}).get("feeds_failed") or {}).items():
        problems.append(f"feed {url} failed: {err}")
    if x_health.get("status") not in (None, "healthy"):
        problems.append(f"X ingestion reported status {x_health.get('status')}")
    if reddit_health.get("failed"):
        problems.append("subreddits that failed to scrape: "
                        + ", ".join(reddit_health["failed"]))
    if summarizer.tweets_dropped:
        problems.append(f"{summarizer.tweets_dropped} X posts dropped to fit the "
                        f"${SUMMARY_COST_CEILING_USD:.2f} budget")
    if usage.get("cost_usd", 0) > SUMMARY_COST_CEILING_USD:
        problems.append(f"cost ${usage['cost_usd']:.4f} is over the "
                        f"${SUMMARY_COST_CEILING_USD:.2f} ceiling")
    if failed:
        problems.append("the model call failed -- the digest carries an error instead "
                        "of a summary")

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "ok": not failed,
        "window_days": TIME_HORIZON_DAYS,
        "problems": problems,
        "topic": {
            "key": topic.key,
            "name": topic.name,
            "every_n_days": topic.every_n_days,
            "priorities": topic.priorities_blurb,
        },
        "reddit": {
            "total": len(posts),
            "by_subreddit": dict(Counter(p.get("subreddit") for p in posts)),
            "configured": list(SUBREDDITS),
            "failed": reddit_health.get("failed") or [],
            "comments": sum(len(p.get("comments") or []) for p in posts),
            "sort_by": SORT_BY,
            "fresh": reddit_fresh,
            "note": reddit_note,
            "file": str(reddit_file) if reddit_file else None,
            "health": reddit_health,
        },
        "feeds": {
            "total": len(feed_items),
            "by_feed": dict(Counter(i.get("feed") or i.get("site") or "(unknown)"
                                    for i in feed_items).most_common()),
            "by_source": dict(Counter(i.get("source") for i in feed_items)),
            "with_link": sum(1 for i in feed_items if i.get("url")),
            "fresh": rss_fresh,
            "note": rss_note,
            "file": str(rss_file) if rss_file else None,
            "health": rss_health,
        },
        "x": {
            "total": len(tweets),
            "by_account": dict(Counter(t.get("handle") for t in tweets)),
            "fresh": x_fresh,
            "note": x_note,
            "file": str(x_file) if x_file else None,
            "health": x_health,
            "spend": twitter_spend(x_health),
        },
        "links": {
            "attempted": len(pages),
            "limit": LINK_FETCH_MAX_PAGES,
            "read": len(readable(pages)),
            "walled": len(unreadable(pages)),
            "cache_hits": (link_stats or {}).get("cached"),
            "by_status": dict(Counter(
                p.get("status", "?").split(":")[0].strip() if p.get("status") != "ok"
                else "read" for p in pages)),
        },
        "llm": dict(usage, **{
            "model": summarizer.model,
            "tier": SUMMARY_SERVICE_TIER,
            "prompt_chars": summarizer.prompt_chars,
            "summary_chars": len(summary),
            "tweets_dropped": summarizer.tweets_dropped,
            "ceiling": SUMMARY_COST_CEILING_USD,
        }),
        "output": {"file": str(output_file) if output_file else None},
    }


class LLMSummarizer:
    def __init__(self, model_name: str = None):
        """Initialize the LLM summarizer with OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        if model_name is None:
            from config import SUMMARY_MODEL
            model_name = SUMMARY_MODEL
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.last_cost = None
        self.last_usage = {}
        self.tweets_dropped = 0
        self.prompt_chars = 0
        
    def create_summary_prompt(self, posts: List[Dict], tweets: List[Dict] = None,
                              pages: List[Dict] = None, *, topic=None,
                              feed_items: List[Dict] = None) -> str:
        """Create the summarization prompt for one topic.

        `topic=None` means the AI digest, which is how every pre-multi-topic
        caller (and every existing test) still gets exactly the prompt it got
        before: config.SUMMARY_PROMPT_TEMPLATE, the AI window, no feed items.

        The two template families use different slot names -- the AI template
        predates the others and says {TIME_HORIZON_DAYS}, the shared skeleton in
        prompts.py says {window_days} and also wants {feed_content}. Both name
        sets are passed to .format() every time, and str.format ignores keys a
        template does not reference. So one build() serves all four topics with
        no branching, and adding a slot to the skeleton cannot break the AI
        prompt.
        """
        import topics as topics_mod
        topic = topic or topics_mod.topic("ai")

        from config import REDDIT_SELFTEXT_CHARS, REDDIT_COMMENT_CHARS

        def clip(text, n):
            return text[:n] + "..." if len(text) > n else text

        # format all posts into a digest
        posts_content = ""
        for i, post in enumerate(posts, 1):
            posts_content += f"\n{i}. {post['title']} [{post['score']} pts]\n"
            posts_content += f"   r/{post['subreddit']}\n"

            # a link post's destination -- its content shows up in SOURCE C
            if post.get('url') and not is_blocked_link(post['url']):
                posts_content += f"   links to: {post['url']}\n"

            if post.get('selftext'):
                posts_content += f"   Text: {clip(post['selftext'], REDDIT_SELFTEXT_CHARS)}\n"

            # include top 3 comments for context
            if post.get('comments'):
                posts_content += "   Top comments:\n"
                for comment in post['comments'][:3]:
                    posts_content += (f"   - [{comment['score']}pts] "
                                      f"{clip(comment['body'], REDDIT_COMMENT_CHARS)}\n")
        
        from config import CHARS_PER_TOKEN, SUMMARY_MODEL_PRICE
        ceiling = topic.cost_ceiling_usd

        feed_content = format_feed_items(feed_items or [])

        def build(tws):
            return topic.prompt.format(
                TIME_HORIZON_DAYS=topic.window_days,
                window_days=topic.window_days,
                posts_content=posts_content or "(no reddit posts in this window)",
                tweets_content=format_tweets(tws) or "(no X posts in this window)",
                pages_content=format_pages(pages or []),
                feed_content=feed_content or "(no feed items in this window)",
                prior_notes=state_notes.load(topic.key),
                longrunning=state_notes.load_longrunning(topic.key),
            )

        # Keep EVERY tweet by default. If the corpus has grown past what the
        # budget can pay for, drop the lowest-engagement tail -- but record how
        # many, and say so in the digest. A silent trim is the bug Tim caught;
        # an announced one is a budget working as intended.
        kept = by_engagement(tweets or [])
        price_in = SUMMARY_MODEL_PRICE[0]
        # leave a tenth of the ceiling for output tokens
        budget_chars = int((ceiling * 0.9 / price_in) * 1e6 * CHARS_PER_TOKEN)
        prompt = build(kept)
        self.tweets_dropped = 0
        while len(prompt) > budget_chars and len(kept) > 25:
            drop = max(1, int(len(kept) * 0.1))
            kept = kept[:-drop]
            self.tweets_dropped += drop
            prompt = build(kept)
        if self.tweets_dropped:
            print(f"⚠ prompt over the ${ceiling:.2f} budget: dropped "
                  f"the {self.tweets_dropped} lowest-engagement tweets of "
                  f"{len(tweets or [])} to fit")
        return prompt
    
    def summarize_posts(self, posts: List[Dict], tweets: List[Dict] = None,
                        pages: List[Dict] = None, *, topic=None,
                        feed_items: List[Dict] = None) -> str:
        """Generate one topic's summary. `topic=None` is the AI digest."""

        import topics as topics_mod
        topic = topic or topics_mod.topic("ai")

        tweets, pages, feed_items = tweets or [], pages or [], feed_items or []
        print(f"\n[{topic.key}] Analyzing {len(posts)} reddit posts + "
              f"{len(tweets)} X posts + {len(feed_items)} feed items "
              f"+ {len(readable(pages))} read pages "
              f"({len(unreadable(pages))} walled but still linkable) "
              f"with {self.model}...")

        prompt = self.create_summary_prompt(posts, tweets, pages, topic=topic,
                                            feed_items=feed_items)
        self.prompt_chars = len(prompt)
        print(f"Prompt is {len(prompt):,} chars")
        
        from config import SUMMARY_SERVICE_TIER
        messages = [
            {"role": "system", "content": topic.system},
            {"role": "user", "content": prompt},
        ]
        # reasoning models reject temperature / max_tokens
        extra = ({} if self.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
                 else {"temperature": 0.7, "max_tokens": 1500})

        def call(tier):
            return self.client.chat.completions.create(
                model=self.model, messages=messages, service_tier=tier, **extra)

        try:
            try:
                response = call(SUMMARY_SERVICE_TIER)
            except Exception as e:
                if SUMMARY_SERVICE_TIER == "default":
                    raise
                # flex trades capacity for price; falling back beats no digest
                print(f"[{SUMMARY_SERVICE_TIER} tier unavailable: {e} -- "
                      f"retrying on standard tier at full price]")
                response = call("default")
            summary = strip_blocked_links(response.choices[0].message.content.strip())
            self.last_cost = report_cost(self.model, response.usage)
            self.last_usage = usage_breakdown(response.usage)

            print("Summary generated successfully")
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Failed to generate summary: {str(e)}"
    
    def save_summary(self, summary: str, posts_analyzed: int, filename: str = None,
                     tweets_analyzed: int = 0, banner: str = "", footer: str = "",
                     *, topic=None, feeds_analyzed: int = 0):
        """Save the summary to a file.

        The AI digest keeps writing `summary_<date>.html` and keeps its old
        `<h1>AI Digest - ...` heading and its old counts line -- that file is
        what `send_notification.send_summary_email()` reaches for by hand and
        what the last few months of history look like. New topics are namespaced
        `summary_<key>_<date>.html` so the two never collide.
        """
        import topics as topics_mod
        topic = topic or topics_mod.topic("ai")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            stem = "summary" if topic.key == "ai" else f"summary_{topic.key}"
            filename = f"{stem}_{timestamp}.html"

        counts = f"Analyzed {tweets_analyzed} X posts and {posts_analyzed} Reddit posts"
        if topic.key != "ai":
            counts = (f"Analyzed {feeds_analyzed} feed articles, {tweets_analyzed} "
                      f"X posts and {posts_analyzed} Reddit posts")

        output_dir = Path("extracts/summaries")
        output_dir.mkdir(exist_ok=True, parents=True)

        filepath = output_dir / filename

        # add header
        full_content = f"""<h1>{topic.subject} - {datetime.now().strftime("%Y-%m-%d")}</h1>
<p><em>{counts} from the past {topic.window_days} days</em></p>
{banner}<hr>
{summary}
<hr>
<p><em>Generated at {datetime.now().strftime("%I:%M %p")} by {self.model}{footer}</em></p>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        print(f"Summary saved to {filepath}")
        return str(filepath)


if __name__ == "__main__":
    # The orchestration that used to live here moved to digest_run.py when the
    # pipeline went multi-topic -- there is one code path now, and the AI digest
    # is a row in topics.py rather than a set of constants inlined below. Kept as
    # an alias because `python llm_summarizer.py` is what every runbook, cron
    # line and muscle memory says, and it still means "produce the AI digest".
    import digest_run

    raise SystemExit(digest_run.main(["--only", "ai"]))
