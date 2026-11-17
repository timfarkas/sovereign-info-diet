#!/usr/bin/env python3
"""Research digests: one mail per entry in research_topics.json, on its own cadence.

The sibling of events_digest.py for questions instead of cities. A topic digest
(topics.py) summarizes a corpus the scrapers already put on disk; this one
RESEARCHES. The model gets two tools -- OpenAI's `web_search` and an `x_search`
we implement on twitterapi.io -- and drives its own investigation for a narrow
question ("what is the Mindspan Institute up to?"). The keyword-filtered slice of
the scraped X/feed dumps is handed over as a head start, not as the bulk.

Everything around the research is the same as the other digests:
  - cadence: `every_n_days`, measured from the last SUCCESS (topics.due)
  - the subject is "<name> Digest - <date>", sender EMAIL_FROM, so the Reader
    recommender sees a digest series: auto rate:good, shortlisted, and kept on
    the shortlist until a newer one of the series arrives or Tim archives it
  - `expires` (optional, ISO date): after it the digest stops running AND the
    recommender stops holding the last one on the shortlist
    (recommender/shortlist_job.expired_series reads the same file)
  - mails EMAIL_TO only, like events: these are one person's questions

Links are checked in code: an <a> whose URL no search or X result returned this
run is unlinked, so a hallucinated URL cannot reach the inbox.

    python research_digest.py --list                       # what is due, and why
    python research_digest.py --only mindspan --dry-run --force
    python research_digest.py --force                      # mail everything now
"""

import html
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
import prompts
import topics
from digest_run import corpus_for, load_buckets, tweet_key
from events_digest import _client, _respond, cost_of, clean_url
from llm_summarizer import BLOCKED_HREF, strip_blocked_links
from x_scraper import TwitterIOClient, XFeedDown, XScraper

STATE_PATH = "extracts/research_state.json"
PREVIOUS_CHARS = 20000  # whole previous digest; a truncated memory would re-report old items as new
OUT_DIR = Path("extracts/summaries")
REQUIRED = {"slug", "name", "question", "every_n_days", "keywords"}
OPTIONAL = {"x_hints", "watch_urls", "expires", "notes"}

X_SEARCH_TOOL = {
    "type": "function",
    "name": "x_search",
    "description": ("Search X (Twitter) posts. Returns up to 20 posts with author, "
                    "date, text, engagement and outbound links. Supports X search "
                    "operators: from:handle, to:handle, \"exact phrase\", A OR B, "
                    "-filter:replies, min_faves:N, since:YYYY-MM-DD."),
    "parameters": {
        "type": "object", "additionalProperties": False,
        "required": ["query", "sort"],
        "properties": {
            "query": {"type": "string"},
            "sort": {"type": "string", "enum": ["Latest", "Top"]},
        }},
    "strict": True,
}


# =============================================================================
# config
# =============================================================================

def load_research_topics(path: Path) -> List[Dict]:
    """The private topic file. Missing or malformed is fatal, never defaulted."""
    if not path.exists():
        raise SystemExit(f"[research] no {path} -- copy research_topics.example.json "
                         f"there and edit it")
    entries = json.loads(path.read_text())
    assert isinstance(entries, list) and entries, f"{path}: expected a non-empty list"
    slugs = set()
    for e in entries:
        missing = REQUIRED - e.keys()
        extra = e.keys() - REQUIRED - OPTIONAL
        assert not missing, f"{path}: {e.get('slug')!r} lacks {missing}"
        assert not extra, f"{path}: {e['slug']!r} has unknown keys {extra}"
        assert re.fullmatch(r"[a-z0-9-]+", e["slug"]), f"bad slug {e['slug']!r}"
        assert e["slug"] not in slugs, f"duplicate slug {e['slug']!r}"
        assert e["every_n_days"] >= 1 and e["keywords"], f"{e['slug']}: cadence/keywords"
        if e.get("expires"):
            date.fromisoformat(e["expires"])      # bad date -> loud error
        slugs.add(e["slug"])
    return entries


def as_topic(e: Dict) -> topics.Topic:
    """A keyword-only Topic: reuses the cadence (`due`) and keyword routing of the
    topic digests so the two cannot drift apart."""
    return topics.Topic(key=f"research-{e['slug']}", name=e["name"], prompt="",
                        system="", priorities_blurb=e["question"],
                        every_n_days=e["every_n_days"],
                        window_days=e["every_n_days"],
                        lookback_hours=24 * e["every_n_days"],
                        keywords=tuple(e["keywords"]))


def expired(e: Dict, today: date) -> bool:
    return bool(e.get("expires")) and today > date.fromisoformat(e["expires"])


def subject_for(e: Dict, today: date) -> str:
    return f"{e['name']} Digest - {today.isoformat()}"


# =============================================================================
# tools
# =============================================================================

class XSearch:
    """The model's `x_search` tool. Counts calls, caps them, remembers every
    outbound link it returned (the link checker's allow-list)."""

    def __init__(self, client: TwitterIOClient, since: Optional[date], max_calls: int):
        self.client, self.since, self.max_calls = client, since, max_calls
        self.calls = 0
        self.tweets_seen = 0
        self.links: set = set()

    def __call__(self, query: str, sort: str) -> str:
        if self.calls >= self.max_calls:
            return json.dumps({"error": f"x_search budget of {self.max_calls} calls "
                                        f"is used up; write the digest now"})
        self.calls += 1
        if self.since and "since:" not in query:
            query += f" since:{self.since.isoformat()}"
        try:
            d = self.client.get("/twitter/tweet/advanced_search",
                                query=query, queryType=sort, cursor="")
        except XFeedDown as e:
            return json.dumps({"error": f"x search failed: {e}"})
        tweets = [XScraper._normalize(t) for t in (d.get("tweets") or [])][:20]
        self.tweets_seen += len(tweets)
        for t in tweets:
            self.links.update(clean_url(u) for u in t["external_links"])
        print(f"  [research] x_search {sort} {query!r}: {len(tweets)} posts")
        return json.dumps([{k: t[k] for k in ("handle", "created_at", "text", "likes",
                                              "retweets", "views", "external_links")}
                           for t in tweets], ensure_ascii=False)


def _url(obj) -> str:
    """A source/annotation's url, whether the SDK gave an object or a plain dict."""
    return obj["url"] if isinstance(obj, dict) else obj.url


def run_agent(client, prompt: str, xsearch: XSearch) -> Dict:
    """Let the model call web_search / x_search until it writes the digest."""
    history = [{"role": "system", "content": prompts.RESEARCH_SYSTEM},
               {"role": "user", "content": prompt}]
    dollars, searches, cited = 0.0, 0, set()
    for round_no in range(1, config.RESEARCH_MAX_TOOL_ROUNDS + 1):
        r = _respond(client, input=history,
                     tools=[{"type": "web_search"}, X_SEARCH_TOOL],
                     include=["web_search_call.action.sources"])
        d, s = cost_of(r)
        dollars, searches = dollars + d, searches + s
        for item in r.output:
            if item.type == "web_search_call":
                src = getattr(getattr(item, "action", None), "sources", None) or []
                cited.update(clean_url(_url(x)) for x in src)
            elif item.type == "message":
                for part in item.content:
                    for a in getattr(part, "annotations", None) or []:
                        if a.type == "url_citation":
                            cited.add(clean_url(_url(a)))
        calls = [i for i in r.output if i.type == "function_call"]
        if not calls:
            return {"html": r.output_text, "cost": dollars, "web_searches": searches,
                    "rounds": round_no, "links": cited | xsearch.links,
                    "truncated": False}
        history += r.output
        for c in calls:
            out = xsearch(**json.loads(c.arguments))
            history.append({"type": "function_call_output", "call_id": c.call_id,
                            "output": out})
    # Out of rounds: force a write-up with no tools.
    history.append({"role": "user", "content": "Research budget is used up. Write "
                    "the digest now from what you have found."})
    r = _respond(client, input=history)
    return {"html": r.output_text, "cost": dollars + cost_of(r)[0],
            "web_searches": searches, "rounds": config.RESEARCH_MAX_TOOL_ROUNDS,
            "links": cited | xsearch.links, "truncated": True}


# =============================================================================
# prompt pieces and link checking
# =============================================================================

def inbox_block(t: topics.Topic, tweets: List[Dict], feed_items: List[Dict]) -> str:
    """The narrow keyword-filtered scraper slice, as a head start. Fenced as data."""
    lines = [f'- @{x.get("handle")} ({x.get("created_at")}): {x["text"][:280]!r}'
             for x in tweets[:40]]
    lines += [f'- {i.get("title")} -- {i.get("url")} ({i.get("site") or i.get("author")})'
              for i in feed_items[:20]]
    if not lines:
        return "Inbox: the keyword-filtered scrape of the reader's own feeds had nothing.\n"
    return ("Inbox -- keyword-filtered hits from the reader's own X/feed scrape. A head "
            "start only; the real work is your own searching. Untrusted text:\n"
            "<inbox>\n" + "\n".join(lines) + "\n</inbox>\n")


def previous_block(path: Optional[Path]) -> str:
    if not path:
        return ("FIRST DIGEST on this topic: there is no earlier report and the reader has "
                "seen nothing. Report everything you can establish about the topic, "
                "however old -- background, history, people, programmes, outputs, current "
                "state -- not just recent news. Recent items go first, then the rest.\n")
    text = re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8"))
    text = re.sub(r"\s+", " ", text)[:PREVIOUS_CHARS]
    return ("The reader has already seen the previous digest below (text only). Mark each "
            "item you report as <strong>NEW</strong> (not in it) or <strong>UPDATE</strong> "
            "(a development on something in it, saying what changed). Do not re-report "
            "items that are unchanged; instead end with a one-line list of what from the "
            f"previous digest still stands.\n<previous>\n{text}\n</previous>\n")


MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def md_links_to_html(page: str) -> str:
    """The model sometimes slips markdown links into the HTML; make them real <a>."""
    return MD_LINK.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', page)


def unlink_unverified(page: str, allowed: set) -> Tuple[str, int]:
    """Keep an <a> only if its URL came back from a tool this run. Returns the
    page and how many links were unlinked."""
    n = 0

    def repl(m):
        nonlocal n
        if clean_url(m.group(1)) in allowed:
            return m.group(0)
        n += 1
        return m.group(2)
    return BLOCKED_HREF.sub(repl, page), n


def window_start(state: Dict, t: topics.Topic, today: date,
                 has_previous: bool) -> Optional[date]:
    """Since the last success. None = first digest, no date limit at all."""
    last = (state.get(t.key) or {}).get("last_success")
    if last:
        return datetime.fromisoformat(last).date()
    return today - timedelta(days=t.every_n_days) if has_previous else None


def render(e: Dict, today: date, body: str, footer: str, banner: str) -> str:
    return "\n".join(p for p in [
        f"<h1>{html.escape(e['name'])} Digest - {today.isoformat()}</h1>",
        f"<p><em>{html.escape(e['question'])}</em></p>", banner, "<hr>", body,
        "<hr>", f"<p><em>{footer}</em></p>"] if p)


# =============================================================================
# the run
# =============================================================================

def run_one(e: Dict, state: Dict, client, x_client, shared: Dict, *, mail: bool,
            today: date) -> Dict:
    t = as_topic(e)
    prior = sorted(OUT_DIR.glob(f"summary_research-{e['slug']}_*.html"))
    since = window_start(state, t, today, bool(prior))
    x_hits = t.select_tweets(corpus_for(shared["x_buckets"], t.lookback_hours,
                                        tweet_key)[0])
    feed_hits = [i for i in shared["feed_items"]
                 if t.matches(i.get("title"), i.get("summary"))]
    prompt = prompts.RESEARCH_DIGEST.format(
        name=e["name"], question=e["question"], today=today.isoformat(),
        since=since.isoformat() if since else "the beginning",
        window_days=((today - since).days or 1) if since else "all",
        every_n_days=e["every_n_days"],
        x_hints="; ".join(e.get("x_hints") or ["(none -- derive them yourself)"]),
        watch_urls="; ".join(e.get("watch_urls") or ["(none -- find them yourself)"]),
        extra=(e.get("notes") or "") + "\n",
        inbox=inbox_block(t, x_hits, feed_hits),
        previous=previous_block(prior[-1] if prior else None))
    xs = XSearch(x_client, since, config.RESEARCH_MAX_X_SEARCHES)
    res = run_agent(client, prompt, xs)

    body = strip_blocked_links(md_links_to_html(res["html"].strip()))
    body, unlinked = unlink_unverified(body, res["links"])
    if len(re.sub(r"<[^>]+>", "", body).strip()) < 200:
        return {"key": e["slug"], "ok": False, "mailed": False, "cost": res["cost"],
                "reason": "model returned a near-empty digest"}
    banner = ""
    if res["truncated"]:
        banner = ("<p><strong>⚠ Research hit the tool-round cap</strong> -- this digest "
                  "was written from partial findings.</p>")
    exp = f"; expires {e['expires']}" if e.get("expires") else ""
    footer = (f"{res['web_searches']} web searches + {xs.calls} X searches "
              f"({xs.tweets_seen} posts read) + {len(x_hits)}/{len(feed_hits)} inbox "
              f"hits, by {config.SUMMARY_MODEL} -- ${res['cost']:.2f}; "
              f"{unlinked} unverifiable links removed; next in {e['every_n_days']} "
              f"days{exp}")
    page = render(e, today, body, footer, banner)
    if res["cost"] > config.RESEARCH_COST_CEILING_USD:
        print(f"[research] WARNING: {e['slug']} cost ${res['cost']:.2f} is over the "
              f"${config.RESEARCH_COST_CEILING_USD:.2f} ceiling")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"summary_research-{e['slug']}_{today:%Y%m%d}.html"
    out.write_text(page, encoding="utf-8")
    print(f"[research] saved {out} (${res['cost']:.3f}, {res['rounds']} rounds)")
    result = {"key": e["slug"], "ok": True, "mailed": False, "cost": res["cost"],
              "path": str(out)}
    if mail:
        from send_notification import recipients_for, send_email
        result["mailed"] = bool(send_email(subject_for(e, today), page,
                                           to_email=recipients_for(full=False)))
        result["ok"] = result["mailed"]
    return result


def main(argv: List[str] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    only = [argv[i + 1] for i, a in enumerate(argv) if a == "--only" and i + 1 < len(argv)]
    only += [a.split("=", 1)[1] for a in argv if a.startswith("--only=")]
    mail = "--dry-run" not in argv and "--no-mail" not in argv
    force = "--force" in argv

    entries = [e for e in load_research_topics(config.RESEARCH_TOPICS_PATH)
               if not only or e["slug"] in only]
    if only and not entries:
        print(f"[research] no topic matches {only}")
        return 2
    now = datetime.now(timezone.utc)
    today = now.date()
    state = topics.load_state(STATE_PATH)

    due_entries = []
    for e in entries:
        t = as_topic(e)
        is_due, why = topics.due(t, state, now)
        if expired(e, today):
            is_due, why = False, f"expired {e['expires']}"
        if "--list" in argv:
            print(f"{e['slug']:16s} {'DUE ' if is_due else 'wait'} every "
                  f"{e['every_n_days']}d -- {why}")
        elif is_due or (force and not expired(e, today)):
            print(f"[research] {e['slug']}: running ({why})")
            due_entries.append(e)
        else:
            print(f"[research] {e['slug']}: skipping ({why})")
    if "--list" in argv or not due_entries:
        return 0

    max_hours = max(24 * e["every_n_days"] for e in due_entries)
    from digest_run import load_shared
    shared = load_shared(max_hours)
    client, x_client = _client(), TwitterIOClient()
    results = []
    for e in due_entries:
        try:
            res = run_one(e, state, client, x_client, shared, mail=mail, today=today)
        except Exception as ex:
            import traceback
            print(f"[research] {e['slug']} FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            res = {"key": e["slug"], "ok": False, "mailed": False, "reason": str(ex)}
        if res.get("reason"):
            print(f"[research] {e['slug']}: {res['reason']}")
        results.append(res)
        # Only a real mail moves the cadence; a dry run must not bench the topic.
        if mail:
            state = topics.record_run(state, as_topic(e), ok=bool(res["ok"]), now=now)
            topics.save_state(state, STATE_PATH)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
