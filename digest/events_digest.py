#!/usr/bin/env python3
"""Weekly events digest: what is on in EVENTS_CITY over the coming week.

Not a topic row in topics.py, on purpose. A topic summarizes a corpus the
scrapers already put on disk; this one has no corpus. It RESEARCHES: one
web-searching model call per category in config.EVENTS_CATEGORIES, each
returning structured events, then one tool-free call that scores every event
0-10 against Tim's interests. The mail is rendered here, in code, from those
two outputs -- so every link in it is a URL a search returned, never one a
model wrote into prose.

Runs as the last leg of run_pipeline.sh, every night, and decides for itself
whether it is due: it is due when there has been no successful send since the
most recent EVENTS_WEEKDAY (Sunday) in the city's timezone. So the normal case
is one mail per Sunday, and a Sunday that fails is retried on Monday instead of
skipped for a week.

It mails EMAIL_TO only, never EMAIL_TO_EXTRA: the ranking is against one
person's interests in one city. The subject is "<City> Events Digest - <date>",
which is what makes the recommender treat it as a digest series (sender name +
trailing date, see recommender/shortlist_job.privileged_digests).

    python events_digest.py --dry-run --force   # research + score, mail nothing
    python events_digest.py --force             # send now, due or not
"""

import html
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import config
import prompts
from x_scraper import is_blocked_link

STATE_PATH = Path("extracts/events_state.json")
OUT_DIR = Path("extracts/summaries")

EVENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["events"],
    "properties": {"events": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "start", "end", "venue", "organizer", "url",
                     "description"],
        "properties": {
            "title": {"type": "string"},
            "start": {"type": "string"},
            "end": {"type": ["string", "null"]},
            "venue": {"type": ["string", "null"]},
            "organizer": {"type": ["string", "null"]},
            "url": {"type": "string"},
            "description": {"type": "string"},
        }}}},
}

SCORE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scores"],
    "properties": {"scores": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "score", "why"],
        "properties": {
            "id": {"type": "integer"},
            "score": {"type": "integer"},
            "why": {"type": "string"},
        }}}},
}


# =============================================================================
# when
# =============================================================================

def today_local(now: datetime) -> date:
    return now.astimezone(ZoneInfo(config.EVENTS_TIMEZONE)).date()


def window(today: date) -> Tuple[date, date]:
    """Inclusive date range the digest covers: today through today+N-1."""
    return today, today + timedelta(days=config.EVENTS_WINDOW_DAYS - 1)


def due(last_success: Optional[str], today: date) -> Tuple[bool, str]:
    """Due iff nothing was sent since the most recent EVENTS_WEEKDAY."""
    anchor = today - timedelta(days=(today.weekday() - config.EVENTS_WEEKDAY) % 7)
    if last_success and date.fromisoformat(last_success) >= anchor:
        return False, f"already sent {last_success} (this week starts {anchor})"
    return True, f"no send since {anchor} (last: {last_success or 'never'})"


def load_state() -> Dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


# =============================================================================
# research and scoring
# =============================================================================

def _client():
    from openai import OpenAI
    # flex can queue; a 01:00 cron can afford to wait, a hung socket it cannot
    return OpenAI(timeout=900)


def _respond(client, **kwargs):
    """One Responses call on the configured tier, standard tier as fallback."""
    try:
        return client.responses.create(model=config.SUMMARY_MODEL,
                                       service_tier=config.SUMMARY_SERVICE_TIER,
                                       **kwargs)
    except Exception as e:
        if config.SUMMARY_SERVICE_TIER == "default":
            raise
        print(f"[events] {config.SUMMARY_SERVICE_TIER} tier failed ({e}) -- "
              f"retrying on standard tier")
        return client.responses.create(model=config.SUMMARY_MODEL,
                                       service_tier="default", **kwargs)


def cost_of(response) -> Tuple[float, int]:
    """(dollars, web searches) for one response, at SUMMARY_MODEL_PRICE."""
    p_in, p_cached, p_out = config.SUMMARY_MODEL_PRICE
    u = response.usage
    cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
    searches = sum(1 for o in response.output if o.type == "web_search_call")
    dollars = ((u.input_tokens - cached) * p_in + cached * p_cached
               + u.output_tokens * p_out) / 1e6
    return dollars + searches * config.WEB_SEARCH_PRICE_PER_CALL, searches


def research(client, category: Dict, start: date, end: date) -> Dict:
    prompt = prompts.EVENTS_RESEARCH.format(
        city=config.EVENTS_CITY, timezone=config.EVENTS_TIMEZONE,
        start=start.isoformat(), end=end.isoformat(),
        category=category["name"], interests=category["interests"],
        hints=category["hints"])
    r = _respond(client, input=prompt, tools=[{"type": "web_search"}],
                 text={"format": {"type": "json_schema", "name": "events",
                                  "schema": EVENT_SCHEMA, "strict": True}})
    dollars, searches = cost_of(r)
    events = json.loads(r.output_text)["events"]
    for e in events:
        e["category"] = category["name"]
    print(f"[events] {category['name']}: {len(events)} events, "
          f"{searches} searches, ${dollars:.3f}")
    return {"events": events, "cost": dollars, "searches": searches}


def score(client, events: List[Dict]) -> Tuple[Dict[int, Dict], float]:
    interests = "; ".join(c["interests"] for c in config.EVENTS_CATEGORIES)
    listing = "\n".join(
        json.dumps({"id": e["id"], "title": e["title"], "start": e["start"],
                    "venue": e["venue"], "organizer": e["organizer"],
                    "category": e["category"], "description": e["description"]},
                   ensure_ascii=False)
        for e in events)
    r = _respond(client, input=prompts.EVENTS_SCORE.format(
                     city=config.EVENTS_CITY, interests=interests, events=listing),
                 text={"format": {"type": "json_schema", "name": "scores",
                                  "schema": SCORE_SCHEMA, "strict": True}})
    scores = {s["id"]: s for s in json.loads(r.output_text)["scores"]}
    return scores, cost_of(r)[0]


# =============================================================================
# cleaning
# =============================================================================

def clean_url(url: str) -> str:
    """Query and fragment stripped -- that is where the tracking lives, and it
    is what makes the same event found by two categories collide."""
    p = urlsplit(url.strip())
    return urlunsplit((p.scheme, p.netloc.lower(), p.path.rstrip("/"), "", ""))


def parse_start(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.strip())
    except (ValueError, AttributeError):
        return None


def keep(events: List[Dict], start: date, end: date) -> Tuple[List[Dict], Dict[str, int]]:
    """In-window, linkable, deduplicated events, each given an integer id.

    Dedupe is on cleaned URL, then on (title, day): the same meetup is often
    found by two categories, sometimes via two different listing sites.
    """
    dropped = {"bad date": 0, "outside window": 0, "blocked link": 0,
               "duplicate": 0}
    out, seen = [], set()
    for e in events:
        when = parse_start(e.get("start"))
        if when is None:
            dropped["bad date"] += 1
            continue
        if not start <= when.date() <= end:
            dropped["outside window"] += 1
            continue
        if not e.get("url", "").startswith("http") or is_blocked_link(e["url"]):
            dropped["blocked link"] += 1
            continue
        keys = {clean_url(e["url"]),
                (e["title"].strip().lower(), when.date().isoformat())}
        if keys & seen:
            dropped["duplicate"] += 1
            continue
        seen |= keys
        out.append({**e, "id": len(out), "when": when})
    return out, dropped


# =============================================================================
# rendering
# =============================================================================

def _when(e: Dict, with_day: bool) -> str:
    """"Tue 3 Nov, 19:00"; just the time under a day heading; "all day" if no time."""
    w = e["when"]
    time = f"{w:%H:%M}" if len(e["start"].strip()) > 10 else ""
    if not with_day:
        return time or "all day"
    day = f"{w:%a} {w.day} {w:%b}"
    return f"{day}, {time}" if time else day


def _line(e: Dict, why: bool) -> str:
    """A top pick (`why`: dated, with the scorer's reason) or a by-day line."""
    esc = html.escape
    where = f" — {esc(e['venue'])}" if e.get("venue") else ""
    body = esc(e["why"]) if why else esc(e["description"])
    return (f'<li><strong>{esc(_when(e, with_day=why))}</strong> — '
            f'<a href="{esc(e["url"], quote=True)}">{esc(e["title"])}</a>'
            f'{where}. {body} <em>[{esc(e["category"])}, {e["score"]}/10]</em></li>')


def render(events: List[Dict], start: date, end: date, footer: str,
           banner: str = "") -> str:
    """Top picks by score, then everything else worth mailing, by day."""
    shown = [e for e in events if e["score"] >= config.EVENTS_MIN_SCORE]
    top = sorted((e for e in shown if e["score"] >= config.EVENTS_TOP_SCORE),
                 key=lambda e: (-e["score"], e["when"]))[:config.EVENTS_TOP_PICKS]
    top_ids = {e["id"] for e in top}
    rest = sorted((e for e in shown if e["id"] not in top_ids),
                  key=lambda e: e["when"])

    parts = [f"<h1>{html.escape(config.EVENTS_CITY)} Events Digest - "
             f"{start.isoformat()}</h1>",
             f"<p><em>{start:%a} {start.day} {start:%b} to {end:%a} {end.day} "
             f"{end:%b}, ranked by fit with your interests.</em></p>",
             banner, "<hr>"]
    if top:
        parts += ["<h3>Top picks</h3>", "<ul>",
                  *(_line(e, why=True) for e in top), "</ul>"]
    if rest:
        parts.append("<h3>Also this week</h3>")
        for day in sorted({e["when"].date() for e in rest}):
            parts += [f"<h4>{day:%A} {day.day} {day:%B}</h4>", "<ul>",
                      *(_line(e, why=False) for e in rest
                        if e["when"].date() == day), "</ul>"]
    parts += ["<hr>", f"<p><em>{footer}</em></p>"]
    return "\n".join(p for p in parts if p)


# =============================================================================
# the run
# =============================================================================

def main(argv: List[str] = None) -> int:
    # Flags it shares with digest_run.py, because run_pipeline.sh hands both of
    # them the same "$@". Anything else on that line is digest_run's business.
    argv = sys.argv[1:] if argv is None else argv
    mail = "--dry-run" not in argv and "--no-mail" not in argv
    force = "--force" in argv

    now = datetime.now().astimezone()
    today = today_local(now)
    state = load_state()
    is_due, why = due(state.get("last_success"), today)
    if not is_due and not force:
        print(f"[events] skipping: {why}")
        return 0
    print(f"[events] running: {why}" + (" (forced)" if force and not is_due else ""))

    start, end = window(today)
    client = _client()
    failed = []

    def one(cat):
        try:
            return research(client, cat, start, end)
        except Exception as e:
            print(f"[events] {cat['name']} FAILED: {type(e).__name__}: {e}")
            failed.append(cat["name"])
            return {"events": [], "cost": 0.0, "searches": 0}

    with ThreadPoolExecutor(max_workers=3) as pool:
        found = list(pool.map(one, config.EVENTS_CATEGORIES))
    raw = [e for f in found for e in f["events"]]
    cost = sum(f["cost"] for f in found)
    searches = sum(f["searches"] for f in found)

    events, dropped = keep(raw, start, end)
    print(f"[events] {len(raw)} found, {len(events)} kept, dropped {dropped}")
    if not events:
        # Zero events in a city like Berlin is a broken run, not a quiet week.
        print("[events] nothing survived -- refusing to mail an empty digest")
        return 1

    scores, score_cost = score(client, events)
    cost += score_cost
    unscored = [e["title"] for e in events if e["id"] not in scores]
    if unscored:
        print(f"[events] WARNING: scorer skipped {len(unscored)}: {unscored}")
    events = [{**e, "score": scores[e["id"]]["score"], "why": scores[e["id"]]["why"]}
              for e in events if e["id"] in scores]

    banner = ""
    if failed:
        banner = (f"<p><strong>⚠ Research failed for: "
                  f"{html.escape(', '.join(failed))}</strong> — those categories "
                  f"are missing this week; see ~/logs/ai-digest.log.</p>")
    shown = sum(e["score"] >= config.EVENTS_MIN_SCORE for e in events)
    footer = (f"{shown} of {len(events)} events shown (score ≥ "
              f"{config.EVENTS_MIN_SCORE}), from {searches} web searches by "
              f"{config.SUMMARY_MODEL} — ${cost:.2f}")
    page = render(events, start, end, footer, banner)

    if cost > config.EVENTS_COST_CEILING_USD:
        print(f"[events] WARNING: cost ${cost:.3f} is over the "
              f"${config.EVENTS_COST_CEILING_USD:.2f} ceiling")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"summary_events_{today:%Y%m%d}.html"
    out.write_text(page, encoding="utf-8")
    (OUT_DIR / f"events_{today:%Y%m%d}.json").write_text(json.dumps(
        {"window": [start.isoformat(), end.isoformat()], "dropped": dropped,
         "failed": failed, "cost": cost, "searches": searches,
         "events": [{**e, "when": e["when"].isoformat()} for e in events]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[events] saved {out} -- {shown} shown, ${cost:.3f}")

    if not mail:
        return 0
    from send_notification import recipients_for, send_email
    subject = f"{config.EVENTS_CITY} Events Digest - {today.isoformat()}"
    if not send_email(subject, page, to_email=recipients_for(full=False)):
        return 1
    STATE_PATH.write_text(json.dumps({"last_success": today.isoformat()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
