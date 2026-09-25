#!/usr/bin/env python3
"""Turns the run history in `stats_store` into two static HTML status pages.

    http://192.168.2.6:8080/ai-digest/       -- the nightly digest
    http://192.168.2.6:8080/recommender/     -- the shortlist recommender

Every rendered number comes out of the recorded rows, so the page can be
rebuilt from history at any time without running either job:

    .venv-rec/bin/python stats_page.py

Design notes worth keeping:

* **Dark only, on purpose.** One reader, on a VPN, who likes dark. The three
  series colours (`#3987e5 #d95926 #199e70`) are the reference palette's first
  three dark steps and were run through the skill's validator against this
  surface at `--pairs all`: every check passes, worst CVD dE 9.4, worst
  normal-vision dE 20.9. Do not add a fourth series without re-validating --
  slot four is yellow and it collides with orange.
* **No JavaScript, no CDN.** Hover detail rides on SVG `<title>`, which every
  browser renders natively, so the page works from a cold cache on a VPN with
  no egress. Every chart is also backed by a `<details>` table, which is both
  the accessibility fallback and the thing you actually want when a number
  looks wrong.
* **Missing is not zero.** A metric a run did not record renders as a dash. A
  status page that silently prints 0 for "we never measured this" is worse
  than no status page.
"""

import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
import stats_store

# -- palette (validated dark steps -- see module docstring) -------------------
SURFACE = "#1a1a19"
SURFACE_2 = "#232322"
LINE = "#3a3a37"
TEXT = "#ffffff"
TEXT_2 = "#c3c2b7"
TEXT_3 = "#8a897f"
SERIES = ["#3987e5", "#d95926", "#199e70"]
GOOD, WARN, BAD = "#199e70", "#c98500", "#e66767"

CSS = f"""
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 2rem 1.5rem 5rem; background: {SURFACE}; color: {TEXT};
  font: 15px/1.55 ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
}}
.wrap {{ max-width: 1040px; margin: 0 auto; }}
a {{ color: {SERIES[0]}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
h1 {{ font-size: 1.5rem; margin: 0 0 .2rem; letter-spacing: -.01em; }}
h2 {{
  font-size: .8rem; text-transform: uppercase; letter-spacing: .14em;
  color: {TEXT_3}; margin: 2.6rem 0 .9rem; font-weight: 600;
}}
h3 {{ font-size: .95rem; margin: 0 0 .15rem; font-weight: 600; }}
.sub {{ color: {TEXT_3}; font-size: .8rem; margin: 0 0 .3rem; }}
.head {{ display: flex; flex-wrap: wrap; gap: .8rem; align-items: baseline;
        justify-content: space-between; border-bottom: 1px solid {LINE};
        padding-bottom: 1rem; }}
.pill {{ display: inline-block; padding: .18rem .6rem; border-radius: 999px;
        font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
        border: 1px solid currentColor; }}
.tiles {{ display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }}
.tile {{ background: {SURFACE_2}; border: 1px solid {LINE}; border-radius: 6px; padding: .8rem .9rem; }}
.tile .k {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: {TEXT_3}; }}
.tile .v {{ font-size: 1.7rem; line-height: 1.15; margin-top: .25rem; font-variant-numeric: tabular-nums; }}
.tile .n {{ font-size: .75rem; color: {TEXT_2}; margin-top: .2rem; }}
.card {{ background: {SURFACE_2}; border: 1px solid {LINE}; border-radius: 6px;
        padding: 1rem 1.1rem; margin-bottom: .8rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
th, td {{ text-align: left; padding: .4rem .55rem; border-bottom: 1px solid {LINE};
         vertical-align: top; }}
th {{ color: {TEXT_3}; font-weight: 600; font-size: .7rem; text-transform: uppercase;
     letter-spacing: .08em; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr:last-child td {{ border-bottom: none; }}
details {{ margin-top: .7rem; }}
summary {{ cursor: pointer; color: {TEXT_3}; font-size: .75rem; letter-spacing: .06em;
          text-transform: uppercase; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1rem; font-size: .78rem;
          color: {TEXT_2}; margin: .1rem 0 .5rem; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
          margin-right: .35rem; vertical-align: baseline; }}
.muted {{ color: {TEXT_3}; }}
.why {{ color: {TEXT_2}; font-size: .78rem; }}
.warnbox {{ border-left: 3px solid {WARN}; padding: .6rem .9rem; background: {SURFACE_2};
           border-radius: 0 6px 6px 0; margin-bottom: .8rem; font-size: .85rem; }}
footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid {LINE};
         color: {TEXT_3}; font-size: .75rem; }}
svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
.post {{ border-left: 2px solid {LINE}; padding: .35rem 0 .35rem .7rem;
        margin: .5rem 0; font-size: .82rem; }}
.post .why {{ font-size: .75rem; }}
"""


# -- formatting ---------------------------------------------------------------

DASH = '<span class="muted">--</span>'


def esc(value):
    return html.escape(str(value), quote=True)


def num(value, digits=0, suffix=""):
    """Numbers render as numbers; a missing measurement renders as a dash."""
    if value is None:
        return DASH
    return f"{value:,.{digits}f}{suffix}"


def pct(value, digits=1):
    return DASH if value is None else f"{value * 100:.{digits}f}%"


def parse_time(stamp):
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def ago(stamp, now=None):
    when = parse_time(stamp)
    if not when:
        return "never"
    delta = (now or datetime.now(timezone.utc)) - when
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{delta.total_seconds() / 60:.0f} min ago"
    if hours < 48:
        return f"{hours:.1f} h ago"
    return f"{hours / 24:.1f} days ago"


def day(stamp):
    when = parse_time(stamp)
    return when.strftime("%m-%d") if when else "?"


def get(row, *path, default=None):
    """Nested lookup that tolerates rows written by an older version."""
    node = row
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


# -- components ---------------------------------------------------------------

def pill(text, color):
    return f'<span class="pill" style="color:{color}">{esc(text)}</span>'


def tile(key, value, note=""):
    note_html = f'<div class="n">{note}</div>' if note else ""
    return (f'<div class="tile"><div class="k">{esc(key)}</div>'
            f'<div class="v">{value}</div>{note_html}</div>')


def tiles(items):
    return '<div class="tiles">' + "".join(items) + "</div>"


def table(headers, rows, aligns=None):
    """headers: [str]; rows: [[html, ...]]; aligns: 'l'/'n' per column."""
    if not rows:
        return '<p class="muted">nothing recorded yet.</p>'
    aligns = aligns or ["l"] * len(headers)
    # a key/value table has no headers worth showing; an empty header row is
    # just a stray rule across the card
    head = "" if not any(headers) else "<thead><tr>" + "".join(
        f'<th class="{"num" if a == "n" else ""}">{esc(h)}</th>'
        for h, a in zip(headers, aligns)
    ) + "</tr></thead>"
    body = ""
    for row in rows:
        body += "<tr>" + "".join(
            f'<td class="{"num" if a == "n" else ""}">{cell}</td>'
            for cell, a in zip(row, aligns)
        ) + "</tr>"
    return f"<table>{head}<tbody>{body}</tbody></table>"


def details(label, body):
    return f"<details><summary>{esc(label)}</summary>{body}</details>"


def legend(names):
    if len(names) < 2:      # one series needs no legend -- the title names it
        return ""
    return '<div class="legend">' + "".join(
        f'<span><span class="swatch" style="background:{SERIES[i % len(SERIES)]}"></span>'
        f'{esc(n)}</span>' for i, n in enumerate(names)
    ) + "</div>"


def line_chart(labels, series, fmt=lambda v: f"{v:,.2f}", height=190):
    """Multi-series line chart. series: [(name, [value|None, ...])].

    Values are plotted against run index, not against calendar time -- these
    are nightly jobs, and a missed night should show as a gap in the sequence
    of runs rather than being silently interpolated over.
    """
    series = [s for s in series if any(v is not None for v in s[1])]
    if not labels or not series:
        return '<p class="muted">not enough runs yet to plot.</p>'
    flat = [v for _, values in series for v in values if v is not None]
    lo, hi = min(flat), max(flat)
    if hi == lo:
        lo, hi = lo - max(abs(lo) * 0.1, 0.5), hi + max(abs(hi) * 0.1, 0.5)
    pad = (hi - lo) * 0.12
    lo, hi = lo - pad, hi + pad
    w, h, ml, mr, mt, mb = 760, height, 52, 58, 12, 26
    plot_w, plot_h = w - ml - mr, h - mt - mb

    def x_at(i):
        return ml + (plot_w if len(labels) == 1 else plot_w * i / (len(labels) - 1))

    def y_at(v):
        return mt + plot_h * (1 - (v - lo) / (hi - lo))

    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for frac in (0, 0.25, 0.5, 0.75, 1):
        value = hi - (hi - lo) * frac
        y = mt + plot_h * frac
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" '
                   f'stroke="{LINE}" stroke-width="1"/>')
        out.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="10" fill="{TEXT_3}">{esc(fmt(value))}</text>')
    step = max(1, len(labels) // 9)
    for i, label in enumerate(labels):
        if i % step == 0 or i == len(labels) - 1:
            out.append(f'<text x="{x_at(i):.1f}" y="{h - 8}" text-anchor="middle" '
                       f'font-size="10" fill="{TEXT_3}">{esc(label)}</text>')

    for si, (name, values) in enumerate(series):
        color = SERIES[si % len(SERIES)]
        run, runs = [], []
        for i, v in enumerate(values):
            if v is None:
                if run:
                    runs.append(run)
                run = []
            else:
                run.append((x_at(i), y_at(v)))
        if run:
            runs.append(run)
        for segment in runs:
            if len(segment) > 1:
                path = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}"
                                for k, (x, y) in enumerate(segment))
                out.append(f'<path d="{path}" fill="none" stroke="{color}" '
                           f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            elif segment:
                x, y = segment[0]
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        # generous invisible hit targets: this is the hover layer, no JS needed
        for i, v in enumerate(values):
            if v is None:
                continue
            out.append(
                f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="9" fill="transparent">'
                f'<title>{esc(name)} - {esc(labels[i])}: {esc(fmt(v))}</title></circle>'
            )
        last = next((i for i in range(len(values) - 1, -1, -1) if values[i] is not None), None)
        if last is not None and len(series) <= 3:   # direct label, not a number on every point
            out.append(f'<text x="{x_at(last) + 8:.1f}" y="{y_at(values[last]) + 4:.1f}" '
                       f'font-size="10" fill="{TEXT_2}">{esc(fmt(values[last]))}</text>')
    out.append("</svg>")

    rows = [[esc(labels[i])] + [num_or_dash(values[i], fmt) for _, values in series]
            for i in range(len(labels))]
    grid = table(["run"] + [n for n, _ in series], rows, ["l"] + ["n"] * len(series))
    return legend([n for n, _ in series]) + "".join(out) + details("data", grid)


def num_or_dash(value, fmt):
    return DASH if value is None else esc(fmt(value))


def bar_chart(items, total_label="", fmt=lambda v: f"{v:,.0f}", log=False,
              bodies=None):
    """Horizontal bars for one categorical series. items: [(label, value)].

    `log=True` scales bar length by log10(1+v). Use it only where the series
    really does span orders of magnitude -- a long tail of accounts with one
    post each against a head with forty. The number printed at the end of every
    bar is always the true value, and the table view carries them all, so the
    scale changes the picture and never the figures.

    `bodies` is {label: html}; when given, each row becomes expandable and that
    html is what you get when you open it.
    """
    items = [(k, v) for k, v in items if v is not None]
    if not items:
        return '<p class="muted">nothing recorded.</p>'
    scale = (lambda v: math.log10(1 + max(v, 0))) if log else (lambda v: max(v, 0))
    top = max(scale(v) for _, v in items) or 1
    row_h, label_w, bar_w = 22, 210, 470
    h = row_h * len(items) + 6
    out = [f'<svg viewBox="0 0 {label_w + bar_w + 60} {h}" role="img">']
    for i, (label, value) in enumerate(items):
        y = i * row_h + 4
        width = max(2.0, bar_w * scale(value) / top)
        out.append(f'<text x="0" y="{y + 12}" font-size="11" fill="{TEXT_2}">'
                   f'{esc(label[:34])}</text>')
        out.append(f'<rect x="{label_w}" y="{y + 3}" width="{width:.1f}" height="13" '
                   f'rx="4" fill="{SERIES[0]}">'
                   f'<title>{esc(label)}: {esc(fmt(value))}{esc(total_label)}</title></rect>')
        out.append(f'<text x="{label_w + width + 8:.1f}" y="{y + 14}" font-size="11" '
                   f'fill="{TEXT_2}">{esc(fmt(value))}</text>')
    out.append("</svg>")
    grid = table(["item", "n"], [[esc(k), esc(fmt(v))] for k, v in items], ["l", "n"])
    note = ('<p class="sub">bar length is log10(1+n); the number on each bar is '
            'the real count</p>' if log else "")
    drill = ""
    if bodies:
        drill = "".join(
            details(f"{label} ({fmt(value)})", bodies[label])
            for label, value in items if bodies.get(label)
        )
    return note + "".join(out) + drill + details("data", grid)


def contribution_chart(contributions, labels=None):
    """Diverging bars: what pushed this document's log-odds up or down.

    Blue right / red left with a neutral zero line -- the documented diverging
    pair. Units are log-odds against the average document, and because the model
    is linear they are exact rather than indicative: they sum to the score.
    """
    items = [(k, v) for k, v in contributions.items() if v is not None]
    if not items:
        return ""
    labels = labels or {}
    span = max(abs(v) for _, v in items) or 1.0
    row_h, label_w, half = 24, 150, 190
    w, h = label_w + 2 * half + 70, row_h * len(items) + 8
    mid = label_w + half
    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    out.append(f'<line x1="{mid}" y1="0" x2="{mid}" y2="{h - 6}" '
               f'stroke="{TEXT_3}" stroke-width="1"/>')
    for i, (key, value) in enumerate(items):
        y = i * row_h + 5
        length = max(2.0, half * abs(value) / span)
        x = mid if value >= 0 else mid - length
        colour = SERIES[0] if value >= 0 else BAD
        out.append(f'<text x="0" y="{y + 12}" font-size="11" fill="{TEXT_2}">'
                   f'{esc(labels.get(key, key))}</text>')
        out.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{length:.1f}" height="13" '
                   f'rx="4" fill="{colour}">'
                   f'<title>{esc(key)}: {value:+.3f} log-odds</title></rect>')
        anchor_x = mid + length + 8 if value >= 0 else mid - length - 8
        align = "start" if value >= 0 else "end"
        out.append(f'<text x="{anchor_x:.1f}" y="{y + 14}" font-size="11" '
                   f'text-anchor="{align}" fill="{TEXT_2}">{value:+.2f}</text>')
    out.append("</svg>")
    return "".join(out)


CONTRIBUTION_LABELS = {
    "topic": "topic (what it is about)",
    "length": "length (log words)",
    "format": "format (article/rss/email/...)",
}

EVIDENCE_LABELS = {
    "rated": "you rated it", "favorited": "you favourited it",
    "read": "you read it", "opened": "you opened it",
    "passed": "shown and skipped", "ignored": "went stale unopened",
}


def why_this_score(attr):
    """The expandable explanation behind one pick's score."""
    if not attr:
        return ""
    contributions = attr.get("contributions") or {}
    base_p = attr.get("baseline_probability")
    final_p = attr.get("probability")
    body = (
        f'<p class="why">The average document in your library scores '
        f'<strong>{pct(base_p, 2)}</strong>. These three terms take it to '
        f'<strong>{pct(final_p, 2)}</strong>. Bars are log-odds and they add up '
        f'exactly -- the model is linear, so this is the score, not an '
        f'approximation of it.</p>'
        + contribution_chart(contributions, CONTRIBUTION_LABELS)
    )
    for side, heading in (("like", "closest things you engaged with"),
                          ("unlike", "closest things you did not")):
        hits = attr.get(side) or []
        if not hits:
            continue
        body += f'<p class="sub" style="margin-top:.7rem">{heading}</p>'
        body += table(["", "cos", "why it is labelled that way"], [
            [esc(h.get("title") or h.get("id")), num(h.get("similarity"), 2),
             esc(EVIDENCE_LABELS.get(h.get("reason"), h.get("reason") or "?"))]
            for h in hits], ["l", "n", "l"])
    body += ('<p class="why" style="margin-top:.7rem">The topic term is a single '
             'number because the encoder is frozen and its 384 dimensions have no '
             'individual meaning -- a bar per dimension would be 384 bars of noise. '
             'The nearest-neighbour lists above are the readable form of that term: '
             'what this looks like among things you have already judged.</p>')
    fmt = (contributions.get("format") or 0)
    if abs(fmt) > abs(contributions.get("topic") or 0):
        body += ('<p class="why"><strong>Format is outweighing topic on this one.</strong> '
                 'The category one-hots track where a document lives -- the fresh feed '
                 'is mostly <code>rss</code>, the backlog mostly <code>article</code> -- '
                 'and location correlates with the label, so a large format term means '
                 'the score is being driven by what kind of thing it is rather than what '
                 'it is about. That is the known reason scores do not compare across '
                 'the two pools.</p>')
    if base_p is not None and base_p > 0.4:
        body += (f'<p class="why">The {pct(base_p, 1)} baseline is not the real base '
                 'rate -- roughly one document in eight is a positive. The head is '
                 'fitted with <code>class_weight="balanced"</code>, so its probabilities '
                 'are calibrated to a balanced prior. Compare scores to each other, '
                 'never read one as a probability that you will read the thing.</p>')
    return details("why this score", body)


# -- drill-down: the posts themselves -----------------------------------------
#
# Rendered straight off the dumps each run already writes to extracts/, whose
# paths the run row records. That keeps the history rows small -- 2,230 tweets
# in every row would be ~90 MB of jsonl after six months -- and means the page
# and the digest are reading literally the same bytes.

def load_dump(path):
    """A recorded source dump, or None if it has been cleaned up since."""
    if not path:
        return None
    f = Path(path)
    if not f.is_absolute():
        f = Path(__file__).resolve().parent / f
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clip(text, limit):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def external_links(urls):
    """Only links that leave the walled platforms -- the rest are dead to him."""
    from x_scraper import is_blocked_link
    keep = [u for u in (urls or []) if not is_blocked_link(u)]
    if not keep:
        return ""
    return " ".join(f'<a href="{esc(u)}">{esc(u.split("//")[-1][:60])}</a>'
                    for u in keep[:3])


def tweet_bodies(tweets):
    """{handle: html} -- every post that account contributed, busiest first."""
    by_handle = {}
    for tweet in tweets or []:
        by_handle.setdefault(tweet.get("handle"), []).append(tweet)
    bodies = {}
    for handle, items in by_handle.items():
        items.sort(key=lambda t: (t.get("likes") or 0) + (t.get("retweets") or 0),
                   reverse=True)
        rows = ""
        for t in items:
            links = external_links(t.get("external_links"))
            quoted = (f'<div class="why">quoting: {esc(clip(t.get("quoted_text"), 220))}</div>'
                      if t.get("quoted_text") else "")
            rows += (
                f'<div class="post"><div class="why">'
                f'{num(t.get("likes"))} likes &middot; {num(t.get("retweets"))} RT '
                f'&middot; {esc(str(t.get("created_at") or "")[:16])}</div>'
                f'<div>{esc(clip(t.get("text"), 800))}</div>{quoted}'
                + (f'<div class="why">{links}</div>' if links else "")
                + '</div>'
            )
        bodies[handle] = rows
    return bodies


def reddit_bodies(posts):
    """{subreddit: html} -- the posts, their scores and their top comment."""
    by_sub = {}
    for post in posts or []:
        by_sub.setdefault(post.get("subreddit"), []).append(post)
    bodies = {}
    for sub, items in by_sub.items():
        items.sort(key=lambda p: -(p.get("score") or 0))
        rows = ""
        for p in items:
            link = external_links([p["url"]] if p.get("url") else [])
            comments = p.get("comments") or []
            top = max(comments, key=lambda c: c.get("score") or 0) if comments else None
            rows += (
                f'<div class="post"><div><strong>{esc(clip(p.get("title"), 200))}</strong></div>'
                f'<div class="why">{num(p.get("score"))} pts &middot; '
                f'{len(comments)} top-level comments</div>'
                + (f'<div class="why">{link}</div>' if link else "")
                + (f'<div class="why">{esc(clip(p.get("selftext"), 600))}</div>'
                   if p.get("selftext") else "")
                + (f'<div class="why">top comment [{num(top.get("score"))}]: '
                   f'{esc(clip(top.get("body"), 300))}</div>' if top else "")
                + '</div>'
            )
        bodies[sub] = rows
    return bodies


def page(title, subtitle, status, body, other):
    """Shell shared by both pages. `other` is (href, label) for the sibling page."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head><body><div class="wrap">
<div class="head">
  <div><h1>{esc(title)}</h1><p class="sub">{subtitle}</p></div>
  <div>{status}</div>
</div>
{body}
<footer>
  rendered {esc(generated)} &middot; <a href="{esc(other[0])}">{esc(other[1])}</a>
  &middot; <a href="../">all pages</a><br>
  source: <code>~/projects/ai-news/stats_page.py</code>, history in
  <code>data/run_stats/*.jsonl</code>. Rebuild any time with
  <code>python stats_page.py</code>.
</footer>
</div></body></html>
"""


# -- digest page --------------------------------------------------------------

def render_digest(rows):
    if not rows:
        return page("AI digest", "no run recorded yet", pill("unknown", TEXT_3),
                    '<p class="muted">The pipeline has not written a stats row yet. '
                    'It will after its next 01:00 UTC run.</p>',
                    ("../recommender/", "recommender status"))
    last = rows[-1]
    recent = rows[-30:]
    labels = [day(r.get("run_at") or r.get("recorded_at")) for r in recent]

    reddit_n = get(last, "reddit", "total")
    x_n = get(last, "x", "total")
    cost = get(last, "llm", "cost_usd")
    ceiling = get(last, "llm", "ceiling")
    read_pages = get(last, "links", "read")
    walled = get(last, "links", "walled")

    problems = list(get(last, "problems", default=[]))
    if not last.get("ok", True):
        problems.append("the summarizer leg did not produce a digest")
    status = (pill("healthy", GOOD) if not problems
              else pill("needs a look", WARN))

    warn_html = ""
    if problems:
        warn_html = ('<div class="warnbox"><strong>Flagged this run</strong><ul>'
                     + "".join(f"<li>{esc(p)}</li>" for p in problems) + "</ul></div>")

    over = cost is not None and ceiling is not None and cost > ceiling
    spend = get(last, "x", "spend", default={}) or {}
    x_usd = spend.get("usd")
    total_usd = None if cost is None and x_usd is None else (cost or 0) + (x_usd or 0)
    head_tiles = tiles([
        tile("posts analyzed", num((reddit_n or 0) + (x_n or 0)),
             f"{num(x_n)} X &middot; {num(reddit_n)} reddit"),
        tile("cost per digest", DASH if total_usd is None else f"${total_usd:.4f}",
             (f'{"$%.4f" % cost if cost is not None else "?"} llm + '
              f'{"$%.4f" % x_usd if x_usd is not None else "?"} twitterapi')),
        tile("pages read", num(read_pages),
             f"{num(walled)} walled but linked"),
        tile("prompt", num(get(last, "llm", "prompt_chars"), suffix=" ch"),
             f'{num(get(last, "llm", "input_tokens"))} input tokens'),
        tile("last run", esc(ago(last.get("run_at"))),
             esc((last.get("run_at") or "")[:16].replace("T", " "))),
    ])

    # -- per source, this run
    by_sub = get(last, "reddit", "by_subreddit", default={}) or {}
    by_acct = get(last, "x", "by_account", default={}) or {}
    configured = get(last, "reddit", "configured", default=list(by_sub)) or []
    sub_items = sorted(
        [(s, by_sub.get(s, 0)) for s in dict.fromkeys(list(configured) + list(by_sub))],
        key=lambda kv: -kv[1],
    )
    acct_items = sorted(by_acct.items(), key=lambda kv: -kv[1])[:40]

    health = get(last, "x", "health", default={}) or {}
    x_rows = [
        ["status", esc(health.get("status", "?"))],
        ["accounts followed", num(health.get("accounts"))],
        ["search batches", f'{num(health.get("batches_attempted"))} attempted, '
                           f'{num(health.get("batches_failed"))} failed'],
        ["api calls", num(health.get("api_calls"))],
        ["new tweets fetched", num(health.get("tweets_new"))],
        ["reached the model", f'{num(x_n)}'
         + (f' <span style="color:{WARN}">({num(get(last, "llm", "tweets_dropped"))} '
            f'dropped to fit the budget)</span>'
            if get(last, "llm", "tweets_dropped") else "")],
        ["per-run cap hit", "yes" if health.get("tweet_cap_hit") else "no"],
    ]
    if health.get("errors"):
        x_rows.append(["last error", f'<span class="why">{esc(str(health["errors"][-1])[:300])}</span>'])

    posts = load_dump(get(last, "reddit", "file"))
    tweets = load_dump(get(last, "x", "file"))
    sub_bodies = reddit_bodies(posts) if posts else None
    acct_bodies = tweet_bodies(tweets) if tweets else None
    missing = ('<p class="sub">The source dump for this run is no longer on disk, '
               'so the posts themselves cannot be shown.</p>')

    sources = (
        '<h2>sources, this run</h2>'
        f'<div class="card"><h3>reddit</h3>'
        f'<p class="sub">{num(reddit_n)} posts across {len(sub_items)} subreddits, '
        f'{num(get(last, "reddit", "comments"))} comments pulled in. '
        f'Window {num(last.get("window_days"))} days, sorted by '
        f'{esc(get(last, "reddit", "sort_by", default="?"))}. '
        f'Open a subreddit to read what it actually contributed.</p>'
        + bar_chart(sub_items, " posts", log=True, bodies=sub_bodies)
        + ("" if sub_bodies or not reddit_n else missing) + '</div>'
        f'<div class="card"><h3>X / twitter</h3>'
        + table(["", ""], x_rows)
        + ('<p class="sub" style="margin-top:.8rem">busiest accounts this run '
           f'-- top {len(acct_items)} of {len(by_acct)}. Open one to read its posts.</p>'
           + bar_chart(acct_items, " posts", log=True, bodies=acct_bodies)
           + ("" if acct_bodies else missing) if acct_items else "")
        + '</div>'
    )

    # -- summarization
    llm = last.get("llm") or {}
    summarization = (
        '<h2>summarization</h2><div class="card">'
        + table(["", ""], [
            ["model", f'{esc(llm.get("model", "?"))} '
                      f'<span class="muted">({esc(llm.get("tier", "?"))} tier)</span>'],
            ["prompt", f'{num(llm.get("prompt_chars"))} chars -> '
                       f'{num(llm.get("input_tokens"))} input tokens '
                       f'({num(llm.get("cached_tokens"))} cached)'],
            ["output", f'{num(llm.get("output_tokens"))} tokens '
                       f'({num(llm.get("reasoning_tokens"))} of them reasoning), '
                       f'{num(llm.get("summary_chars"))} chars of html'],
            ["compression", (f'{llm["prompt_chars"] / llm["summary_chars"]:.0f}x'
                             if llm.get("prompt_chars") and llm.get("summary_chars")
                             else DASH)],
            ["cost", (f'${llm["cost_usd"]:.4f}' if llm.get("cost_usd") is not None else DASH)
                     + (f' &middot; ${llm["cost_usd"] * 365:.2f}/yr at this rate'
                        if llm.get("cost_usd") else "")],
            ["cost per post", (f'${llm["cost_usd"] / ((reddit_n or 0) + (x_n or 0)) * 1000:.3f}'
                               ' per 1000 posts'
                               if llm.get("cost_usd") and (reddit_n or x_n) else DASH)],
            ["corpus trimmed", (f'{num(llm.get("tweets_dropped"))} lowest-engagement '
                                f'X posts dropped to fit the budget'
                                if llm.get("tweets_dropped") else "no")],
        ]) + '</div>'
    )

    # -- what the night cost, both vendors
    remaining = spend.get("credits_remaining")
    used = spend.get("credits_used")
    rate = spend.get("credits_per_usd")
    runs_left = (remaining / used) if (remaining and used) else None
    money = (
        '<h2>what it cost</h2><div class="card">'
        + table(["vendor", "unit", "usd"], [
            [f'openai <span class="muted">{esc(get(last, "llm", "model", default="?"))}</span>',
             f'{num(get(last, "llm", "input_tokens"))} in / '
             f'{num(get(last, "llm", "output_tokens"))} out tokens',
             DASH if cost is None else f"${cost:.4f}"],
            ["twitterapi.io",
             f'{num(used)} credits'
             + (f' of {num(remaining)} left' if remaining is not None else ""),
             DASH if x_usd is None else f"${x_usd:.4f}"],
            ["<strong>total</strong>", "",
             DASH if total_usd is None else f"<strong>${total_usd:.4f}</strong>"],
        ], ["l", "l", "n"])
        + (f'<p class="why" style="margin-top:.8rem">At this burn rate the '
           f'twitterapi.io balance is good for about <strong>{runs_left:,.0f} more '
           f'runs</strong>.</p>' if runs_left else "")
        + (f'<p class="why">The credit count is measured -- balance before minus '
           f'balance after, with a {config.TWITTERAPI_CREDIT_SETTLE_SECONDS}s settle '
           f'wait because the debit lands late. The dollar figure is a conversion at '
           f'{rate:,.0f} credits per dollar from <code>config.py</code>, and that rate '
           f'is <strong>unverified</strong> -- check it against a twitterapi.io invoice '
           f'and fix the constant if it is wrong.</p>' if rate and x_usd is not None
           else '<p class="why">twitterapi.io spend is not recorded for this run. '
                'It is measured from the balance before and after the scrape, so it '
                'appears from the first run after that instrumentation landed.</p>')
        + (f'<p class="why">The OpenAI ceiling is ${ceiling:.2f} per digest and '
           f'applies to the model call only; twitterapi.io is outside it.</p>'
           if ceiling else "")
        + '</div>'
    )

    # -- links
    by_status = get(last, "links", "by_status", default={}) or {}
    links = (
        '<h2>link enrichment</h2><div class="card">'
        f'<p class="sub">{num(get(last, "links", "attempted"))} links followed, '
        f'{num(read_pages)} readable, {num(walled)} behind a wall (those get listed '
        f'in the digest so you can open them yourself), '
        f'{num(get(last, "links", "cache_hits"))} served from cache.</p>'
        + bar_chart(sorted(by_status.items(), key=lambda kv: -kv[1]), " links")
        + '</div>'
    )

    # -- over time
    trends = (
        '<h2>over time</h2>'
        '<div class="card"><h3>posts reaching the model</h3>'
        '<p class="sub">Per nightly run. A flat zero on one series is a dead leg, '
        'not a quiet news day.</p>'
        + line_chart(labels, [
            ("X posts", [get(r, "x", "total") for r in recent]),
            ("reddit posts", [get(r, "reddit", "total") for r in recent]),
            ("pages read", [get(r, "links", "read") for r in recent]),
        ], fmt=lambda v: f"{v:,.0f}") + '</div>'
        '<div class="card"><h3>cost per digest</h3>'
        + (f'<p class="sub">Ceiling is ${ceiling:.2f} per digest.</p>' if ceiling else "")
        + line_chart(labels, [("usd", [get(r, "llm", "cost_usd") for r in recent])],
                     fmt=lambda v: f"${v:.3f}")
        + '</div>'
    )

    history = '<h2>run history</h2><div class="card">' + table(
        ["run", "X", "reddit", "pages", "in tok", "out tok", "cost", "ok"],
        [[esc((r.get("run_at") or "")[:16].replace("T", " ")),
          num(get(r, "x", "total")), num(get(r, "reddit", "total")),
          num(get(r, "links", "read")),
          num(get(r, "llm", "input_tokens")), num(get(r, "llm", "output_tokens")),
          (f'${get(r, "llm", "cost_usd"):.4f}' if get(r, "llm", "cost_usd") is not None else DASH),
          ("ok" if r.get("ok", True) else f'<span style="color:{BAD}">failed</span>')]
         for r in reversed(rows[-14:])],
        ["l", "n", "n", "n", "n", "n", "n", "l"],
    ) + '</div>'

    subtitle = (f'{len(rows)} run{"" if len(rows) == 1 else "s"} recorded &middot; '
                f'last {esc(ago(last.get("run_at")))} &middot; 01:00 UTC nightly')
    return page("AI digest", subtitle, status,
                warn_html + head_tiles + sources + money + summarization + links
                + trends + history,
                ("../recommender/", "recommender status"))


# -- recommender page ---------------------------------------------------------

SLOT_NAMES = {
    "feed": "ranked, fresh feed",
    "feed-random": "random control, fresh feed",
    "later": "ranked, resurfaced backlog",
    "later-random": "random control, backlog",
}


def render_recommender(rows):
    if not rows:
        return page("Shortlist recommender", "no run recorded yet", pill("unknown", TEXT_3),
                    '<p class="muted">The nightly cycle has not written a stats row yet. '
                    'It will after its next 02:00 UTC run.</p>',
                    ("../ai-digest/", "digest status"))
    last = rows[-1]
    recent = rows[-30:]
    labels = [day(r.get("run_at") or r.get("recorded_at")) for r in recent]

    holdout = last.get("holdout") or {}
    auc = holdout.get("auc")
    corpus = last.get("corpus") or {}
    lab = last.get("labels") or {}
    picks = last.get("picks") or []

    problems = list(last.get("problems") or [])
    status = (pill("healthy", GOOD) if last.get("ok", True) and not problems
              else pill("needs a look", WARN if last.get("ok", True) else BAD))
    warn_html = ""
    if problems:
        warn_html = ('<div class="warnbox"><strong>Flagged this run</strong><ul>'
                     + "".join(f"<li>{esc(p)}</li>" for p in problems) + "</ul></div>")
    if last.get("dry_run"):
        warn_html += ('<div class="warnbox">This was a <strong>dry run</strong> -- '
                      'decisions were made, no tags were written.</div>')
    stale = bool(get(last, "sync", "skipped"))
    stale_note = ('<p class="why" style="margin-top:.8rem"><strong>This run skipped '
                  'the sync</strong>, so every reading state below is frozen at the '
                  'last real sync. Anything opened, finished or rated since then is '
                  'not in the local store yet and will read as untouched. The 02:00 '
                  'cycle always syncs first, so this only ever affects a hand-run '
                  '<code>--skip-sync</code>.</p>' if stale else "")
    if stale:
        warn_html += ('<div class="warnbox">Sync was skipped -- reading states are '
                      'as of the previous sync, not as of now.</div>')

    head_tiles = tiles([
        tile("holdout auc", num(auc, 3),
             (f'vs {holdout["auc_random"]:.3f} random'
              if holdout.get("auc_random") is not None else "not enough signal yet")),
        tile("shortlisted", num(len(picks)),
             f'{sum(1 for p in picks if not str(p.get("slot", "")).endswith("-random"))} ranked, '
             f'{sum(1 for p in picks if str(p.get("slot", "")).endswith("-random"))} random'),
        tile("labels", num(lab.get("usable")),
             f'{num(lab.get("positives"))} positive'),
        tile("corpus", num(corpus.get("total")),
             f'{num(corpus.get("embedded"))} embedded'),
        tile("last run", esc(ago(last.get("run_at"))),
             esc(str(last.get("cycle") or "")[:10])),
    ])

    # -- retraining performance over time
    perf = (
        '<h2>retraining performance</h2>'
        '<div class="card"><h3>held-out AUC, run by run</h3>'
        '<p class="sub">Model against its two baselines, on documents saved in the '
        'last 30 days and excluded from that night\'s fit. 0.5 is a coin flip.</p>'
        + line_chart(labels, [
            ("model", [get(r, "holdout", "auc") for r in recent]),
            ("word-count baseline", [get(r, "holdout", "auc_word_count") for r in recent]),
            ("random baseline", [get(r, "holdout", "auc_random") for r in recent]),
        ], fmt=lambda v: f"{v:.3f}")
        + '<p class="why" style="margin-top:.8rem"><strong>Read this as a smoke '
          'test, not as performance.</strong> The labels are derived from the corpus '
          '<em>as it stands tonight</em>, so something read yesterday counts as a '
          'positive inside the training half of an older split. The honest number is '
          'the ranked-vs-random comparison further down; this one catches a model '
          'that has fallen over, nothing finer.</p>'
        '</div>'
        '<div class="card"><h3>training set growth</h3>'
        + line_chart(labels, [
            ("usable labels", [get(r, "labels", "usable") for r in recent]),
            ("positives", [get(r, "labels", "positives") for r in recent]),
        ], fmt=lambda v: f"{v:,.0f}") + '</div>'
    )

    if holdout:
        k = next((key.split("_at_")[1] for key in holdout
                  if key.startswith("precision_at_") and key.count("_") == 2), "10")
        perf += '<div class="card"><h3>tonight\'s fit, in full</h3>' + table(
            ["", "model", "word count", "random"],
            [["AUC", num(holdout.get("auc"), 3), num(holdout.get("auc_word_count"), 3),
              num(holdout.get("auc_random"), 3)],
             [f"precision@{k}", num(holdout.get(f"precision_at_{k}"), 3),
              num(holdout.get(f"precision_at_{k}_word_count"), 3),
              num(holdout.get(f"precision_at_{k}_random"), 3)]],
            ["l", "n", "n", "n"],
        ) + table(["", ""], [
            ["trained on", f'{num(holdout.get("train_n"))} documents'],
            ["held out", f'{num(holdout.get("test_n"))} documents, '
                         f'{pct(holdout.get("test_positive_rate"))} of them positive'],
            ["features", num(last.get("features"))],
        ]) + '</div>'

    # -- tonight's shortlist
    pick_rows = []
    for p in sorted(picks, key=lambda p: (str(p.get("slot", "")), -(p.get("score") or 0))):
        title = esc(p.get("title") or "(untitled)")
        link = p.get("url")
        title_html = f'<a href="{esc(link)}">{title}</a>' if link else title
        meta = " &middot; ".join(filter(None, [
            esc(p.get("site_name") or ""), esc(p.get("author") or ""),
            f'{p["word_count"]:,} words' if p.get("word_count") else "",
        ]))
        pick_rows.append([
            f'{title_html}<div class="why">{meta}</div>'
            f'<div class="why">{esc(p.get("reason") or "")}</div>'
            + (f'<div class="why">nearest thing you have read: '
               f'{esc(p["nearest"]["title"])} (cos {p["nearest"]["similarity"]:.2f})</div>'
               if p.get("nearest") else "")
            + why_this_score(p.get("attribution")),
            esc(SLOT_NAMES.get(p.get("slot"), p.get("slot") or "?")),
            num(p.get("score"), 3) if p.get("score") is not None else DASH,
        ])
    pools = last.get("pools") or {}
    shortlist = (
        '<h2>tonight\'s shortlist</h2>'
        f'<div class="card"><p class="sub">Ranked '
        f'{num(pools.get("feed_candidates"))} eligible fresh feed items and '
        f'{num(pools.get("backlog_sampled"))} sampled out of '
        f'{num(pools.get("later_candidates"))} eligible backlog items; '
        f'{len(picks)} made the cut. Titles link into Readwise Reader.</p>'
        + table(["document", "slot", "score"], pick_rows, ["l", "l", "n"])
        + '<p class="why" style="margin-top:.8rem">Scores are not comparable across '
          'the two pools -- backlog items score near 1.0 because location correlates '
          'with the label. Compare within a slot, never between.</p></div>'
    )

    # -- what happened to the last batch
    evicted = last.get("evicted") or []
    verdict_html = {"good": f'<span style="color:{GOOD}">rated good</span>',
                    "bad": f'<span style="color:{BAD}">rated bad</span>'}
    ev_rows = [[esc(e.get("title") or e.get("id") or "?"),
                esc(SLOT_NAMES.get(e.get("slot"), e.get("slot") or "?")),
                esc(e.get("outcome") or "?"),
                verdict_html.get(e.get("rating"), DASH),
                pct(e.get("progress")) if e.get("progress") is not None else DASH]
               for e in evicted]
    evictions = (
        '<h2>the outgoing batch</h2><div class="card">'
        f'<p class="sub">{len(evicted)} documents came off the shortlist to make room. '
        'What became of them is the feedback that trains tomorrow\'s model.</p>'
        + table(["document", "slot it came from", "outcome", "your verdict",
                 "progress"], ev_rows, ["l", "l", "l", "l", "n"])
        + stale_note
        + '<p class="why" style="margin-top:.8rem">A "passed" here is not '
          'automatically a negative label. A document evicted on the same day it '
          'was added was never really offered, so it is excluded from training -- '
          'otherwise the job would manufacture negatives out of its own '
          'scheduling.</p></div>'
    )

    # -- live arms: the honest metric
    arms = last.get("arms") or {}
    arm_rows = []
    for slot in ("feed", "feed-random", "later", "later-random"):
        a = arms.get(slot) or {}
        arm_rows.append([
            esc(SLOT_NAMES.get(slot, slot)),
            num(a.get("shown")), num(a.get("opened")), pct(a.get("open_rate")),
            num(a.get("read")), pct(a.get("read_rate")),
            num(a.get("rated_good")), num(a.get("rated_bad")),
        ])
    shown_total = sum((arms.get(s) or {}).get("shown") or 0 for s in SLOT_NAMES)
    live = (
        '<h2>ranked vs random, live</h2><div class="card">'
        '<p class="sub">Every document ever shortlisted, by the arm that chose it, '
        'scored on what you actually did with it. Each random arm is drawn from the '
        'same pool as the ranked picks on the line above it, so the two rows are '
        'directly comparable.</p>'
        + table(["arm", "shown", "opened", "open rate", "read", "read rate",
                 "rated good", "rated bad"], arm_rows,
                ["l", "n", "n", "n", "n", "n", "n", "n"])
        + '<p class="why" style="margin-top:.8rem">Read rate is behavioural and '
          'says nothing about whether it was worth reading -- a document he '
          'finished and then tagged <code>rate:bad</code> counts in both the read '
          'column and the rated-bad column. That is deliberate: collapsing them '
          'would let the ranked arm bank a rejection as a win.</p>' 
        + f'<p class="why" style="margin-top:.8rem">{shown_total} shortlisted documents '
          'total so far. At roughly 10 a night with 2 of them random, this table needs '
          'months before the gap between a ranked row and its control means anything. '
          'It is the metric that will eventually be trustworthy, not the one that is '
          'trustworthy today.</p>' + stale_note + '</div>'
    )

    # -- overnight signals
    signals = last.get("overnight")
    measured = signals is not None
    signals = signals or {}
    sig_rows = [
        ["documents Readwise touched", num(signals.get("updated"))],
        ["newly rated <code>rate:good</code>", num(signals.get("rated_good"))],
        ["newly rated <code>rate:bad</code>", num(signals.get("rated_bad"))],
        ["newly opened", num(signals.get("opened"))],
        ["newly read past the threshold", num(signals.get("read"))],
        ["newly archived", num(signals.get("archived"))],
        ["new arrivals in the feed", num(signals.get("new_docs"))],
    ]
    reasons = (lab.get("by_reason") or {})
    overnight = (
        '<h2>signals captured overnight</h2><div class="card">'
        + (f'<p class="sub">Since the previous sync at '
           f'{esc(str((last.get("sync") or {}).get("since") or "?")[:16].replace("T", " "))}. '
           'These are what tonight\'s retrain saw that last night\'s did not.</p>'
           if measured else
           '<p class="sub">Not measured this run -- the sync was skipped, so the '
           'store was never asked what changed. Zero here would have been a lie.</p>')
        + (table(["", ""], sig_rows) if measured else "")
        + '<p class="sub" style="margin-top:1rem">the whole label set, by where it '
          'came from</p>'
        + bar_chart(sorted(reasons.items(), key=lambda kv: -kv[1]), " labels")
        + '<p class="why" style="margin-top:.8rem">Weakest evidence first: an '
          '"ignored" label just means a feed item went stale unopened, which is why '
          'those carry a 0.3 weight against 3.0 for an explicit rating. Rating things '
          '<code>rate:good</code> / <code>rate:bad</code> in Reader is the highest-value '
          'thing you can do for this system.</p></div>'
    )

    # -- pipeline mechanics
    sync = last.get("sync") or {}
    embed = last.get("embed") or {}
    mechanics = (
        '<h2>pipeline</h2><div class="card">' + table(
            ["stage", "did", "took"],
            [["sync", f'{num(sync.get("documents"))} documents from Readwise'
                      + (" (full backfill)" if sync.get("full") else ""),
              num(sync.get("duration_s"), 1, " s")],
             ["embed", f'{num(embed.get("embedded"))} new vectors'
                       + (f', {num(embed.get("dropped_superseded"))} dropped from a '
                          f'superseded recipe' if embed.get("dropped_superseded") else ""),
              num(embed.get("duration_s"), 1, " s")],
             ["train", ("logistic regression on "
                        f'{num(lab.get("usable"))} labels'
                        if last.get("trained") else
                        "below the label threshold -- ranked by taste vector instead"),
              num(last.get("train_duration_s"), 1, " s")],
             ["select + write", f'{len(picks)} added, {len(evicted)} evicted',
              num(last.get("duration_s"), 1, " s")]],
            ["l", "l", "n"],
        )
        + table(["", ""], [
            ["encoder", esc(last.get("embed_key") or "?")],
            ["corpus by location", " &middot; ".join(
                f"{esc(k or 'none')} {v:,}"
                for k, v in sorted((corpus.get("by_location") or {}).items(),
                                   key=lambda kv: -kv[1]))or DASH],
        ]) + '</div>'
    )

    history = '<h2>run history</h2><div class="card">' + table(
        ["run", "auc", "labels", "pos", "synced", "embedded", "added", "evicted"],
        [[esc((r.get("run_at") or "")[:16].replace("T", " ")),
          num(get(r, "holdout", "auc"), 3),
          num(get(r, "labels", "usable")), num(get(r, "labels", "positives")),
          num(get(r, "sync", "documents")), num(get(r, "embed", "embedded")),
          num(len(r.get("picks") or [])), num(len(r.get("evicted") or []))]
         for r in reversed(rows[-14:])],
        ["l", "n", "n", "n", "n", "n", "n", "n"],
    ) + '</div>'

    subtitle = (f'{len(rows)} run{"" if len(rows) == 1 else "s"} recorded &middot; '
                f'last {esc(ago(last.get("run_at")))} &middot; 02:00 UTC nightly')
    return page("Shortlist recommender", subtitle, status,
                warn_html + head_tiles + perf + shortlist + evictions + live
                + overnight + mechanics + history,
                ("../ai-digest/", "digest status"))


# -- writing ------------------------------------------------------------------

def write_page(name, markup):
    target = Path(config.HTML_SERVE_DIR) / name
    target.mkdir(parents=True, exist_ok=True)
    path = target / "index.html"
    path.write_text(markup, encoding="utf-8")
    return path


def render_digest_page():
    return write_page(config.DIGEST_PAGE, render_digest(stats_store.load("digest")))


def render_recommender_page():
    return write_page(config.RECOMMENDER_PAGE,
                      render_recommender(stats_store.load("shortlist")))


def render_all():
    return [render_digest_page(), render_recommender_page()]


if __name__ == "__main__":
    for written in render_all():
        print(f"[stats] wrote {written}")
    sys.exit(0)
