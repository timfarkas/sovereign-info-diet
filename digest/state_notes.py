#!/usr/bin/env python3
"""Per-topic scratchpad notes for cross-run continuity.

Free text, deliberately -- not a structured schema. A schema invites a model
to fabricate fields it does not have an honest answer for; a paragraph it
writes in its own words is easier to read, easier to trust, and easier for a
human to audit on the status page. Every run appends one entry; the file
keeps only the last NOTES_KEPT entries so the prompt does not grow without
bound.

Only the three shared-skeleton topics use this (their template is the only
one with the {{prior_notes}} slot and the trailer instructions that produce
an entry to append -- see prompts.py). The AI digest's own template never
references either, so `load()` for "ai" returns the empty-state default and
`append()` is simply never called for it from digest_run.py.
"""
import re
from datetime import datetime
from pathlib import Path

NOTES_DIR = Path("extracts/state_notes")
NOTES_KEPT = 5

_NO_NOTES = "(no notes from prior runs yet -- this is the first.)"

_TRAILER_RE = re.compile(
    r"<!--\s*STATE-NOTES-START\s*-->(.*?)<!--\s*STATE-NOTES-END\s*-->",
    re.DOTALL,
)
_ENTRY_SPLIT_RE = re.compile(r"(?=^## )", re.MULTILINE)


def _path(topic_key: str) -> Path:
    return NOTES_DIR / f"{topic_key}.md"


def load(topic_key: str) -> str:
    """The prior runs' notes, oldest first, ready to drop into a prompt."""
    p = _path(topic_key)
    if not p.exists():
        return _NO_NOTES
    return p.read_text().strip() or _NO_NOTES


def append(topic_key: str, entry: str, now: datetime = None) -> None:
    """Add one run's note, keeping only the last NOTES_KEPT entries."""
    entry = (entry or "").strip()
    if not entry:
        return
    now = now or datetime.now()
    p = _path(topic_key)
    existing = p.read_text() if p.exists() else ""
    block = f"## {now.strftime('%Y-%m-%d')}\n{entry}\n"
    entries = [e.strip() for e in _ENTRY_SPLIT_RE.split(existing) if e.strip()]
    entries.append(block.strip())
    entries = entries[-NOTES_KEPT:]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n\n".join(entries) + "\n")


def extract(html: str) -> tuple:
    """Split a model's raw output into (mailed_html, notes_text).

    notes_text is "" when the template had no trailer to begin with -- the
    AI digest, which never asks for one, round-trips untouched.
    """
    m = _TRAILER_RE.search(html)
    if not m:
        return html, ""
    notes = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    clean = (html[:m.start()] + html[m.end():]).strip()
    return clean, notes
