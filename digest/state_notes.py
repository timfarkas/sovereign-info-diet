#!/usr/bin/env python3
"""Per-topic scratchpad notes for cross-run continuity.

Free text, deliberately -- not a structured schema. A schema invites a model
to fabricate fields it does not have an honest answer for; a paragraph it
writes in its own words is easier to read, easier to trust, and easier for a
human to audit on the status page. Every run appends one entry; the file
keeps only the last NOTES_KEPT entries so the prompt does not grow without
bound.

Two kinds of state live here, with different persistence semantics:

* **Rolling notes** (`load`/`append`) -- one short entry per run, FIFO-capped
  at NOTES_KEPT, for "what did recent runs already cover." `append()` always
  adds to what is there.
* **Long-running board** (`load_longrunning`/`save_longrunning`) -- a single
  wholesale-replaced blob for things expected to stay relevant for months (an
  ongoing prosecution, a multi-year buildout). The model re-emits its full
  current understanding of the board every run, so `save_longrunning()`
  overwrites rather than appends -- omitting an item IS how the model closes
  it out.

Both are free text, deliberately -- not a structured schema. A schema invites
a model to fabricate fields it does not have an honest answer for; prose it
writes in its own words is easier to read, easier to trust, and easier for a
human to audit on the status page.

All four topics use this: every template carries the {{prior_notes}}/
{{longrunning}} slots and the trailer instructions that produce entries to
save (shared once in prompts.py). A topic with no history yet gets the
empty-state defaults from `load()`/`load_longrunning()`.
"""
import re
from datetime import datetime
from pathlib import Path

NOTES_DIR = Path("extracts/state_notes")
NOTES_KEPT = 5

_NO_NOTES = "(no notes from prior runs yet -- this is the first.)"
_NO_LONGRUNNING = "(no long-running items tracked yet.)"

_TRAILER_RE = re.compile(
    r"<!--\s*STATE-NOTES-START\s*-->(.*?)<!--\s*STATE-NOTES-END\s*-->",
    re.DOTALL,
)
_LONGRUNNING_RE = re.compile(
    r"<!--\s*LONG-RUNNING-START\s*-->(.*?)<!--\s*LONG-RUNNING-END\s*-->",
    re.DOTALL,
)
_ENTRY_SPLIT_RE = re.compile(r"(?=^## )", re.MULTILINE)


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _path(topic_key: str) -> Path:
    return NOTES_DIR / f"{topic_key}.md"


def _longrunning_path(topic_key: str) -> Path:
    return NOTES_DIR / f"{topic_key}_longrunning.md"


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


def load_longrunning(topic_key: str) -> str:
    """The board exactly as the model last left it, ready to drop into a prompt."""
    p = _longrunning_path(topic_key)
    if not p.exists():
        return _NO_LONGRUNNING
    return p.read_text().strip() or _NO_LONGRUNNING


def save_longrunning(topic_key: str, text: str) -> None:
    """Replace the board wholesale -- the model re-emits it in full each run,
    so a blank or narrower board here is a deliberate close-out, not a bug."""
    text = (text or "").strip()
    p = _longrunning_path(topic_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text + "\n" if text else "")


def extract(html: str) -> tuple:
    """Split a model's raw output into (mailed_html, notes_text, longrunning).

    notes_text is "" when the template had no STATE-NOTES trailer. longrunning
    is None when there was no LONG-RUNNING trailer -- distinct from "", which
    would wipe the saved board -- so a run that forgets the trailer leaves the
    board untouched instead of silently erasing it.
    """
    clean = html
    notes = ""
    m = _TRAILER_RE.search(clean)
    if m:
        notes = _strip_tags(m.group(1))
        clean = (clean[:m.start()] + clean[m.end():])
    longrunning = None
    m2 = _LONGRUNNING_RE.search(clean)
    if m2:
        longrunning = _strip_tags(m2.group(1))
        clean = (clean[:m2.start()] + clean[m2.end():])
    return clean.strip(), notes, longrunning
