#!/usr/bin/env python3
"""Append-only run history for both nightly jobs.

One JSON object per run, one file per job, newest last. That is the whole
storage layer -- no schema, no migrations, no second sqlite file. A row is
whatever the job knew about itself when it finished; the renderer reads
defensively because old rows will always be missing fields newer rows have.

Kept separate from the renderer on purpose: recording must be cheap and
boring enough that a job can do it unconditionally, while rendering is
allowed to be slow and opinionated.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config


def _path(kind):
    return Path(config.STATS_DIR) / f"{kind}.jsonl"


def record(kind, payload, keep=None):
    """Append one run to `kind`'s history and trim to the last `keep` rows.

    Rewrites the file when it trims -- at 180 rows of a few KB that is
    nothing, and it keeps the file a plain readable jsonl with no vacuum step.
    """
    keep = keep or config.STATS_KEEP_RUNS
    path = _path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    rows = load(kind)
    if len(rows) > keep:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n"
                    for r in rows[-keep:]),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    return path


def load(kind, limit=None):
    """Every recorded run, oldest first. A corrupt line is skipped, not fatal.

    A half-written row (OOM kill mid-append) must not take the status page
    down with it -- the page is the thing that would have told you about the
    kill in the first place.
    """
    path = _path(kind)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:] if limit else rows


def latest(kind):
    rows = load(kind, limit=1)
    return rows[0] if rows else None
