#!/usr/bin/env python3
"""Configuration for the shared status-page engine.

Deliberately independent of digest/config.py and recommender/config.py --
html_status/ is used by both jobs and must not require either one's
dependencies or environment just to import.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.getenv("DOTENV_PATH") or None)

# History lives at the repo root, one level up from this package, so it
# outlives whichever job wrote to it most recently and isn't nested inside
# either job's own folder.
STATS_DIR = str(Path(__file__).resolve().parent.parent / "data" / "run_stats")
STATS_KEEP_RUNS = 180          # ~6 months of nightly rows, then the oldest fall off

# Opt-in: unset HTML_SERVE_DIR and nothing under html_status/ ever writes a
# page. No code-level default -- see the "status page" section of README.md.
HTML_SERVE_DIR = os.getenv("HTML_SERVE_DIR")
DIGEST_PAGE = "ai-digest"      # -> <HTML_SERVE_DIR>/ai-digest/
RECOMMENDER_PAGE = "recommender"


def digest_stats_kind(topic_key):
    """History file name for a topic. Mirrors digest/topics.py Topic.stats_kind.

    Duplicated rather than imported on purpose: importing digest/topics.py would
    drag digest/config.py, its environment and its dependencies into the shared
    renderer, which is exactly what this module's docstring says it must not do.
    The AI topic keeps the bare `digest` name so months of existing history stay
    readable with no migration.
    """
    return "digest" if topic_key == "ai" else f"digest-{topic_key}"


def digest_page(topic_key):
    """Directory under HTML_SERVE_DIR for a topic. Mirrors Topic.page."""
    return DIGEST_PAGE if topic_key == "ai" else f"digest-{topic_key}"


def digest_topic_keys():
    """Topic keys with history on disk, AI first -- and AI always, history or not.

    Discovered from the stats directory rather than configured, so a new topic in
    digest/topics.py shows up here the first time it records a run and nothing
    has to be kept in sync by hand.

    `ai` is unconditional because the AI digest page must exist on a box that has
    never run the pipeline -- "no run recorded yet" is that page's job, and a
    missing page looks like a webserver fault instead. Caught by
    recommender/test_stats.py, which renders both pages into an empty tmpdir.
    """
    found = ["ai"]
    for f in sorted(Path(STATS_DIR).glob("digest*.jsonl")):
        stem = f.stem
        key = "ai" if stem == "digest" else stem[len("digest-"):]
        if key not in found:
            found.append(key)
    return sorted(found, key=lambda k: (k != "ai", k))
