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
