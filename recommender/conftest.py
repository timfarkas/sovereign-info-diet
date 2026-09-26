"""Makes html_status/ importable from tests run inside this folder.

pytest puts this folder on sys.path (as the test files' own directory), which
is enough for same-directory imports (config, recommender_model, ...) but not
for html_status/, a package one level up at the repo root. This file runs
before any test module in this folder is imported, so the insert lands in
time for their own top-level `from html_status import ...` lines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
