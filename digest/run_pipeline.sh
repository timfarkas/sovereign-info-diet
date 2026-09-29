#!/bin/bash
# Nightly digests: X + Reddit + subscribed feeds -> per-topic LLM summaries -> email.
# No source leg is allowed to kill the run. digest_run.py refuses to mail an empty
# digest, isolates each topic in its own try/except, and stamps a maintenance
# banner on the mail when a leg is unhealthy -- so Tim learns about breakage from
# his inbox rather than from a digest quietly not arriving.
# Resolved relative to this script, not a hardcoded home dir, so the repo runs
# the same on any box it's cloned onto.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export DOTENV_PATH="$SCRIPT_DIR/../.env"
PY="$SCRIPT_DIR/.venv/bin/python"

echo "=== $(date -u +%FT%TZ) pipeline start ==="

# One scrape per source serves every topic -- X credits are per-tweet, so a
# per-topic scrape would pay four times for largely the same corpus.
$PY x_scraper.py      || echo "[pipeline] x_scraper.py exited $? -- continuing, digests will flag it"
$PY reddit_scraper.py || echo "[pipeline] reddit_scraper.py exited $? -- continuing, digests will flag it"
$PY rss_scraper.py    || echo "[pipeline] rss_scraper.py exited $? -- continuing, digests will flag it"

# Summarize and mail every topic that is due tonight. Each topic mails its own
# file, which is why send_notification.py is no longer a separate leg: with four
# summaries in one directory, "mail the newest one" is ambiguous.
$PY digest_run.py "$@"
rc=$?
echo "=== $(date -u +%FT%TZ) pipeline end (rc=$rc) ==="
exit $rc
