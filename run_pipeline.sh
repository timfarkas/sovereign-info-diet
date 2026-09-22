#!/bin/bash
# Daily AI digest: X + Reddit -> LLM summary -> email.
# Neither source leg is allowed to kill the run. The summarizer refuses to mail
# an empty digest if BOTH produced nothing, and stamps a maintenance banner on
# the mail when a leg is unhealthy -- so Tim learns about breakage from his
# inbox rather than from the digest quietly not arriving.
cd /home/kyro/projects/ai-news
export DOTENV_PATH=/home/kyro/projects/ai-news/.env
PY=/home/kyro/projects/ai-news/.venv/bin/python

echo "=== $(date -u +%FT%TZ) pipeline start ==="

$PY x_scraper.py      || echo "[pipeline] x_scraper.py exited $? -- continuing, digest will flag it"
$PY reddit_scraper.py || echo "[pipeline] reddit_scraper.py exited $? -- continuing, digest will flag it"

$PY llm_summarizer.py && $PY send_notification.py
rc=$?
echo "=== $(date -u +%FT%TZ) pipeline end (rc=$rc) ==="
exit $rc
