#!/bin/bash
# Nightly Readwise shortlist cycle. Deliberately NOT part of run_pipeline.sh:
# it has its own venv (onnxruntime and scikit-learn have no business anywhere
# near the digest's dependencies) and its own slot at 02:00, an hour after the
# digest, so the two never hold their memory at the same time on a 3.7 GB box.
#
# Resolved relative to this script, not a hardcoded home dir, so the repo runs
# the same on any box it's cloned onto.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export DOTENV_PATH="$SCRIPT_DIR/../.env"
PY="$SCRIPT_DIR/.venv-rec/bin/python"

echo "=== $(date -u +%FT%TZ) shortlist start ==="
$PY shortlist_job.py "$@"
rc=$?
echo "=== $(date -u +%FT%TZ) shortlist end (rc=$rc) ==="
exit $rc
