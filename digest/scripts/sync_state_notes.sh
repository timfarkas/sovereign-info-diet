#!/usr/bin/env bash
# Pull/push the ai-news state-notes scratchpads between this machine and the
# box, for editing by hand.
#
# Run this FROM YOUR OWN MACHINE, not on the box -- it is two rsync calls,
# nothing more. There is no sync daemon and nothing new running on the box;
# it just moves files over the ssh access you already have.
#
# Deliberately does NOT cover digest/prompts.py. That file is tested,
# git-reviewed source (test_topics.py asserts on its structure), and a live
# edit-in-place shortcut would bypass the review the ticket workflow already
# gives it. It stays on the existing path: tell Kyro what to change, it
# edits, tests and opens a PR. This script only touches the freeform runtime
# notes under extracts/state_notes/, which the pipeline itself already
# overwrites every run and which carry no such review requirement.
#
# Usage:
#   export SYNC_HOST=<your ssh alias for the box>
#   ./sync_state_notes.sh pull [topic|all]   # box -> ./state_notes/ (here)
#   ./sync_state_notes.sh push [topic|all]   # ./state_notes/ (here) -> box
#
# "topic" is a topic key (e.g. europe, pandemic, geopolitics) and matches
# both its rolling-notes file and its long-running-board file. Default is
# "all" -- every topic's files.

set -euo pipefail

HOST="${SYNC_HOST:?set SYNC_HOST to your ssh alias for the box first}"
REMOTE_DIR="~/projects/ai-news/digest/extracts/state_notes/"
LOCAL_DIR="./state_notes/"
ACTION="${1:?usage: $0 pull|push [topic|all]}"
TOPIC="${2:-all}"

mkdir -p "$LOCAL_DIR"

pattern="*.md"
[ "$TOPIC" != "all" ] && pattern="${TOPIC}*.md"

case "$ACTION" in
  pull)
    rsync -avz --include="$pattern" --exclude='*' "$HOST:$REMOTE_DIR" "$LOCAL_DIR"
    ;;
  push)
    rsync -avz --include="$pattern" --exclude='*' "$LOCAL_DIR" "$HOST:$REMOTE_DIR"
    ;;
  *)
    echo "usage: $0 pull|push [topic|all]" >&2
    exit 1
    ;;
esac
