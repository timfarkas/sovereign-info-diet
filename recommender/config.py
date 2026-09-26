#!/usr/bin/env python3

# Configuration for the Readwise shortlist recommender.
#
# Ranks the Readwise Reader firehose and tags the best few `shortlist`, which is
# the tag behind Reader's built-in "Shortlist" view. Everything else is left
# alone, so a bad ranking costs the reader nothing but a mediocre shortlist.

import os

from dotenv import load_dotenv

load_dotenv(os.getenv("DOTENV_PATH") or None)

# No default: a box this depends on should say so loudly rather than quietly
# writing into whichever directory happened to be $HOME at the time.
HOME_DIR = os.environ["HOME_DIR"].rstrip("/")
PROJECT_DIR = f"{HOME_DIR}/projects/ai-news/recommender"

RECOMMENDER_DB = f"{PROJECT_DIR}/data/recommender.sqlite3"

# Local ONNX embeddings. Measured 2026-09-23: onnxruntime with default threads
# and batch size OOM'd this 3.7 GB box while the digest cron was resident, so
# these defaults are small on purpose. The cache dir is NOT under /tmp because
# /tmp gets cleared and re-downloading the model on every boot is silly.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
# Bump whenever document_text() changes what goes into the vector. Stored
# embeddings are keyed by model+version, so a recipe change re-embeds from
# scratch instead of silently mixing two different embedding spaces.
#   v1: title, site, summary
#   v2: + author -- publications and authors repeat constantly in this corpus
EMBED_TEXT_VERSION = 2
EMBED_THREADS = 1
EMBED_BATCH_SIZE = 16
EMBED_CACHE_DIR = f"{PROJECT_DIR}/.model-cache"
EMBED_SUMMARY_CHARS = 1000     # summaries are short; this is a guard, not a budget
EMBED_KEY = f"{EMBED_MODEL}#v{EMBED_TEXT_VERSION}"   # how embeddings are keyed in the store

# Tags. `shortlist` is Reader's own -- verified 2026-09-23 by shortlisting a
# document by hand and reading the tag back off the API. The rating tags are
# ours, and deliberately binary: a scale invites agonising over the middle.
SHORTLIST_TAG = "shortlist"
RATE_GOOD_TAG = "rate:good"
RATE_BAD_TAG = "rate:bad"

# Shape of a cycle: two pools, each with its own unranked random pick.
#   feed  -- 7 slots from the fresh firehose, 1 of them random
#   later -- 3 slots resurfaced from the backlog, 1 of them random
# The random pick in each pool is the measurement arm, and it has to come from
# the SAME pool as the ranked picks it is compared against: `later` items differ
# from fresh feed by age and by having already survived a selection step, so a
# random draw from one pool says nothing about ranking in the other.
FEED_SLOTS = 7
FEED_RANDOM_SLOTS = 1
LATER_SLOTS = 3
LATER_RANDOM_SLOTS = 1
SHORTLIST_SIZE = FEED_SLOTS + LATER_SLOTS
RESURFACE_SAMPLE_SIZE = 300    # random draw from `later` that resurfacing ranks
# Near-duplicates are not hypothetical: the first dry run handed back three
# copies of the same newsletter, eating half the ranked slots. Cosine on the
# embeddings catches both exact repeats and the same story from two sources.
# Only ranked picks are deduplicated -- a random arm that skipped duplicates
# would no longer be a uniform draw, and it is the measurement.
DEDUP_SIMILARITY = 0.93

# Length balance. Left alone the ranker builds a long-form monoculture: measured
# 2026-09-24, the fresh-feed pool is 57% under 800 words but the ranked picks
# came out 67% over 2500, a 5.6x over-representation. That is a real learned
# preference, not a bug -- he does read long-form -- but it leaves nothing on the
# list for a five-minute gap. So each pool reserves a floor of short picks.
# Costs some predicted relevance by construction; the random arms will show
# whether that costs anything real.
SHORT_WORDS = 800
FEED_SHORT_SLOTS = 2       # of the 6 ranked feed picks
LATER_SHORT_SLOTS = 1      # of the 2 ranked later picks
RESHOW_COOLDOWN_DAYS = 60      # do not wave the same document around again
FEED_CANDIDATE_DAYS = 7        # how far back a fresh feed item can be and still qualify

# Labelling. Absence of a positive signal is not the same as a negative one --
# most of these are deliberately weak, and several only fire at all when there
# is contextual evidence he was actually triaging that day.
FEED_STALE_DAYS = 4
# What counts as having read something, measured in words actually consumed
# (reading_progress x word_count) rather than percentage. Percentage alone is
# biased against long-form: measured 2026-09-24 on the real corpus, a bare
# progress>0.5 rule missed 68 documents where he read 500+ words without
# finishing, and credited 65 where he "finished" under 200 words. The progress
# clause stays as an OR so that deliberately finishing something short still
# counts.
READ_WORDS = 500
READ_PROGRESS = 0.8
OPENED_WORDS = 100     # below this an open says nothing either way
LABEL_WEIGHTS = {
    "rated": 3.0,           # he tagged it rate:good / rate:bad
    "favorited": 2.0,       # favorite / important
    "read": 1.5,            # archived with real reading progress
    "opened": 1.0,          # opened but not finished
    # Shortlisted, shown, evicted unopened -- but only on a day he mostly kept
    # up with the shortlist (see CYCLE_READ_MAJORITY below). On a quiet day,
    # skipping something says nothing; on a day he read most of the list, it
    # does. Deliberately small: rate:bad is the signal to lean on going
    # forward, this is just not throwing away a slight one.
    "passed": 0.2,
    # Archived without ever being opened -- an explicit "no", not neglect.
    "archived_unread": 0.3,
    # A stale, never-opened feed item -- but only counted as a negative on a
    # day he was actively shelving other feed items too (see SHELVED below).
    # Otherwise the firehose simply outran him, which is not a taste signal.
    "ignored": 0.1,
}
# A cycle's shortlist counts as "he mostly kept up" once more than this
# fraction of it was actually read.
CYCLE_READ_MAJORITY = 0.5
MIN_LABELS_TO_TRAIN = 40       # below this, fall back to the taste-vector cold start
