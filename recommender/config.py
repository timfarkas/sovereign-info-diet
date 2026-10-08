#!/usr/bin/env python3

# Configuration for the Readwise shortlist recommender.
#
# Ranks the Readwise Reader firehose and tags the best few `shortlist`, which is
# the tag behind Reader's built-in "Shortlist" view. Everything else is left
# alone, so a bad ranking costs the reader nothing but a mediocre shortlist.

import math
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

# Shape of a cycle: two pools, each with its own unranked random picks.
#   feed  -- 9 slots from the fresh firehose, 3 of them random
#   later -- 3 slots resurfaced from the backlog, 1 of them random
# The ranked slots of each pool are split between the two RANKERS below by
# team-draft interleaving, so each ranker fills half of them. Three random feed
# picks a night instead of one (KYRO-30): the random arm is both the yardstick
# and the only unbiased training signal on the new feeds, and at one a night
# it took two weeks to collect fifteen.
# The random pick in each pool is the measurement arm, and it has to come from
# the SAME pool as the ranked picks it is compared against: `later` items differ
# from fresh feed by age and by having already survived a selection step, so a
# random draw from one pool says nothing about ranking in the other.
FEED_SLOTS = 9
FEED_RANDOM_SLOTS = 3
LATER_SLOTS = 3
LATER_RANDOM_SLOTS = 1
SHORTLIST_SIZE = FEED_SLOTS + LATER_SLOTS
RESURFACE_SAMPLE_SIZE = 300    # random draw from `later` that resurfacing ranks
# The fresh-feed random draw is damped against floods (Tim, KYRO-30): a
# publication with n posts in the pool draws with total mass log(1+n), not n,
# so each of its posts with weight log(1+n)/n. Measured 2026-10-08 on the
# 1,896-doc pool: uniform gave bloomberg 62% + guardian 27% of draws and sites
# with <=7 posts a week 4%; sqrt gave 25% + 16% / 39%; log gives 10% + 9% / 55%.
# Still a known-probability draw, so its labels stay honest -- but its open rate
# is now the floor of a de-spammed feed, not of the raw one. The backlog draw
# stays uniform: `later` is what he saved by hand, there is no flood in it.
RANDOM_PUBLICATION_MASS = math.log1p
# Near-duplicates are not hypothetical: the first dry run handed back three
# copies of the same newsletter, eating half the ranked slots. Cosine on the
# embeddings catches both exact repeats and the same story from two sources.
# Only ranked picks are deduplicated -- a random arm that skipped duplicates
# would no longer be a known-probability draw, and it is the measurement.
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
    "rated": 3.0,          # he tagged it rate:good / rate:bad
    "favorited": 2.0,      # favorite / important
    "read": 1.5,           # archived with real reading progress
    "opened": 1.0,         # opened but not finished
    "passed": 1.0,         # shortlisted, shown, evicted unopened
    "ignored": 0.3,        # stale feed item, never opened
}
MIN_LABELS_TO_TRAIN = 40       # below this, fall back to the taste-vector cold start

# The "exposure" labelling scheme (KYRO-30). The `current` scheme above treats
# every stale unopened feed item as a 0.3 negative, which was fine while he
# cleared the feed (Feb-May 2026: ~90% of arrivals opened, though mostly by
# bulk "mark all seen" -- see BULK_CLEAR_OPENS) and stopped being
# fine when he stopped (Jun-Sep: ~3-5%) and then went false when ~300 RSS
# items/day started arriving (Oct: ~0.1%). Most of those were never seen, so
# "ignored" there means "never considered", not "not interested".
# So: positives at full weight; each negative is scaled by how much of its
# arrival block he organically opened -- a skip in a block where he opened 95 of
# 100 is a hard negative, a skip in a block where he opened 3 says ~nothing.
# Organic = never shortlisted: shortlist opens say nothing about the other 99.
# Tim's proposal also gated blocks on >=10 opens; measured redundant with the
# scaling (identical held-out AUC, KYRO-30) and left out.
EXPOSURE_BLOCK = 100
EXPOSURE_WEIGHTS = {
    "rated_good": 2.0,
    "rated_bad": 3.0,
    "read": 0.5,
    "saved": 0.3,          # favourited, or moved from the feed into `later`
    "passed": 0.2,         # shortlisted, evicted unopened -- exposure 1 by construction
    "bounced": 0.1,        # opened, read under MIN_READ           (x block exposure)
    "ignored": 0.2,        # stale feed item, never opened     (x block exposure)
}
# "Read" for the exposure scheme is any real reading: >= OPENED_WORDS words, or
# >= MIN_READ_PROGRESS of a short post (Tim, KYRO-30). On the bulk-free held-out
# set this beat the READ_WORDS/READ_PROGRESS bar (AUC 0.773 vs 0.730). Grading
# positives by sqrt(words read) / fraction read was tried and did not help.
MIN_READ_PROGRESS = 0.10
# Feb-Jun 2026 he cleared the feed with "mark all seen", mostly without reading
# the titles: ~5,900 of ~6,900 feed opens landed in bursts of >= 20 in one
# minute (up to 507 in one second). Such an open is not a bounce, so it gets no
# label at all.
BULK_CLEAR_OPENS = 20

# Publication as a feature: a one-hot over site_name for sites with at least
# this many labelled documents, plus one shared "other" column. Only safe with
# the exposure labels -- with the `current` ones it learns "the Guardian is bad"
# straight out of the October flood, which he never looked at.
PUBLICATION_MIN_DOCS = 15

# Full-text embeddings (KYRO-30): mistral-embed over each document's whole text,
# fetched with the nightly sync (withHtmlContent) and kept as plain text in the
# store. Text = document_text() + the footer-cut body, cut into CHUNK-word pieces
# (CHUNK_CHARS caps base64/url soup, which tokenises at ~1 token/char); the doc
# vector is the word-weighted mean of its chunk vectors, L2-normed. Bump the key
# if any of that changes. ~2.5k tokens/doc at $0.10/M: a 300-doc night is ~$0.08.
MISTRAL_EMBED_KEY = "mistral-embed#fulltext1"
MISTRAL_CHUNK_WORDS = 1500
MISTRAL_CHUNK_CHARS = 8000
MISTRAL_BATCH_WORDS = 9000     # per request
MISTRAL_BATCH_CHUNKS = 32
# A night that would embed more words than this is a bug (a key bump, a lost
# table), not news: it stops loudly instead of quietly spending ~$4 on a re-embed.
# Re-run with --embed-limit, or raise this, on purpose.
MISTRAL_MAX_WORDS_PER_RUN = 5_000_000

# The two rankers the nightly cycle interleaves. `control` is what ships; the
# `candidate` is what is being tested against it. To try a new idea, edit
# `candidate` -- the slate, the arms and the refinement page all follow. Promote
# a candidate by copying it into `control`. "embedding" names the vectors a
# ranker is fitted on (default EMBED_KEY).
# 2026-10-09 (KYRO-30, Tim): promoted exposure labels + publication on mistral
# full text. Offline it beat the old control by ~0.14 AUC; mistral vs bge was a
# wash on opens but ahead on rate:bad and fraction read, with too few ratings to
# be sure -- so the candidate is the same ranker on bge, and the live arms say
# whether the full text earns its keep.
RANKERS = {
    "control": {"scheme": "exposure", "publications": True, "embedding": MISTRAL_EMBED_KEY},
    "candidate": {"scheme": "exposure", "publications": True},
}
# Extra rows for the refinement page's offline table only -- never drafted into
# the live slate. A row whose vectors are not in the store is skipped.
OFFLINE_RANKERS = {
    "retired control (current labels, bge)": {"scheme": "current", "publications": False},
}
