#!/usr/bin/env python3

# Configuration for the digital info filter

TIME_HORIZON_DAYS = 2   # look back this many days
POSTS_TO_ANALYZE = 60   # reddit posts to scrape for analysis

# Model for the digest. gpt-6-sol on the flex service tier: measured at well
# under $0.20/digest on standard tier before the corpus grew, and flex is half
# price for the same model -- we are a 01:00 cron, so latency costs us nothing.
# gpt-6-astra is the better model but ~4x the ceiling at this prompt size.
SUMMARY_MODEL = "gpt-6-sol"
SUMMARY_SERVICE_TIER = "flex"     # "flex" = half price, slower. "default" = standard.

# $ per 1M tokens (input, cached input, output) for SUMMARY_MODEL on the tier
# above, so the run can print what it actually cost. Two numbers for the one
# model we use, not a table of every model -- re-check when you change either.
SUMMARY_MODEL_PRICE = (1.00, 0.10, 5.00)
SUMMARY_COST_CEILING_USD = 0.25   # exceeding it prints a warning
# Measured 2026-09-23 on a real digest: 489,224 chars -> 140,636 input tokens.
CHARS_PER_TOKEN = 3.48

# --- linked-page enrichment --------------------------------------------------
# Fetch the pages posts point at so the summarizer can read the source instead
# of reporting that it cannot. Budgeted: pages x chars is prompt tokens, and
# prompt tokens are the cost ceiling above.
LINK_FETCH_ENABLED = True
LINK_FETCH_MAX_PAGES = 30
LINK_FETCH_MAX_CHARS = 2000        # per page, after text extraction
REDDIT_SELFTEXT_CHARS = 1200       # was 200 -- the model was reasoning about stubs
REDDIT_COMMENT_CHARS = 500         # was 150


# --- X/Twitter ingestion (via twitterapi.io) ---------------------------------
# The account universe is the union of whoever these accounts follow.
X_SEED_ACCOUNTS = ["FarkasTim", "IsaakFreeman", "johannes_hage"]
X_ACCOUNT_LIST_TTL_DAYS = 7      # following lists move slowly; don't re-pay daily
# X search silently returns nothing for queries past ~500 chars (measured
# 2026-09-22: 466 chars fine, 514 chars -> 0 results for accounts that posted).
# So batches are packed by query length with headroom, not by account count.
X_MAX_QUERY_CHARS = 400
X_MAX_ACCOUNTS_PER_BATCH = 20
# Circuit breakers, not budgets: we want EVERY post from the account list, and
# pagination stops on its own when a batch is exhausted. These only exist so a
# runaway loop cannot spend unbounded twitterapi.io credit (~15 credits/tweet).
X_MAX_TWEETS_PER_RUN = 8000
X_MAX_PAGES_PER_BATCH = 50       # 20 tweets/page, so 1000 tweets per batch
X_MAX_ACCOUNTS = None            # None = all of them
X_MAX_TWEETS_IN_PROMPT = None    # None = every tweet we fetched reaches the model

# r/ControlProblem added for the alignment/safety/x-risk half of the brief.
# r/AI is NOT here on purpose: it 404s, which is why an earlier commit dropped it.
SUBREDDITS = ["singularity", "DeepLearning", "MachineLearning", "LocalLLaMA",
              "ControlProblem"]

SORT_BY = "top"  # "hot", "new", or "top" - top gets best posts from the period

# --- twitterapi.io spend ------------------------------------------------------
# The API bills in credits and exposes only a balance, never a price. Measured
# on 2026-09-25 by fetching a known number of tweets and watching the balance:
# ~15 credits per tweet, and -- the part that matters -- the debit lands tens of
# seconds AFTER the call returns, so reading the balance immediately reports
# zero spend. Hence the settle wait.
TWITTERAPI_CREDIT_SETTLE_SECONDS = 45
# UNVERIFIED. twitterapi.io's own dashboard is the only source for this; 15
# credits/tweet against their published $0.15/1k tweets implies 100k credits per
# dollar, which is where this number comes from, but nobody has checked it
# against an invoice. Set it to None to make the page show credits only.
TWITTERAPI_CREDITS_PER_USD = 100_000


# --- subscribed feeds and newsletters (the RSS leg) ---------------------------
# Two paths, both live; see rss_scraper.py for why neither is a fallback for the
# other. Turning both off leaves the topic digests with X and reddit only, which
# is a much worse digest but not a broken one.
RSS_READWISE_ENABLED = True
RSS_DIRECT_ENABLED = True
# `rss` is what the brief literally asked for. `email` is here because that is
# where his actual analysis lives -- measured 2026-09-29 against the live Reader
# API, Money Stuff / ChinaTalk / Noahpinion / SemiAnalysis / Sentinel are all
# category=email newsletters forwarded into the feed, and an rss-only filter
# handed the geopolitics topic wire headlines with none of the interpretation.
RSS_CATEGORIES = ("rss", "email")
# Wide enough to cover the longest topic window (3 days) plus a missed cron.
RSS_WINDOW_DAYS = 4
RSS_MAX_ITEMS_PER_FEED = 40         # per direct feed, per run
RSS_SUMMARY_MAX_CHARS = 1200        # feed abstracts; the body comes from link_fetcher
RSS_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
# His own digests are auto-forwarded into the Reader feed, so without this the
# pipeline summarises its own output and the summary-of-a-summary compounds
# nightly. Matched on author and on title prefix because the forwarded copy
# keeps both.
RSS_EXCLUDE_AUTHORS = ("Meta Minsky",)
RSS_EXCLUDE_TITLE_PREFIXES = ("AI Digest", "Geopolitics, Markets",
                              "Pandemic Preparedness", "Europe, the EU")


# --- reddit quota -------------------------------------------------------------
# Was POSTS_TO_ANALYZE // len(SUBREDDITS) == 12, computed over the AI topic's
# five subreddits. The union across all four topics is 19 subreddits, so keeping
# that formula would have silently cut the AI digest from 12 posts per subreddit
# to 3. It is a fixed per-subreddit quota now, and POSTS_TO_ANALYZE stays as the
# AI topic's own budget so nothing else that reads it changes meaning.
POSTS_PER_SUBREDDIT = POSTS_TO_ANALYZE // len(SUBREDDITS)
