# State of this box

What is running, what it is doing, and what is known to be wrong with it.
Last updated 2026-09-30.

## Two independent jobs

| | digest | shortlist recommender |
|---|---|---|
| folder | `digest/` | `recommender/` |
| entry point | `digest/run_pipeline.sh` | `recommender/run_shortlist.sh` |
| cron | 01:00 UTC | 02:00 UTC |
| log | `~/logs/ai-digest.log` | `~/logs/shortlist.log` |
| venv | `digest/.venv` | `recommender/.venv-rec` |
| what it does | X + Reddit + subscribed feeds → LLM → one digest email per topic | ranks Readwise Reader, tags `shortlist` |
| fails how | silently, into its log | silently, into its log |

They are deliberately separate: different dependencies (onnxruntime and
scikit-learn have no business near the digest), and an hour apart so they never
hold memory simultaneously on a 3.7 GB box.

**Neither job tells anyone when it breaks** -- nothing pushes an alert. What
exists now is a pull surface, in `html_status/` (shared, at the repo root):
each run appends a JSON row to `data/run_stats/<job>.jsonl` unconditionally,
and -- only if `HTML_SERVE_DIR` is set in `.env` -- re-renders a static page
there. Set on this box to `/home/kyro/html_serve/`, reachable on the WireGuard
mesh only:

| | page |
|---|---|
| digest — AI/tech | <http://192.168.2.6:8080/ai-digest/> |
| digest — geopolitics | <http://192.168.2.6:8080/digest-geopolitics/> |
| digest — pandemic | <http://192.168.2.6:8080/digest-pandemic/> |
| digest — europe | <http://192.168.2.6:8080/digest-europe/> |
| recommender | <http://192.168.2.6:8080/recommender/> |

Both carry a status pill and a "flagged this run" box, so a dead leg is one
glance rather than one `grep`. Two things the pages have already shown that
this file only asserted: the category one-hots contribute roughly +2.9 log-odds
to a backlog `article` against -0.6 to a feed `rss` item, which is most of the
gap between the two pools and is visible per pick under "why this score"; and
`class_weight="balanced"` puts the average document at ~52%, so a score is not
a probability that he will read the thing. Rebuild either from history without
running a job: `python -m html_status.stats_page` (prints one line and exits
if `HTML_SERVE_DIR` is unset). The rendering call is wrapped in a try/except in
both jobs on purpose -- a rendering bug must never be the reason a digest does
not go out, or the reason a cycle that already wrote tags to Readwise reports
failure. It prints the traceback rather than swallowing it.

## The digest, as built

One nightly run, four topics, defined as rows in `digest/topics.py`. The AI
digest runs every night; geopolitics, pandemic preparedness and Europe run every
three days each, staggered by nothing more than when they last succeeded. Each
source is scraped once and sliced per topic.

| topic | cadence | window | subreddits | direct feeds | ceiling | measured |
|---|---|---|---|---|---|---|
| AI/tech | nightly | 2d | 5 | — | $0.25 | $0.15–0.24 |
| geopolitics, markets, supply chains | 3d | 3d | 5 | 6 | $0.15 | $0.089 |
| pandemic preparedness & bio-risk | 3d | 3d | 5 | 5 | $0.15 | $0.034 |
| Europe, the EU & liberal values | 3d | 3d | 4 | 6 | $0.15 | $0.057 |

Measured on the live corpus 2026-09-30, `gpt-6-sol` flex tier. The three new
topics add roughly $0.06/day amortised on top of the AI digest's ~$0.20, and the
feed leg itself is free (Readwise plus direct fetches, no per-item billing). X
credits do not change at all: the topics read the dumps already on disk rather
than widening the X query.

Known and deliberate:

- **The AI digest is a topic row, not a special case, but its values are pinned.**
  `digest/test_topics.py` asserts each of them, and the prompt it builds is
  byte-identical to the pre-multi-topic code's on the same corpus (verified by
  diffing a 420 KB prompt against a pristine checkout). If you edit the AI row,
  those tests are the thing that tells you.
- **The AI topic draws no feed items.** Its contract is the pre-existing one, X +
  Reddit. An empty keyword set means "take everything" for tweets, so the feed
  selector needs an explicit guard -- without it the AI digest filled up with
  Austrian domestic politics.
- **The `feed_urls` lists are a starting point, not his subscriptions.** The
  primary path is Readwise, i.e. whatever he actually subscribes to; these are the
  independent fallback and are deliberately short and high-signal. Two obvious
  additions (`thediplomat.com`, `freightwaves.com`) are live but publish ~45 items
  per window each, which would pin a topic against its cap permanently and turn
  selection into "whatever was newest".
- **`bruegel.org`, `ecfr.eu`, `brookings.edu` and `outbreaknewstoday.com` 403/302
  a scripted fetch.** Dropped rather than worked around: a feed that needs a
  browser-shaped request is a feed whose owner does not want one.
- **WHO's Disease Outbreak News feed is 404 at every documented URL.** WHO reaches
  the pandemic topic by keyword only.

## The box

2 vCPU (Xeon Skylake), 3.7 GB RAM, ~16 GB free disk, 2 GB swap. Measured costs:

- full Readwise sync, 14.6k documents: 7m51s, 162 MB peak (rate-limit bound)
- embedding 14.6k documents: 52 min, ~580 MB peak, ~4.4 docs/s
- nightly incremental: seconds

A local *generative* LLM does not fit (7B at q4 wants ~4.5 GB and would collide
with the digest). Local *embeddings* fit comfortably. Keep any single job under
~900 MB.

## The recommender, as built

`sync → embed → train → select → write`, nightly.

- **encoder**: `BAAI/bge-small-en-v1.5`, int8 ONNX via fastembed, 384-dim,
  L2-normalized, CPU, `threads=1`, `batch=16`. **Frozen** — never fine-tuned.
- **embedded text** (recipe v2): `title \n author \n site_name \n summary[:1000]`.
  Changing this means bumping `EMBED_TEXT_VERSION`, which forces a clean re-embed.
- **head**: `LogisticRegression`, L2, `C=1.0`, `class_weight="balanced"`, plus
  per-tier sample weights. 392 features: the embedding, `log1p(word_count)`, and
  seven category one-hots. Retrained from scratch nightly (milliseconds).
- **slate**: 10 documents — 7 fresh feed (1 an unranked random draw) and 3
  resurfaced from `later` (1 unranked). Ranked picks are deduplicated by cosine
  ≥ 0.93 and reserve a floor of short reads; the random arms are neither
  deduplicated nor length-constrained, because they are the measurement.

### Numbers as of 2026-09-24

Held-out time split, evaluated against the length-aware labels:

| holdout | trained on percentage labels | trained on words-read labels |
|---|---|---|
| 30 days (824 docs) | AUC 0.666 | AUC 0.661 |
| 60 days (1721 docs) | AUC 0.646 | **AUC 0.659** |
| 120 days (4538 docs) | AUC 0.558 | **AUC 0.569** |

The 30-day window holds ~34 positives and cannot resolve a difference this size;
the larger windows can, and they favour words-read consistently. Against the
random and word-count baselines the model gets AUC ~0.67 vs 0.46 and 0.52.

Adding author to the embedded text moved AUC 0.658 → 0.682, measured with
everything else held fixed.

## Corpus facts (full sync, 2026-09-24)

14,656 documents: 11,784 `feed`, 2,036 `later`, 809 `archive`. Backlog reaches
back to 2019. Author is populated on 99%.

Read rates vary enormously by publication — Astral Codex Ten 21 reads of 146
items, reuters.com 8 of 1,183 — which is why author and site name are embedded.

Labels derived: 6,199, of which 848 positive (measured 2026-09-29). The bulk --
5,337 -- is `ignored`: stale feed items he never opened. That is crude, and it
is also the only thing giving the ranker something to push against.

**The 2026-09-25 labelling change was reverted on 2026-09-29 (ticket KYRO-22).**
It had gated both weak negatives on same-day corroborating evidence, added an
`archived_unread` reason, and dropped the negative weights to 0.2 / 0.1. The
argument was good and the outcome was bad -- Tim reported the recommendations
got worse, and the logs agree:

| run | labels | positives | positive rate | AUC | AUC of a random ranker on the same split |
|---|---|---|---|---|---|
| 09-24, before | 6,026 | 812 | 4.9% | 0.690 | 0.505 |
| 09-25, before | 6,045 | 821 | 5.9% | 0.721 | 0.457 |
| 09-28, after | 1,293 | 836 | 53.9% | 0.525 | **0.539** |
| 09-29, after | 1,304 | 845 | 56.8% | 0.534 | 0.467 |
| 09-29, reverted | 6,199 | 848 | 8.9% | 0.742 | 0.521 |

Gating removed ~5,100 negatives and left a training set that was 54-57%
positive, at which point the model ranked no better than shuffling. Compare
only *within* a row: each labelling scheme defines its own holdout, so the AUC
column is not comparable down the table, but model-vs-random on one split is.

If this gets attempted again, the way to do it is online: keep the labels and
A/B the two rankers against the random arms over a couple of weeks. Do not
trust an offline AUC to adjudicate it -- see "known wrong" #1 below.

## Invariants — things that will silently break if changed

- **Age must never become a model feature.** Old → archived → read, so age
  predicts the label almost perfectly and yields a model that ranks by "is old"
  while scoring beautifully. Recency gates candidacy instead. There is a test.
- **Each random arm must be drawn from the same pool as the ranked picks it is
  compared against.** An earlier version drew random from `later` while ranking
  `feed` and called it unbiased; it was not.
- **Only remove the `shortlist` tag from documents this job added.** Every add
  and eviction is logged; a hand-shortlisted document is never touched.
- **Ratings override behaviour.** `rate:bad` on something fully read is a
  negative.

## Known wrong, in rough priority order

1. **The offline evaluation leaks.** Labels come from the corpus as it stands
   now, so something read yesterday counts as a positive inside the training half
   of an older split, and a new arrival can be counted as ignored before it had a
   chance. The fix is point-in-time label reconstruction with per-day slates.
   Until then the offline number is a smoke test, and the live ranked-vs-random
   comparison is the real metric.
2. **Negatives conflate "not interested" with "never considered".** Readwise
   exposes no impression event, so *shown-and-skipped* cannot be distinguished
   from *never-seen* except through the eviction log. The affordable fix is
   explicit `rate:bad`.
3. **Feature scales are unaudited.** `log1p(word_count)` runs 6–9 against
   embedding components near 0.05, and `class_weight="balanced"` multiplies with
   the tier weights, so the optimized loss is not the one the config describes.
4. **No label decay.** Taste from 2019 counts the same as taste from 2026.
5. **Scores are not comparable across pools.** `later` items score ~0.99 and feed
   items ~0.7, because location correlates with the label. Fine for ranking
   within a pool, meaningless between them.
6. **Titles are sometimes junk at the source** (FT saves articles as "Subscribe
   to read"). Not a ranking problem; a metadata one.

## Operating it

```bash
cd recommender
./run_shortlist.sh --dry-run      # decide everything, write nothing
./run_shortlist.sh --full-sync    # re-pull every document
.venv-rec/bin/python -m pytest test_shortlist.py --timeout 5
```

Rating documents `rate:good` / `rate:bad` in Reader is the highest-value thing a
human can do for this system; everything else it harvests silently.
