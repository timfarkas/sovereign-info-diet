# State of this box

What is running, what it is doing, and what is known to be wrong with it.
Last updated 2026-09-25.

## Two independent jobs

| | digest | shortlist recommender |
|---|---|---|
| folder | `digest/` | `recommender/` |
| entry point | `digest/run_pipeline.sh` | `recommender/run_shortlist.sh` |
| cron | 01:00 UTC | 02:00 UTC |
| log | `~/logs/ai-digest.log` | `~/logs/shortlist.log` |
| venv | `digest/.venv` | `recommender/.venv-rec` |
| what it does | X + Reddit → LLM → digest email | ranks Readwise Reader, tags `shortlist` |
| fails how | silently, into its log | silently, into its log |

They are deliberately separate: different dependencies (onnxruntime and
scikit-learn have no business near the digest), and an hour apart so they never
hold memory simultaneously on a 3.7 GB box.

**Neither job tells anyone when it breaks.** The digest's absence is the only
alarm for one; a stale shortlist is the only alarm for the other.

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

Labels derived: ~6,000, of which ~800 positive, as of 2026-09-24 -- **stale
now.** The labelling rules changed 2026-09-25: `ignored` (stale, never-opened
feed item) and `passed` (shortlisted, evicted unopened) now only fire when
there is same-day corroborating evidence (something else was archived unread
that day / most of that day's shortlist was actually read), instead of firing
unconditionally. A dry run against the live corpus post-change measured 1,237
labelled (832 positive) -- most of the old unconditional "stale feed, never
opened" bulk no longer qualifies, and a new `archived_unread` reason picks up
part of the slack. Net effect: far fewer labels, but each one should be a
cleaner signal. Worth a fresh measurement pass once this has run for a while.

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
