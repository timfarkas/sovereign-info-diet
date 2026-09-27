# digital info filter

Two independent nightly jobs sharing one repo and one `.env`:

- **`digest/`** — X/Twitter + Reddit → linked-page enrichment → llm gate → daily
  digest email.
- **`recommender/`** — ranks the Readwise Reader firehose and tags the best few
  `shortlist`, documented in its own section below.

They live in separate folders on purpose: separate dependencies (onnxruntime
and scikit-learn have no business near the digest), separate venvs, and
separate cron slots an hour apart so the two never hold memory at the same
time on a 3.7 GB box.

`.env` lives at the repo root and is shared by both. It needs:
`HOME_DIR` (absolute path to your home directory -- used to build absolute
paths for cron, since cron does not run with your shell's `$HOME`),
`REDDIT_CLIENT_ID`, `REDDIT_SECRET`, `TWITTER_IO_API_KEY`, `OPENAI_API_KEY`,
`EMAIL_FROM`, `EMAIL_TO` (digest), and `READWISE_API_KEY` (recommender).
`HTML_SERVE_DIR` is optional, for both -- see "status page" below.

## digest/

```bash
cd digest
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### the pipeline

`digest/run_pipeline.sh` (cron, 01:00 UTC) runs four legs:

| leg | what it does |
|---|---|
| `x_scraper.py` | pulls X timelines via twitterapi.io for the union of who `X_SEED_ACCOUNTS` follow |
| `reddit_scraper.py` | pulls `SUBREDDITS` via praw |
| `llm_summarizer.py` | fetches linked pages (`link_fetcher.py`), then summarizes all three sources |
| `send_notification.py` | mails the latest summary |

Either source leg may fail without killing the run; the summarizer refuses to mail
an empty digest and stamps a maintenance banner on the mail when a leg is
unhealthy, so breakage shows up in the inbox rather than in silence.

### knobs

All in `digest/config.py`. The ones that move cost or shape:

- `SUMMARY_MODEL` / `MODEL_PRICING` / `SUMMARY_COST_CEILING_USD` — every run prints
  its measured cost and warns past the ceiling.
- `TIME_HORIZON_DAYS` — digest window. Widening it is safe: the seen-ids store
  stops already-summarized posts from reappearing.
- `X_MAX_TWEETS_IN_PROMPT`, `LINK_FETCH_MAX_PAGES`, `LINK_FETCH_MAX_CHARS` — the
  three levers on prompt size, i.e. on cost.
- `X_MAX_QUERY_CHARS` — leave it under ~450. X search silently returns zero
  results for over-long queries instead of erroring.

### conventions worth knowing

- **No social-platform links in the output.** x.com / reddit.com / t.co links are
  stripped from the model's HTML (`strip_blocked_links`) and never fetched
  (`is_blocked_link`). The digest cites the paper/repo/article instead.
- **Fetched pages are untrusted text.** They enter the prompt fenced and labelled
  as data, not instruction.

## recommender/

Ranks the Readwise Reader firehose and tags the best few `shortlist`, which is
the tag behind Reader's own "⭐ Shortlist" view. It has its own venv and its own
cron slot on purpose -- see `recommender/run_shortlist.sh`.

```bash
cd recommender
uv venv .venv-rec
uv pip install --python .venv-rec/bin/python -r requirements.txt
./run_shortlist.sh --dry-run     # decides everything, writes nothing
```

### the cycle

`sync -> embed -> train -> select -> write`, nightly at 02:00 UTC.

| stage | what it does |
|---|---|
| sync | `list/?updatedAfter=` deltas into `data/recommender.sqlite3` |
| embed | bge-small ONNX, **on this box**, nothing sent to an API |
| train | logistic regression on frozen embeddings, retrained from scratch each run |
| select | ranks fresh feed items, plus resurfaced slots from the `later` backlog |
| write | one `bulk_update` adding/removing the `shortlist` tag |

### where the labels come from

Nobody has to sit down and rate a training set. The archive already is one, but
absence of a positive is deliberately *not* the same as a negative -- most
negative reasons only fire when there is contextual evidence he was actually
looking at things that day, so a quiet week doesn't get read as him disliking
everything in it:

| reason | signal | weight |
|---|---|---|
| `rated` | he tagged it `rate:good` / `rate:bad` -- overrides everything else | 3.0 |
| `favorited` | `favorite` / `important` tag | 2.0 |
| `read` | archived with real reading progress | 1.5 |
| `opened` | opened but not finished | 1.0 |
| `passed` | shortlisted, shown, evicted unopened -- only on a day he read more than half of that day's shortlist | 0.2 |
| `archived_unread` | archived without ever opening it -- an explicit "no" | 0.3 |
| `ignored` | stale, never-opened feed item -- only on a day he also archived something else unread | 0.1 |

Rating tags are the strongest signal and the one worth leaning on going
forward; the rest exist so the model has something to learn from before enough
ratings accumulate.

### two things that are easy to get wrong

- **Age must never be a feature.** Old documents are archived, archived means read,
  so age predicts the label almost perfectly and yields a model that ranks by "is
  old" while scoring beautifully. Recency is applied at selection time instead.
- **Ten slots: 7 from fresh feed, 3 resurfaced from the backlog, and one unranked
  random pick inside each of those two groups.** Without exploration the model only
  ever sees its own picks and the feedback loop eats itself. Each random arm is drawn
  from the *same* pool as the ranked picks it will be compared against -- `later`
  items differ from fresh feed by age and by having already survived a selection
  step, so a random draw from one pool says nothing about ranking in the other.

Known limits of the offline number, worth keeping in view: labels are derived from
the corpus *as it stands now*, so a document read yesterday counts as a positive even
in the training half of an older time split, and a recently arrived item can be
counted as ignored before it had a fair chance. It measures "would he open this",
not "was he glad he read it". The live ranked-vs-random comparison is the honest
metric; the offline one is a smoke test.

The job only ever removes the `shortlist` tag from documents it added itself
(every add and eviction is logged), so anything shortlisted by hand is left alone.

## status page

Every run of either job appends one JSON row to `data/run_stats/`. That part is
unconditional -- cheap, and useful history on its own. Turning it into a page is
**opt-in**: set `HTML_SERVE_DIR` in `.env` to a directory some webserver serves,
and both jobs render a self-contained HTML page there after every run. Leave it
unset and nothing under `html_status/` ever writes to disk.

```
HTML_SERVE_DIR=/home/html          # -> <that dir>/ai-digest/, <that dir>/recommender/
```

What each page shows:

- **`ai-digest/`** -- posts per source (log scale, and every bar opens to the
  posts that account or subreddit actually contributed), link-enrichment hit
  rate, and what the night cost across *both* vendors: OpenAI tokens and
  twitterapi.io credits
- **`recommender/`** -- held-out AUC run by run against its baselines, tonight's
  ten picks each with a "why this score" that splits the logit into
  topic/length/format and lists the labelled documents it resembles, the
  outgoing batch with his explicit verdict, ranked-vs-random measured live, and
  the signals that arrived overnight

`html_status/` sits at the repo root, shared by both `digest/` and
`recommender/` -- it is deliberately independent of either job's own
`config.py`. `stats_store.py` is the storage (append-only jsonl, trimmed to
`STATS_KEEP_RUNS`), `stats_page.py` is the renderer. `shortlist_stats.py`
(recommender-specific analysis: live arms, overnight signals, score
attribution) lives in `recommender/` instead, since it's tightly coupled to
`recommender_model.py`. Rebuild both pages from history alone, without running
either job:

```bash
python -m html_status.stats_page
```

(prints a one-line note and exits instead if `HTML_SERVE_DIR` is unset)

The score explanation is exact rather than indicative: the head is linear, so
`baseline + topic + length + format == logit` holds to floating point, and there
is a test that says so. It is deliberately *not* a bar per embedding dimension --
the encoder is frozen and its 384 dimensions have no individual meaning, so the
nearest-labelled-neighbour lists are the readable form of the topic term. The
control arm never gets an explanation: scoring the measurement arm after the fact
would turn it into another of the model's opinions.

Two rules the renderer holds to. A metric a run did not record renders as a dash,
never as zero -- "we never measured this" and "it was 0" are different claims.
And no JavaScript and no CDN: hover detail rides on SVG `<title>`, and every
chart is backed by a `<details>` table.

## tests

```bash
cd digest && .venv/bin/python -m pytest test_x_ingestion.py --timeout 5
cd recommender && .venv-rec/bin/python -m pytest test_shortlist.py test_stats.py --timeout 5
```

Covers the failure modes: API outage, resume-from-cache, dedupe, the query-length
cliff, link hygiene, and the cost ceiling. `test_stats.py` needs the recommender
venv; the digest-side status tests live in `digest/test_x_ingestion.py` because
`digest_stats` sits in `llm_summarizer.py`, which imports `openai`.
