# digital info filter

X/Twitter + Reddit → linked-page enrichment → llm gate → daily digest email.

## setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`.env` needs: `REDDIT_CLIENT_ID`, `REDDIT_SECRET`, `TWITTER_IO_API_KEY`,
`OPENAI_API_KEY`, `EMAIL_FROM`, `EMAIL_TO`. `HTML_SERVE_DIR` is optional -- see
"status pages" below.

## the pipeline

`run_pipeline.sh` (cron, 01:00 UTC) runs four legs:

| leg | what it does |
|---|---|
| `x_scraper.py` | pulls X timelines via twitterapi.io for the union of who `X_SEED_ACCOUNTS` follow |
| `reddit_scraper.py` | pulls `SUBREDDITS` via praw |
| `llm_summarizer.py` | fetches linked pages (`link_fetcher.py`), then summarizes all three sources |
| `send_notification.py` | mails the latest summary |

Either source leg may fail without killing the run; the summarizer refuses to mail
an empty digest and stamps a maintenance banner on the mail when a leg is
unhealthy, so breakage shows up in the inbox rather than in silence.

## knobs

All in `config.py`. The ones that move cost or shape:

- `SUMMARY_MODEL` / `MODEL_PRICING` / `SUMMARY_COST_CEILING_USD` — every run prints
  its measured cost and warns past the ceiling.
- `TIME_HORIZON_DAYS` — digest window. Widening it is safe: the seen-ids store
  stops already-summarized posts from reappearing.
- `X_MAX_TWEETS_IN_PROMPT`, `LINK_FETCH_MAX_PAGES`, `LINK_FETCH_MAX_CHARS` — the
  three levers on prompt size, i.e. on cost.
- `X_MAX_QUERY_CHARS` — leave it under ~450. X search silently returns zero
  results for over-long queries instead of erroring.

## conventions worth knowing

- **No social-platform links in the output.** x.com / reddit.com / t.co links are
  stripped from the model's HTML (`strip_blocked_links`) and never fetched
  (`is_blocked_link`). The digest cites the paper/repo/article instead.
- **Fetched pages are untrusted text.** They enter the prompt fenced and labelled
  as data, not instruction.

## the readwise shortlist recommender

A second, independent job: it ranks the Readwise Reader firehose and tags the best
few `shortlist`, which is the tag behind Reader's own "⭐ Shortlist" view. It has
its own venv and its own cron slot on purpose -- see `run_shortlist.sh`.

```bash
uv venv .venv-rec
uv pip install --python .venv-rec/bin/python fastembed scikit-learn requests python-dotenv pytest pytest-timeout
./run_shortlist.sh --dry-run     # decides everything, writes nothing
```

`.env` needs `READWISE_API_KEY` (readwise.io/access_token).

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

Nobody has to sit down and rate a training set. The archive already is one:
documents that got archived with real reading progress are positives, and stale
feed items that were never opened are (noisy, down-weighted) negatives. Rating
tags -- `rate:good` / `rate:bad` -- are the strongest signal and override
behaviour, but they are a refinement, not a prerequisite.

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

## status pages

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

`html_status/` is its own package: `stats_store.py` is the storage (append-only
jsonl, trimmed to `STATS_KEEP_RUNS`), `stats_page.py` is the renderer, and
`shortlist_stats.py` is the recommender-specific analysis (live arms, overnight
signals, score attribution) that feeds the recommender page's row. Rebuild both
pages from history alone, without running either job:

```bash
.venv-rec/bin/python -m html_status.stats_page
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
.venv/bin/python -m pytest test_x_ingestion.py test_reddit_scraper.py --timeout 5
.venv-rec/bin/python -m pytest test_shortlist.py test_stats.py --timeout 5
```

`test_stats.py` needs the recommender venv; the digest-side status tests live in
`test_x_ingestion.py` because `digest_stats` sits in `llm_summarizer`, which
imports `openai`.

Covers the failure modes: API outage, resume-from-cache, dedupe, the query-length
cliff, link hygiene, and the cost ceiling.
