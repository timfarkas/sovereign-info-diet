# Sovereign Information Diet 

You don't want Elon to decide what to feed or not feed you? 
You don't want Zuck to use you as an attention extraction mine?

Then LIBERATE YOURSELF from their dystopian yoke and build your own data ingestion pipeline and recommender algorithm.

This project consists of two things that allow this:
- A content digest system that scrapes Reddit posts, X posts and the RSS feeds and newsletters you subscribe to, and summarizes them per topic, sending them to you as e-mails. Four topics ship: AI/tech nightly, plus geopolitics/markets/supply chains, pandemic preparedness and bio-risk, and Europe/the EU/European liberal values, each every three days.
- A Readwise recommender system that shows you posts similar to ones you read or saved for later or rated well.


## Full Flow

1. `X posts, Reddit posts and subscribed RSS feeds -> digest system -> one content digest e-mail per topic`

2. `Digest e-mails & other newsletters -> auto-forwarded to Readwise feed e-mail (via mail server) -> Readwise Feed`

3. `Readwise Feed -> recommender system picks ten items from feed and 'saved for later' -> Daily Recommendations in Readwise`

The recommender system is re-trained every night on your ratings and reading behavior. Specifically, it is trained on 1. your ratings, 2. your reading and `save for later` behavior.


## Pre-requisites
The entire pipeline requires a VPS or private server to run on every night. Furthermore, you'll need:


**Digest system:**
- Mail server
- Twitter API (inofficial) & Reddit API keys
- OpenAI API keys for summaries


**Recommender system:**
- E-mail server/client capable of forwarding newsletters and digests (e.g. ProtonMail)
- Readwise & Readwise API Key


## Technical Details (LLM-generated)
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
A full digest run mails `EMAIL_TO` plus everyone in `EMAIL_TO_EXTRA`, a
comma-separated list (also `.env` -- not `config.py`, so no address is
committed to this public repo) -- see "the pipeline" below for how `--test`
and `--dry-run` narrow that.

### digest/

```bash
cd digest
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

#### the pipeline

`digest/run_pipeline.sh` (cron, 01:00 UTC) runs four legs:

| leg | what it does |
|---|---|
| `x_scraper.py` | pulls X timelines via twitterapi.io for the union of who `X_SEED_ACCOUNTS` follow |
| `reddit_scraper.py` | pulls the union of every topic's subreddits via praw |
| `rss_scraper.py` | pulls subscribed feeds, via Readwise Reader and by fetching feed URLs directly |
| `digest_run.py` | for each topic that is due: fetches linked pages (`link_fetcher.py`), summarizes, mails |

Each source leg is scraped **once** and sliced per topic, rather than scraped per
topic -- X credits are per-tweet, so four scrapes would pay four times for
largely the same corpus. Any source leg may fail without killing the run, each
topic is isolated in its own try/except, `digest_run.py` refuses to mail an empty
digest, and it stamps a maintenance banner on the mail when a leg is unhealthy --
so breakage shows up in the inbox rather than in silence.

#### topics

A digest is a row in `digest/topics.py`, not a code path. Everything downstream
takes a `Topic` and does not know which one it got, which is what makes adding a
fifth topic a config change:

| field | what it decides |
|---|---|
| `every_n_days` | cadence. Compared against the last **success**, so a failed run retries tomorrow instead of being benched for another cycle |
| `window_days` / `lookback_hours` | how far back the prompt claims to look, and how old a dump on disk may be |
| `subreddits` / `posts_per_subreddit` | its slice of the reddit corpus |
| `x_all` / `keywords` | the whole X corpus (the AI topic) or the tweets matching its keywords |
| `feeds` / `feed_urls` | publications routed to it wholesale, and feeds to fetch directly |
| `cost_ceiling_usd` | warns past this; measured per run |

Routing is a word-bounded keyword regex plus publication matching -- deliberately
the low-bit version. Over ~250 items a night it is auditable, free and instant,
and since the prompt asks the model to discard >80% of what it is handed, a false
positive costs a few hundred tokens where a false negative costs a missed item.
So the keyword lists lean inclusive.

`feed_urls` says which feeds to *fetch*, not which items land in that topic: a
feed added there still needs its publication in `feeds`, or a keyword hit, to
reach anything. That is what lets `who.int` be fetched for the pandemic topic but
routed by keyword, so WHO news about staffing does not land in a bio-risk digest.

```bash
python digest_run.py --list                  # what is due tonight, and why
python digest_run.py --only geopolitics --test  # one topic, mailed to EMAIL_TO only
python digest_run.py --no-mail --force       # all of them, ignoring cadence, no mail
python llm_summarizer.py                     # still means "produce the AI digest"
```

Three mail tiers, checked in this order: `--dry-run`/`--no-mail` sends nothing;
`--test` sends to `EMAIL_TO` only; otherwise (the nightly cron path) it sends
to `EMAIL_TO` plus every address in `EMAIL_TO_EXTRA` (.env).

#### subscribed feeds

Two independent paths, not a primary and a fallback:

- **Readwise Reader** (`RSS_READWISE_ENABLED`) -- what he actually subscribes to,
  already parsed, via `/api/v3/list/` with `location=feed`. `RSS_CATEGORIES`
  defaults to `("rss", "email")`, i.e. newsletters count as subscribed feeds.
- **Direct fetch** (`RSS_DIRECT_ENABLED`) -- `feedparser` over every topic's
  `feed_urls`, so the digest still works on a box with no Readwise account.

Results are merged and deduplicated on the **cleaned** URL (query and fragment
stripped, which is where the campaign tracking lives) plus the title; the
Readwise copy wins a tie. Body text comes from `link_fetcher.py`, because
Readwise's list endpoint returns an empty `content` field -- measured 0/126, with
`summary` populated on 125/126.

His own digests are auto-forwarded into that feed (step 2 of the Full Flow), so
the pipeline would otherwise summarize itself, compounding each pass. They are
excluded at **ingest** (`RSS_EXCLUDE_AUTHORS`, `RSS_EXCLUDE_TITLE_PREFIXES`), so
no later routing change can surface one.

A feed that 404s costs its own items and nothing else. Note that HTTP 200 is not
evidence a feed is alive: `csis.org/rss.xml` returns 200 with entries from 2016,
and `cidrap.umn.edu/rss.xml` from 2022. Check the newest entry's date, not the
status code, before adding one.

#### knobs

All in `digest/config.py`. The ones that move cost or shape:

- `SUMMARY_MODEL` / `MODEL_PRICING` / `SUMMARY_COST_CEILING_USD` — every run prints
  its measured cost and warns past the ceiling.
- `TIME_HORIZON_DAYS` — the AI digest's window. Widening it is safe: the seen-ids
  store stops already-summarized posts from reappearing. Other topics carry their
  own `window_days`.
- `POSTS_PER_SUBREDDIT` — a fixed quota, not `POSTS_TO_ANALYZE // len(SUBREDDITS)`:
  that expression was computed over the AI topic's five subreddits, and the union
  across all topics is 19, so keeping it would have silently cut the AI digest
  from 12 posts per subreddit to 3.
- `RSS_WINDOW_DAYS`, `RSS_MAX_ITEMS_PER_FEED`, `RSS_SUMMARY_MAX_CHARS` — the feed
  leg's size levers. A topic's own `max_feed_items` caps what reaches its prompt,
  and that cap is spent **round-robin across publications**: a daily paper
  publishes ~35 items per window and a weekly law blog ~4, so newest-first
  selection would hand the whole budget to the loudest feed.
- `X_MAX_TWEETS_IN_PROMPT`, `LINK_FETCH_MAX_PAGES`, `LINK_FETCH_MAX_CHARS` — the
  three levers on prompt size, i.e. on cost.
- `X_MAX_QUERY_CHARS` — leave it under ~450. X search silently returns zero
  results for over-long queries instead of erroring.

#### conventions worth knowing

- **No social-platform links in the output.** x.com / reddit.com / t.co links are
  stripped from the model's HTML (`strip_blocked_links`) and never fetched
  (`is_blocked_link`). The digest cites the paper/repo/article instead.
- **Fetched pages are untrusted text.** They enter the prompt fenced and labelled
  as data, not instruction.

### recommender/

Ranks the Readwise Reader firehose and tags the best few `shortlist`, which is
the tag behind Reader's own "⭐ Shortlist" view. It has its own venv and its own
cron slot on purpose -- see `recommender/run_shortlist.sh`.

```bash
cd recommender
uv venv .venv-rec
uv pip install --python .venv-rec/bin/python -r requirements.txt
./run_shortlist.sh --dry-run     # decides everything, writes nothing
```

#### the cycle

`sync -> embed -> train -> select -> write`, nightly at 02:00 UTC.

| stage | what it does |
|---|---|
| sync | `list/?updatedAfter=` deltas into `data/recommender.sqlite3` |
| embed | bge-small ONNX, **on this box**, nothing sent to an API |
| train | logistic regression on frozen embeddings, retrained from scratch each run |
| select | ranks fresh feed items, plus resurfaced slots from the `later` backlog |
| write | one `bulk_update` adding/removing the `shortlist` tag |

#### where the labels come from

Nobody has to sit down and rate a training set -- the archive already is one.
Absence of a positive counts as a weak negative, which is crude but is where
the bulk of the signal lives: the thousands of stale, never-opened feed items
are what give the ranker anything to push *against*. See the note below on the
2026-09-25 attempt to make this more principled, and why it was reverted.

| reason | signal | weight |
|---|---|---|
| `rated` | he tagged it `rate:good` / `rate:bad` -- overrides everything else | 3.0 |
| `favorited` | `favorite` / `important` tag | 2.0 |
| `read` | archived with real reading progress | 1.5 |
| `opened` | opened but not finished | 1.0 |
| `passed` | shortlisted, shown, evicted unopened | 1.0 |
| `ignored` | stale, never-opened feed item | 0.3 |

Rating tags are the strongest signal and the one worth leaning on going
forward; the rest exist so the model has something to learn from before enough
ratings accumulate.

**Reverted 2026-09-29 -- do not re-apply without an online A/B.** Between
2026-09-25 and 2026-09-29 the two weak negatives only fired with same-day
corroborating evidence (`passed` needed him to have read most of that day's
shortlist, `ignored` needed something else archived unread the same day), a new
`archived_unread` reason was added, and the weights dropped to 0.2 / 0.1. The
reasoning was sound -- not opening something on a busy day is not a taste
signal. The effect was not: labels fell 6045 -> 1304, the training set went
from 14% positive to 65% positive, and held-out AUC fell from 0.72 to 0.53
against a random baseline of 0.47. The ranker was at chance. Whatever is wrong
with counting neglect as rejection, it is less wrong than having no negatives.

#### two things that are easy to get wrong

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

### status page

Every run of either job appends one JSON row to `data/run_stats/`. That part is
unconditional -- cheap, and useful history on its own. Turning it into a page is
**opt-in**: set `HTML_SERVE_DIR` in `.env` to a directory some webserver serves,
and both jobs render a self-contained HTML page there after every run. Leave it
unset and nothing under `html_status/` ever writes to disk.

```
HTML_SERVE_DIR=/home/html          # -> <that dir>/ai-digest/, <that dir>/recommender/
```

What each page shows:

- **`ai-digest/`**, **`digest-geopolitics/`**, **`digest-pandemic/`**,
  **`digest-europe/`** -- one page per topic: posts per source (log scale, and
  every bar opens to the posts that account or subreddit actually contributed),
  link-enrichment hit rate, feed ingest health (both paths, what was dropped at
  ingest and why, volume per publication, any feed that failed), and what the run
  cost across *both* vendors: OpenAI tokens and twitterapi.io credits. Topic
  pages are discovered from the history directory, so a new topic gets a page the
  first time it records a run; the AI page exists unconditionally, since on a box
  with no history "no run recorded yet" is that page's job and a missing page
  looks like a webserver fault
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

### tests

```bash
cd digest && .venv/bin/python -m pytest test_x_ingestion.py test_topics.py --timeout 5
cd recommender && .venv-rec/bin/python -m pytest test_shortlist.py test_stats.py --timeout 5
```

Covers the failure modes: API outage, resume-from-cache, dedupe, the query-length
cliff, link hygiene, and the cost ceiling. `test_topics.py` additionally pins
every value the AI digest previously hardcoded -- including that it draws no feed
items at all -- and asserts the cross-cutting prompt rules (no social links, the
untrusted-data fence, fragment-only HTML) as properties of each topic's prompt
rather than by comparing wording, since the AI prompt is deliberately its own
text. `test_stats.py` needs the recommender
venv; the digest-side status tests live in `digest/test_x_ingestion.py` because
`digest_stats` sits in `llm_summarizer.py`, which imports `openai`.
