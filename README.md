# digital info filter

X/Twitter + Reddit → linked-page enrichment → llm gate → daily digest email.

## setup
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

`.env` needs: `REDDIT_CLIENT_ID`, `REDDIT_SECRET`, `TWITTER_IO_API_KEY`,
`OPENAI_API_KEY`, `EMAIL_FROM`, `EMAIL_TO`.

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

## tests

```bash
.venv/bin/python -m pytest test_x_ingestion.py --timeout 5
```

Covers the failure modes: API outage, resume-from-cache, dedupe, the query-length
cliff, link hygiene, and the cost ceiling.
