"""Status pages for the digest and recommender jobs.

`stats_store` records one JSON row per run. `stats_page` renders those rows
into two static HTML pages. `shortlist_stats` is the recommender-specific
analysis (live arms, overnight signals, score attribution) that feeds the
recommender page's row.

Rendering is opt-in: it only writes anything when `config.HTML_SERVE_DIR` is
set (from the `HTML_SERVE_DIR` env var). Recording to `stats_store` is not
gated -- it is cheap, and useful history even for someone who never turns the
pages on. See the "status pages" section of the top-level README.
"""
