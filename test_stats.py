#!/usr/bin/env python3
"""Tests for the run-stats layer and the two status pages (recommender side).

The point of this layer is that it is *never* the reason a job fails, and that
what it shows is true. So the failure modes covered here are: a half-written
history file, a row from an older version missing fields, a run that recorded
nothing at all, a document title carrying HTML, and -- the one that matters --
a missing measurement rendering as a dash instead of as a confident zero.

No network, no model load.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import config
import recommender_model
from html_status import shortlist_stats, stats_page, stats_store
from recommender_store import Store

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


def iso(days_ago):
    return (NOW - timedelta(days=days_ago)).isoformat()


def doc(doc_id, **kwargs):
    base = {
        "id": doc_id,
        "title": f"title {doc_id}",
        "summary": f"summary {doc_id}",
        "site_name": "example.com",
        "category": "rss",
        "location": "feed",
        "word_count": 1000,
        "saved_at": iso(1),
        "updated_at": iso(1),
        "reading_progress": 0.0,
        "tags": {},
    }
    base.update(kwargs)
    return base


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.sqlite3"))
    yield s
    s.close()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Redirect both the history and the served pages into the tmp dir."""
    monkeypatch.setattr(config, "STATS_DIR", str(tmp_path / "stats"))
    monkeypatch.setattr(config, "HTML_SERVE_DIR", str(tmp_path / "html"))
    return tmp_path


def embed(store, documents, seed=0):
    rng = np.random.default_rng(seed)
    vectors = {}
    for d in documents:
        v = rng.standard_normal(config.EMBED_DIM).astype(np.float32)
        vectors[d["id"]] = v / np.linalg.norm(v)
    store.upsert_documents(documents)
    store.save_embeddings(config.EMBED_KEY, vectors)


# -- the history file --------------------------------------------------------


def test_rows_come_back_in_the_order_they_were_written(paths):
    for i in range(3):
        stats_store.record("digest", {"n": i})
    assert [r["n"] for r in stats_store.load("digest")] == [0, 1, 2]
    assert stats_store.latest("digest")["n"] == 2


def test_history_is_trimmed_to_the_cap(paths):
    for i in range(10):
        stats_store.record("digest", {"n": i}, keep=4)
    assert [r["n"] for r in stats_store.load("digest")] == [6, 7, 8, 9]


def test_a_half_written_row_does_not_take_the_page_down(paths):
    """An OOM kill mid-append must not cost you the page that reports it."""
    stats_store.record("digest", {"n": 1})
    path = paths / "stats" / "digest.jsonl"
    path.write_text(path.read_text() + '{"n": 2, "trunc\n')
    stats_store.record("digest", {"n": 3})
    assert [r["n"] for r in stats_store.load("digest")] == [1, 3]


def test_no_history_is_not_an_error(paths):
    assert stats_store.load("digest") == []
    assert stats_store.latest("digest") is None


# -- rendering ---------------------------------------------------------------


def test_both_pages_render_with_no_runs_at_all(paths):
    written = stats_page.render_all()
    assert len(written) == 2
    for path in written:
        assert path.exists()
        assert "not written a stats row yet" in path.read_text()


def test_a_missing_measurement_renders_as_a_dash_not_a_zero(paths):
    """The whole point: "we never measured this" must not read as "it was 0"."""
    stats_store.record("digest", {"run_at": NOW.isoformat(), "reddit": {"total": 4}})
    markup = stats_page.render_digest(stats_store.load("digest"))
    assert "4" in markup
    assert stats_page.DASH in markup          # X count, cost, tokens: all absent
    assert "$0.0000" not in markup


def test_an_old_row_missing_every_new_field_still_renders(paths):
    """Rows outlive the code that wrote them; the renderer must not assume shape."""
    stats_store.record("shortlist", {"run_at": NOW.isoformat()})
    markup = stats_page.render_recommender(stats_store.load("shortlist"))
    assert "Shortlist recommender" in markup


def test_a_title_carrying_html_cannot_break_out_of_the_page(paths):
    stats_store.record("shortlist", {
        "run_at": NOW.isoformat(),
        "picks": [{"id": "x", "title": "<script>alert(1)</script>",
                   "url": "https://read.readwise.io/read/x", "slot": "feed",
                   "score": 0.5, "reason": "ranked #1"}],
    })
    markup = stats_page.render_recommender(stats_store.load("shortlist"))
    assert "<script>alert(1)</script>" not in markup
    assert "&lt;script&gt;" in markup


def test_the_chart_leaves_a_gap_for_a_missed_run_rather_than_joining_over_it(paths):
    svg = stats_page.line_chart(["a", "b", "c"], [("auc", [0.6, None, 0.7])])
    # two isolated points, so no path element bridging the hole
    assert svg.count("<path") == 0
    assert svg.count("<circle") >= 2


def test_a_chart_with_nothing_to_plot_says_so(paths):
    assert "not enough runs" in stats_page.line_chart([], [("auc", [])])
    assert "not enough runs" in stats_page.line_chart(["a"], [("auc", [None])])


def test_every_chart_ships_a_table_view(paths):
    assert "<table" in stats_page.line_chart(["a", "b"], [("auc", [0.6, 0.7])])
    assert "<table" in stats_page.bar_chart([("r/foo", 3)])


def test_a_failed_run_is_visible_on_the_page(paths):
    stats_store.record("digest", {"run_at": NOW.isoformat(), "ok": False,
                                  "problems": ["the model call failed"]})
    markup = stats_page.render_digest(stats_store.load("digest"))
    assert "needs a look" in markup
    assert "the model call failed" in markup


def test_pages_land_where_html_serve_expects_them(paths):
    stats_page.render_all()
    assert (paths / "html" / "ai-digest" / "index.html").exists()
    assert (paths / "html" / "recommender" / "index.html").exists()


# -- what the recommender page is asserting ----------------------------------


def test_outcome_reads_words_actually_consumed_not_percentage():
    assert shortlist_stats.outcome(doc("a", word_count=10_000,
                                       reading_progress=0.1,
                                       first_opened_at=iso(1))) == "read"
    assert shortlist_stats.outcome(doc("b", word_count=200,
                                       reading_progress=0.3,
                                       first_opened_at=iso(1))) == "opened"
    assert shortlist_stats.outcome(doc("c")) == "passed"


def test_an_arm_counts_a_document_once_however_often_it_was_shown(store):
    """Re-showing after the cooldown must not let one document vote twice."""
    store.upsert_documents([doc("a", first_opened_at=iso(1), reading_progress=1.0)])
    store.log_event("a", "added", "2026-01-01", slot="feed", score=0.9)
    store.log_event("a", "evicted", "2026-01-05")
    store.log_event("a", "added", "2026-09-01", slot="feed", score=0.9)
    arms = shortlist_stats.live_arms(store)
    assert arms["feed"]["shown"] == 1
    assert arms["feed"]["read"] == 1


def test_the_random_arm_is_kept_separate_from_the_ranked_one(store):
    store.upsert_documents([
        doc("ranked", first_opened_at=iso(1), reading_progress=1.0),
        doc("control"),
    ])
    store.log_event("ranked", "added", "2026-09-01", slot="feed")
    store.log_event("control", "added", "2026-09-01", slot="feed-random")
    arms = shortlist_stats.live_arms(store)
    assert arms["feed"]["open_rate"] == 1.0
    assert arms["feed-random"]["open_rate"] == 0.0


def test_overnight_signals_only_count_what_moved_since_the_last_sync(store):
    store.upsert_documents([
        doc("fresh", updated_at=iso(0.2), saved_at=iso(0.2)),
        doc("rated", updated_at=iso(0.2), saved_at=iso(10), tags={"rate:good": {}}),
        doc("old", updated_at=iso(30), saved_at=iso(30)),
    ])
    signals = shortlist_stats.overnight_signals(store.documents(), iso(1))
    assert signals["updated"] == 2
    assert signals["new_docs"] == 1
    assert signals["rated_good"] == 1


def test_overnight_signals_survive_a_first_run_with_no_previous_sync(store):
    store.upsert_documents([doc("a")])
    assert shortlist_stats.overnight_signals(store.documents(), None)["updated"] == 0


def test_a_random_pick_explains_itself_as_the_control_arm():
    reason = shortlist_stats.describe(doc("a"), "feed-random", None,
                                      {"pool_size": 412})
    assert "random" in reason and "412" in reason
    assert "ranked #" not in reason


def test_a_ranked_pick_reports_where_it_placed():
    reason = shortlist_stats.describe(doc("a"), "feed", 0.87,
                                      {"rank": 3, "pool_size": 412})
    assert "#3 of 412" in reason and "0.87" in reason


def test_a_short_read_pick_says_it_took_a_reserved_slot():
    reason = shortlist_stats.describe(doc("a", word_count=400), "feed", 0.4,
                                      {"rank": 40, "pool_size": 412,
                                       "short_reserve": True})
    assert "short-read" in reason


def test_the_nearest_read_is_the_one_it_actually_points_at(store):
    target = np.zeros(config.EMBED_DIM, dtype=np.float32)
    target[0] = 1.0
    other = np.zeros(config.EMBED_DIM, dtype=np.float32)
    other[1] = 1.0
    embeddings = {"pick": target, "match": target, "miss": other}
    found = shortlist_stats.nearest_read(
        "pick", embeddings, (["miss", "match"], np.vstack([other, target])))
    assert found["id"] == "match"
    assert found["similarity"] == pytest.approx(1.0)


def test_nearest_read_is_absent_rather_than_wrong_when_there_is_nothing_to_match():
    assert shortlist_stats.nearest_read("pick", {}, None) is None


def test_pick_traces_the_reason_for_every_document_it_chose(store):
    import shortlist_job
    documents = [doc(f"f{i}") for i in range(12)]
    embed(store, documents)
    taste = np.ones(config.EMBED_DIM, dtype=np.float32) / np.sqrt(config.EMBED_DIM)
    trace = {}
    chosen = shortlist_job.pick(store, None, taste, now=NOW, trace=trace)
    assert chosen
    for document, slot, score in chosen:
        assert document["id"] in trace
        detail = trace[document["id"]]
        assert detail["pool_size"] == len(documents)
        # a random pick has no rank; that absence is what makes it the control
        assert (detail["rank"] is None) == slot.endswith("-random")
    assert trace["_pools"]["feed_candidates"] == len(documents)


def test_a_skipped_sync_reports_unmeasured_rather_than_zero(paths):
    """Zero overnight signals and "we never asked" are different claims."""
    stats_store.record("shortlist", {"run_at": NOW.isoformat(), "overnight": None,
                                     "sync": {"skipped": True}})
    markup = stats_page.render_recommender(stats_store.load("shortlist"))
    assert "Not measured this run" in markup
    assert "documents Readwise touched" not in markup


# -- his explicit verdict ----------------------------------------------------


def test_a_document_he_rated_bad_is_not_banked_as_a_win(store):
    """He read it to the end and then said it was bad. The arm that picked it
    does not get to count that as a success it can hide inside "read"."""
    store.upsert_documents([doc("a", first_opened_at=iso(1), reading_progress=1.0,
                                tags={"rate:bad": {}})])
    store.log_event("a", "added", "2026-09-01", slot="feed")
    arms = shortlist_stats.live_arms(store)
    assert arms["feed"]["read"] == 1        # behaviourally he did read it
    assert arms["feed"]["rated_bad"] == 1   # and it is visibly a rejection
    assert arms["feed"]["rated_good"] == 0


def test_rating_is_kept_apart_from_reading():
    assert shortlist_stats.rating(doc("a", tags={"rate:good": {}})) == "good"
    assert shortlist_stats.rating(doc("a", tags={"rate:bad": {}})) == "bad"
    assert shortlist_stats.rating(doc("a")) is None


# -- score attribution -------------------------------------------------------


class FakeModel:
    """A linear head with a known shape, so the split can be checked exactly."""

    def __init__(self, dim):
        n = dim + 1 + len(recommender_model.CATEGORIES)
        self.coef_ = np.zeros((1, n), dtype=np.float64)
        self.coef_[0, :dim] = 0.1          # topic
        self.coef_[0, dim] = 0.5           # length
        self.coef_[0, dim + 1:] = 0.25     # format
        self.intercept_ = np.array([0.3])


def test_the_contributions_add_up_to_the_score_exactly(store):
    """If the bars do not sum to the logit they are decoration, not explanation."""
    documents = [doc(f"f{i}") for i in range(5)]
    embed(store, documents)
    embeddings = store.embeddings(config.EMBED_KEY)
    baseline = shortlist_stats.baseline_features(store.documents(), embeddings)
    model = FakeModel(config.EMBED_DIM)
    attr = shortlist_stats.attribution(documents[0], embeddings[documents[0]["id"]],
                                       model, baseline)
    total = attr["baseline_logit"] + sum(attr["contributions"].values())
    assert total == pytest.approx(attr["logit"], abs=1e-9)
    assert attr["probability"] == pytest.approx(1 / (1 + np.exp(-attr["logit"])), abs=1e-9)
    assert set(attr["contributions"]) == set(shortlist_stats.GROUPS)


def test_the_average_document_gets_no_contribution_at_all(store):
    """The split is taken about the mean, so the mean must land on the baseline."""
    documents = [doc(f"f{i}") for i in range(6)]
    embed(store, documents)
    embeddings = store.embeddings(config.EMBED_KEY)
    baseline = shortlist_stats.baseline_features(store.documents(), embeddings)
    mean_doc = doc("mean", word_count=documents[0]["word_count"])
    mean_vector = np.mean(np.vstack(list(embeddings.values())), axis=0)
    attr = shortlist_stats.attribution(mean_doc, mean_vector, FakeModel(config.EMBED_DIM),
                                       baseline)
    for value in attr["contributions"].values():
        assert value == pytest.approx(0.0, abs=1e-5)


def test_attribution_is_absent_rather_than_guessed_without_a_model(store):
    documents = [doc("a")]
    embed(store, documents)
    embeddings = store.embeddings(config.EMBED_KEY)
    baseline = shortlist_stats.baseline_features(store.documents(), embeddings)
    assert shortlist_stats.attribution(documents[0], embeddings["a"], None, baseline) is None
    assert shortlist_stats.attribution(documents[0], None, FakeModel(config.EMBED_DIM),
                                       baseline) is None


def test_the_evidence_lists_separate_what_he_engaged_with_from_what_he_ignored(store):
    labels = {"good": (1, 1.0, "read"), "bad": (0, 0.3, "ignored")}
    target = np.zeros(config.EMBED_DIM, dtype=np.float32); target[0] = 1.0
    other = np.zeros(config.EMBED_DIM, dtype=np.float32); other[1] = 1.0
    embeddings = {"pick": target, "good": target, "bad": other}
    labelled = shortlist_stats.labelled_matrix(labels, embeddings)
    found = shortlist_stats.evidence("pick", embeddings, labelled, k=2)
    assert [h["id"] for h in found["like"]] == ["good"]
    assert [h["id"] for h in found["unlike"]] == ["bad"]
    assert found["like"][0]["reason"] == "read"


def test_the_control_arm_is_never_given_an_attribution(paths):
    """Explaining a random pick would turn the measurement arm into an opinion."""
    stats_store.record("shortlist", {
        "run_at": NOW.isoformat(),
        "picks": [{"id": "r", "title": "control", "slot": "feed-random",
                   "score": None, "reason": "random draw", "attribution": None}],
    })
    markup = stats_page.render_recommender(stats_store.load("shortlist"))
    assert "why this score" not in markup


def test_a_skipped_sync_warns_on_the_outcome_tables_too(paths):
    """The states in those tables are exactly what goes stale without a sync."""
    stats_store.record("shortlist", {"run_at": NOW.isoformat(),
                                     "sync": {"skipped": True}})
    markup = stats_page.render_recommender(stats_store.load("shortlist"))
    assert "frozen at the last real sync" in markup


# -- the digest drill-down and log scale -------------------------------------


def test_log_bars_still_print_the_true_counts():
    svg = stats_page.bar_chart([("a", 1), ("b", 400)], log=True)
    assert "log10" in svg
    assert ">400<" in svg and ">1<" in svg


def test_a_zero_count_survives_the_log_scale():
    """log(0) is where a naive implementation divides by zero or drops the row."""
    svg = stats_page.bar_chart([("dead", 0), ("alive", 10)], log=True)
    assert "dead" in svg


def test_drill_down_rows_carry_the_posts(paths):
    bodies = {"r/foo": "<p>the actual post</p>"}
    markup = stats_page.bar_chart([("r/foo", 1)], bodies=bodies)
    assert "the actual post" in markup
    assert "<details>" in markup


def test_a_post_never_links_back_to_the_platform():
    """His devices block those domains; a link there is dead and a temptation."""
    bodies = stats_page.tweet_bodies([{
        "handle": "someone", "text": "look",
        "external_links": ["https://x.com/someone/status/1",
                           "https://arxiv.org/abs/2401.00001"],
    }])
    assert "arxiv.org" in bodies["someone"]
    assert "x.com" not in bodies["someone"]


def test_reddit_drill_down_shows_score_and_top_comment():
    bodies = stats_page.reddit_bodies([{
        "subreddit": "singularity", "title": "a title", "score": 42,
        "comments": [{"body": "meh", "score": 1}, {"body": "the good one", "score": 99}],
    }])
    assert "a title" in bodies["singularity"]
    assert "42" in bodies["singularity"]
    assert "the good one" in bodies["singularity"]


def test_a_missing_source_dump_is_reported_not_faked(paths):
    stats_store.record("digest", {"run_at": NOW.isoformat(),
                                  "reddit": {"total": 5, "file": "extracts/gone.json"}})
    markup = stats_page.render_digest(stats_store.load("digest"))
    assert "no longer on disk" in markup


def test_twitter_spend_shows_credits_and_flags_the_unverified_rate(paths):
    stats_store.record("digest", {
        "run_at": NOW.isoformat(),
        "x": {"total": 10, "spend": {"credits_used": 150_000, "credits_remaining": 900_000,
                                     "credits_per_usd": 100_000, "usd": 1.5}},
        "llm": {"cost_usd": 0.25},
    })
    markup = stats_page.render_digest(stats_store.load("digest"))
    assert "150,000" in markup
    assert "$1.5000" in markup
    assert "unverified" in markup
    assert "$1.7500" in markup        # the total, both vendors


def test_no_credit_reading_shows_nothing_rather_than_free(paths):
    """A missing balance must not render as "the X leg cost $0"."""
    stats_store.record("digest", {"run_at": NOW.isoformat(),
                                  "x": {"total": 10, "spend": {"credits_used": None,
                                                               "usd": None}},
                                  "llm": {"cost_usd": 0.25}})
    markup = stats_page.render_digest(stats_store.load("digest"))
    assert "not recorded for this run" in markup
    assert "$0.0000" not in markup


def test_a_bare_platform_url_is_stripped_out_of_quoted_post_text(paths):
    """Unclickable here, still a dead address on his devices and still a nudge."""
    bodies = stats_page.tweet_bodies([{
        "handle": "someone",
        "text": "look at this https://t.co/abc123 and this https://arxiv.org/abs/1",
    }])
    assert "t.co" not in bodies["someone"]
    assert "arxiv.org/abs/1" in bodies["someone"]
    assert "look at this" in bodies["someone"]
