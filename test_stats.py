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
import shortlist_stats
import stats_page
import stats_store
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
    signals = shortlist_stats.overnight_signals(store, iso(1))
    assert signals["updated"] == 2
    assert signals["new_docs"] == 1
    assert signals["rated_good"] == 1


def test_overnight_signals_survive_a_first_run_with_no_previous_sync(store):
    store.upsert_documents([doc("a")])
    assert shortlist_stats.overnight_signals(store, None)["updated"] == 0


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
