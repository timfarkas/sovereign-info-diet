#!/usr/bin/env python3
"""Tests for the offline control-vs-candidate comparison and what feeds its pages.

The contract that matters: every ranker is judged on the same documents, and
none of those documents was in any ranker's training data.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import config
import recommender_model
import recommender_refine
import shortlist_stats
from html_status import stats_page, stats_store
from html_status import config as html_status_config
from recommender_store import Store

NOW = datetime(2026, 9, 25, tzinfo=timezone.utc)


def iso(days_ago):
    return (NOW - timedelta(days=days_ago)).isoformat()


def doc(doc_id, **kwargs):
    base = {"id": doc_id, "title": f"title {doc_id}", "summary": f"summary {doc_id}",
            "site_name": "example.com", "category": "rss", "location": "feed",
            "word_count": 800, "saved_at": iso(30), "reading_progress": 0.0, "tags": {}}
    base.update(kwargs)
    return base


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.sqlite3"))
    yield s
    s.close()


def stock(store, documents, seed=0):
    rng = np.random.default_rng(seed)
    vectors = {}
    for d in documents:
        v = rng.standard_normal(config.EMBED_DIM).astype(np.float32)
        vectors[d["id"]] = v / np.linalg.norm(v)
    store.upsert_documents(documents)
    store.save_embeddings(config.EMBED_KEY, vectors)


def corpus(monkeypatch, blocks=10, per_block=10):
    """`blocks` arrival blocks; even ones scanned (most opened), odd ones not."""
    monkeypatch.setattr(config, "EXPOSURE_BLOCK", per_block)
    docs = []
    for b in range(blocks):
        for i in range(per_block):
            kwargs = {"saved_at": iso(200 - b * 10 - i * 0.1)}
            if b % 2 == 0:
                kwargs["first_opened_at"] = kwargs["saved_at"]
                if i < 3:
                    kwargs.update(reading_progress=1.0, location="archive")
            docs.append(doc(f"b{b}d{i}", **kwargs))
    return docs


def test_the_evaluation_sets_never_reach_training(store, monkeypatch):
    docs = corpus(monkeypatch)
    stock(store, docs)
    store.log_event("b1d0", "added", "2026-06-01")
    store.log_event("b1d0", "evicted", "2026-06-02")
    sets, excluded = recommender_refine.eval_sets(store)
    assert set(sets["shown"]) == {"b1d0"}
    assert sets["triage"] and set(sets["triage"]) <= excluded
    for spec in config.RANKERS.values():
        _, stats = recommender_model.train(store, now=NOW, exclude=excluded, **spec)
        labels = recommender_model.derive_labels(store, now=NOW, scheme=spec["scheme"])
        assert stats["usable"] == len(set(labels) - excluded)


def test_only_blocks_he_was_actually_scanning_are_triage(store, monkeypatch):
    docs = corpus(monkeypatch)
    stock(store, docs)
    sets, _ = recommender_refine.eval_sets(store)
    assert all(int(i[1:].split("d")[0]) % 2 == 0 for i in sets["triage"])


def test_a_shortlist_pick_with_no_outcome_yet_is_not_judged(store, monkeypatch):
    stock(store, corpus(monkeypatch))
    store.log_event("b1d1", "added", "2026-09-24")
    sets, _ = recommender_refine.eval_sets(store)
    assert "b1d1" not in sets["shown"]


def test_neutral_label_lets_a_bad_rating_override_reading():
    assert recommender_refine.neutral_label(
        doc("x", reading_progress=1.0, tags={config.RATE_BAD_TAG: {}})) == 0
    assert recommender_refine.neutral_label(doc("x", reading_progress=0.5)) == 1
    assert recommender_refine.neutral_label(doc("x", reading_progress=0.05)) == 0


def test_an_interval_needs_both_classes():
    assert recommender_refine.auc_with_interval([1, 1, 1], [0.1, 0.2, 0.3]) is None
    a = recommender_refine.auc_with_interval([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert a["auc"] == 1.0 and a["n"] == 4 and a["positives"] == 2


def test_compare_scores_every_ranker_on_the_same_sets(store, monkeypatch):
    docs = corpus(monkeypatch, blocks=12)
    docs += [doc(f"fresh{i}", saved_at=iso(1)) for i in range(5)]
    stock(store, docs)
    result = recommender_refine.compare(store, now=NOW)
    sizes = {name: r["triage"]["n"] for name, r in result["rankers"].items() if r["triage"]}
    assert set(sizes) == set(config.RANKERS)
    assert len(set(sizes.values())) == 1
    assert result["word_count_baseline"]["triage"] is None or \
        result["word_count_baseline"]["triage"]["n"] == next(iter(sizes.values()))
    assert all(len(top) == 5 for top in result["top_fresh"].values())


# -- what the pages show -----------------------------------------------------


def test_publication_gets_its_own_exact_bar(store, monkeypatch):
    """The bars must still sum to the logit with the publication columns on."""
    monkeypatch.setattr(config, "PUBLICATION_MIN_DOCS", 3)
    docs = ([doc(f"g{i}", site_name="good.example", location="archive", reading_progress=1.0)
             for i in range(25)]
            + [doc(f"b{i}", site_name="bad.example", tags={config.RATE_BAD_TAG: {}})
               for i in range(25)])
    stock(store, docs)
    model, _ = recommender_model.train(store, now=NOW, scheme="exposure", publications=True)
    embeddings = store.embeddings(config.EMBED_KEY)
    baseline = shortlist_stats.baseline_features(
        store.documents(), embeddings, recommender_model.publications_of(model))
    attr = shortlist_stats.attribution(docs[0], embeddings["g0"], model, baseline)
    assert set(attr["contributions"]) == set(shortlist_stats.GROUPS) | {"publication"}
    assert attr["contributions"]["publication"] > 0
    assert attr["publication"] == "good.example"
    total = attr["baseline_logit"] + sum(attr["contributions"].values())
    assert total == pytest.approx(attr["logit"], abs=1e-6)


def test_the_spotcheck_shows_exactly_what_was_embedded(store):
    import recommender_embed

    docs = [doc(f"f{i}", summary=f"a summary {i}", author="Someone") for i in range(15)]
    stock(store, docs)
    chosen = [(docs[0], "feed", 0.9), (docs[1], "feed-random", None)]
    out = shortlist_stats.spotcheck(store.documents(), chosen, store, seed=1)
    assert [e["slot"] for e in out["shortlist"]] == ["feed", "feed-random"]
    assert len(out["random_feed"]) == 10
    first = out["shortlist"][0]
    assert first["embedded_text"] == recommender_embed.document_text(store.document("f0"))


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(html_status_config, "STATS_DIR", str(tmp_path / "stats"))
    monkeypatch.setattr(html_status_config, "HTML_SERVE_DIR", str(tmp_path / "html"))
    return tmp_path


def test_the_ab_table_ignores_the_control_history_before_interleaving(store):
    stock(store, [doc("old"), doc("a"), doc("c")])
    store.log_event("old", "added", "2026-09-01", slot="feed")
    assert shortlist_stats.interleaving_since(store) is None
    store.log_event("a", "added", "2026-10-09", slot="feed")
    store.log_event("c", "added", "2026-10-09", slot="feed-cand")
    since = shortlist_stats.interleaving_since(store)
    arms = shortlist_stats.live_arms(store, since=since)
    assert since == "2026-10-09"
    assert arms["feed"]["shown"] == 1 and arms["feed-cand"]["shown"] == 1


def test_subpages_render_with_no_runs_and_with_old_rows(paths):
    assert "no interleaved run" in stats_page.render_refinement([])
    assert "no run recorded" in stats_page.render_spotcheck([])
    stats_store.record("shortlist", {"run_at": NOW.isoformat()})
    rows = stats_store.load("shortlist")
    assert "no interleaved run" in stats_page.render_refinement(rows)
    assert "no run recorded" in stats_page.render_spotcheck(rows)


def full_row():
    auc = {"auc": 0.8, "low": 0.75, "high": 0.85, "n": 100, "positives": 10}
    return {
        "run_at": NOW.isoformat(), "rankers": config.RANKERS,
        "arms_ab_since": "2026-10-09",
        "arms_ab": {"feed": {"shown": 4, "opened": 2, "read": 1},
                    "feed-cand": {"shown": 4, "opened": 3, "read": 2}},
        "refine": {"rankers": {"control": {"shown": auc, "triage": None, "train_n": 5},
                               "candidate": {"shown": auc, "triage": auc, "train_n": 6}},
                   "word_count_baseline": {"shown": auc, "triage": auc},
                   "top_fresh": {"control": [{"title": "<b>bold</b>", "url": "u", "score": 0.5}]}},
        "spotcheck": {"shortlist": [{"title": "t", "slot": "feed", "url": "https://read.readwise.io/read/x",
                                     "embedded_text": "see https://x.com/someone/status/1 and https://example.org/a"}],
                      "random_feed": []},
    }


def test_the_refinement_page_reports_both_arms_and_escapes_titles(paths):
    stats_store.record("shortlist", full_row())
    markup = stats_page.render_refinement(stats_store.load("shortlist"))
    assert "candidate" in markup and "0.800" in markup
    assert "<b>bold</b>" not in markup and "&lt;b&gt;" in markup
    assert "75.0%" in markup                       # candidate open rate, 3 of 4


def test_the_spotcheck_never_shows_a_platform_link(paths):
    stats_store.record("shortlist", full_row())
    markup = stats_page.render_spotcheck(stats_store.load("shortlist"))
    assert "x.com" not in markup
    assert "example.org/a" in markup
    assert "read.readwise.io/read/x" in markup


def test_subpages_land_under_the_recommender_page(paths):
    stats_store.record("shortlist", full_row())
    stats_page.render_recommender_page()
    for sub in ("refinement", "summaries"):
        assert (paths / "html" / "recommender" / sub / "index.html").exists()
    main = (paths / "html" / "recommender" / "index.html").read_text()
    assert 'href="refinement/"' in main and 'href="summaries/"' in main
