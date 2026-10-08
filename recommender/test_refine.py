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
