#!/usr/bin/env python3
"""Tests for the Readwise shortlist recommender.

Covers the failure modes that would actually hurt: evicting a document the reader
shortlisted by hand, re-showing the same thing every night, letting document age
leak into the model, and writing anything at all during a dry run.

No network and no model load -- embeddings are written straight into the store,
so these stay well inside the 5s timeout.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import config
import recommender_model
import shortlist_job
from html_status import config as html_status_config
from readwise_client import ReadwiseClient, ReadwiseError, tag_names
from recommender_store import Store

NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)


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
        "word_count": 800,
        "saved_at": iso(1),
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


def stock(store, documents, embed=True):
    store.upsert_documents(documents)
    if embed:
        # Centred and normalized, like real embeddings -- uniform [0,1) vectors
        # all sit in one orthant and read as near-duplicates of each other.
        rng = np.random.default_rng(0)
        vectors = {}
        for d in documents:
            v = rng.standard_normal(config.EMBED_DIM).astype(np.float32)
            vectors[d["id"]] = v / np.linalg.norm(v)
        store.save_embeddings(config.EMBED_KEY, vectors)


# -- fake transport ----------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def client_with(responses):
    session = FakeSession(responses)
    return ReadwiseClient(token="t", session=session, sleep=lambda _: None), session


# -- client ------------------------------------------------------------------


def test_documents_follows_the_cursor_to_the_last_page():
    client, session = client_with(
        [
            FakeResponse(payload={"results": [{"id": "a"}], "nextPageCursor": "c1"}),
            FakeResponse(payload={"results": [{"id": "b"}], "nextPageCursor": None}),
        ]
    )
    assert [d["id"] for d in client.documents(location="feed")] == ["a", "b"]
    assert session.calls[1][2]["params"]["pageCursor"] == "c1"


def test_rate_limit_is_retried_after_the_header_says_so():
    client, _ = client_with(
        [
            FakeResponse(status=429, headers={"Retry-After": "1"}),
            FakeResponse(payload={"results": [{"id": "a"}], "nextPageCursor": None}),
        ]
    )
    assert [d["id"] for d in client.documents()] == ["a"]


def test_a_real_error_raises_instead_of_returning_empty():
    client, _ = client_with([FakeResponse(status=500, payload={"detail": "boom"})])
    with pytest.raises(ReadwiseError):
        list(client.documents())


def test_bulk_updates_are_chunked_to_the_documented_fifty():
    client, session = client_with([FakeResponse(payload={}), FakeResponse(payload={})])
    client.bulk_set_tags({f"d{i}": ["shortlist"] for i in range(51)})
    assert len(session.calls) == 2
    assert len(session.calls[0][2]["json"]["updates"]) == 50
    assert len(session.calls[1][2]["json"]["updates"]) == 1


def test_tag_names_handles_both_shapes_the_api_uses():
    assert tag_names({"tags": {"shortlist": {}, "favorite": {}}}) == ["favorite", "shortlist"]
    assert tag_names({"tags": ["b", "a"]}) == ["a", "b"]
    assert tag_names({}) == []


# -- store -------------------------------------------------------------------


def test_documents_and_embeddings_survive_a_round_trip(store):
    stock(store, [doc("a", tags={"favorite": {}})], embed=False)
    store.save_embeddings(config.EMBED_KEY, {"a": np.arange(4, dtype=np.float32)})
    assert json.loads(store.document("a")["tags"]) == ["favorite"]
    assert list(store.embeddings(config.EMBED_KEY)["a"]) == [0.0, 1.0, 2.0, 3.0]


def test_missing_embeddings_only_lists_what_is_actually_missing(store):
    stock(store, [doc("a"), doc("b")], embed=False)
    store.save_embeddings(config.EMBED_KEY, {"a": np.zeros(4, dtype=np.float32)})
    assert [d["id"] for d in store.missing_embeddings(config.EMBED_KEY)] == ["b"]


# -- labels ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"tags": {config.RATE_GOOD_TAG: {}}}, (1, "rated")),
        ({"tags": {config.RATE_BAD_TAG: {}}}, (0, "rated")),
        ({"tags": {"favorite": {}}}, (1, "favorited")),
        ({"location": "archive", "reading_progress": 0.9}, (1, "read")),
        ({"first_opened_at": iso(2), "reading_progress": 0.2}, (1, "opened")),
        ({"location": "feed", "saved_at": iso(30)}, (0, "ignored")),
    ],
)
def test_each_labelling_rule_fires(kwargs, expected):
    assert recommender_model.label_for(doc("x", **kwargs), now=NOW) == expected


def test_a_fresh_unopened_feed_item_is_not_yet_a_negative():
    assert recommender_model.label_for(doc("x", saved_at=iso(1)), now=NOW) is None


def test_opened_and_immediately_abandoned_stays_out_of_training():
    """Ambiguous: could be a bounce, could be a deliberate save. Don't guess."""
    assert (
        recommender_model.label_for(
            doc("x", first_opened_at=iso(1), reading_progress=0.02), now=NOW
        )
        is None
    )


def test_a_long_read_counts_even_when_it_was_never_finished():
    """A third of a 9000-word essay is a bigger commitment than finishing a
    300-word note, and a percentage-only rule scores it as the lesser one."""
    assert recommender_model.label_for(
        doc("x", word_count=9000, reading_progress=0.33, first_opened_at=iso(1)), now=NOW
    ) == (1, "read")


def test_finishing_something_very_short_is_not_a_deep_read():
    """80 words consumed is not evidence of much, whatever the percentage says."""
    assert recommender_model.label_for(
        doc("x", word_count=100, reading_progress=0.6, first_opened_at=iso(1)), now=NOW
    ) == (None if config.OPENED_WORDS > 60 else (1, "opened"))


def test_deliberately_finishing_a_short_post_still_counts():
    assert recommender_model.label_for(
        doc("x", word_count=300, reading_progress=0.95, first_opened_at=iso(1)), now=NOW
    ) == (1, "read")


def test_a_shortlisted_document_he_never_opened_becomes_a_negative():
    assert recommender_model.label_for(
        doc("x", saved_at=iso(1)), evicted_unopened=True, now=NOW
    ) == (0, "passed")


def test_rating_tags_outrank_behaviour(store):
    """He said bad; the fact that he read it does not override him saying so."""
    assert recommender_model.label_for(
        doc("x", tags={config.RATE_BAD_TAG: {}}, reading_progress=0.99), now=NOW
    ) == (0, "rated")


def test_derive_labels_uses_the_eviction_log(store):
    stock(store, [doc("a", saved_at=iso(1))])
    assert "a" not in recommender_model.derive_labels(store, now=NOW)
    store.log_event("a", "added", "2026-09-20")
    store.log_event("a", "evicted", "2026-09-21")
    assert recommender_model.derive_labels(store, now=NOW)["a"] == (
        0,
        config.LABEL_WEIGHTS["passed"],
        "passed",
    )


def test_a_same_day_eviction_is_not_evidence_he_passed(store):
    """Two runs in one night must not manufacture negatives out of scheduling."""
    stock(store, [doc("a", saved_at=iso(1))])
    store.log_event("a", "added", "2026-09-20")
    store.log_event("a", "evicted", "2026-09-20")
    assert "a" not in recommender_model.derive_labels(store, now=NOW)


# -- the exposure labelling scheme --------------------------------------------


def block_docs(prefix, opened, unopened, days_ago):
    """One arrival block: `opened` docs he opened and bounced, `unopened` he skipped."""
    out = [doc(f"{prefix}o{i}", saved_at=iso(days_ago), first_opened_at=iso(days_ago))
           for i in range(opened)]
    out += [doc(f"{prefix}u{i}", saved_at=iso(days_ago)) for i in range(unopened)]
    return out


def test_a_skip_he_saw_weighs_more_than_one_he_never_saw(store, monkeypatch):
    """The point of the scheme: 9 of 10 opened means the skip was a real choice."""
    monkeypatch.setattr(config, "EXPOSURE_BLOCK", 10)
    stock(store, block_docs("scanned", 9, 1, 40) + block_docs("flood", 1, 9, 20))
    labels = recommender_model.derive_labels(store, now=NOW, scheme="exposure")
    hard, soft = labels["scannedu0"], labels["floodu0"]
    assert hard[0] == soft[0] == 0
    assert hard[2] == soft[2] == "ignored"
    assert hard[1] == pytest.approx(config.EXPOSURE_WEIGHTS["ignored"] * 0.9)
    assert soft[1] == pytest.approx(config.EXPOSURE_WEIGHTS["ignored"] * 0.1)


def test_a_block_he_never_opened_contributes_no_negatives(store, monkeypatch):
    monkeypatch.setattr(config, "EXPOSURE_BLOCK", 10)
    stock(store, block_docs("flood", 0, 10, 20))
    assert recommender_model.derive_labels(store, now=NOW, scheme="exposure") == {}


def test_shortlist_opens_are_not_evidence_he_scanned_the_block(store, monkeypatch):
    """He opened it because we put it in front of him, not because he browsed."""
    monkeypatch.setattr(config, "EXPOSURE_BLOCK", 10)
    stock(store, block_docs("b", 5, 5, 20))
    for i in range(5):
        store.log_event(f"bo{i}", "added", "2026-09-01")
    labels = recommender_model.derive_labels(store, now=NOW, scheme="exposure")
    assert "bu0" not in labels                    # organic open rate is 0/5
    assert labels["bo0"] == (0, config.EXPOSURE_WEIGHTS["bounced"], "bounced")


def test_positives_keep_full_weight_however_little_he_scanned(store, monkeypatch):
    monkeypatch.setattr(config, "EXPOSURE_BLOCK", 10)
    stock(store, block_docs("flood", 0, 9, 20) + [
        doc("good", saved_at=iso(20), tags={config.RATE_GOOD_TAG: {}}),
    ])
    labels = recommender_model.derive_labels(store, now=NOW, scheme="exposure")
    assert labels["good"] == (1, config.EXPOSURE_WEIGHTS["rated_good"], "rated_good")


def test_a_hand_saved_article_is_not_a_saved_positive(store):
    """Everything hand-saved lives in `later`; only a move out of the feed is a vote."""
    stock(store, [
        doc("moved", location="later", category="rss", saved_at=iso(30)),
        doc("mine", location="later", category="article", saved_at=iso(30)),
    ])
    labels = recommender_model.derive_labels(store, now=NOW, scheme="exposure")
    assert labels["moved"] == (1, config.EXPOSURE_WEIGHTS["saved"], "saved")
    assert "mine" not in labels


def test_an_unknown_scheme_fails_loudly(store):
    with pytest.raises(ValueError):
        recommender_model.derive_labels(store, now=NOW, scheme="exposre")


# -- publication as a feature ------------------------------------------------


def test_every_document_lights_exactly_one_publication_column():
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    pubs = ["the guardian", "wired"]
    width = config.EMBED_DIM + 1 + len(recommender_model.CATEGORIES)
    for name in ("The Guardian ", "wired", "nobody.example", None):
        x = recommender_model.features(doc("a", site_name=name), vector, pubs)
        assert len(x) == recommender_model.feature_count(pubs)
        assert x[width:].sum() == 1.0
    other = recommender_model.features(doc("a", site_name="nobody.example"), vector, pubs)
    assert other[-1] == 1.0


def test_publications_add_no_age_signal():
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    old = recommender_model.features(doc("a", saved_at=iso(900)), vector, ["example.com"])
    new = recommender_model.features(doc("a", saved_at=iso(0)), vector, ["example.com"])
    assert np.array_equal(old, new)


def test_a_model_scores_with_the_vocabulary_it_was_fitted_on(store, monkeypatch):
    monkeypatch.setattr(config, "PUBLICATION_MIN_DOCS", 5)
    good = [doc(f"g{i}", site_name="good.example", location="archive",
                reading_progress=1.0) for i in range(30)]
    bad = [doc(f"b{i}", site_name="bad.example", saved_at=iso(30),
               tags={config.RATE_BAD_TAG: {}}) for i in range(30)]
    stock(store, good + bad + [doc("new", site_name="good.example")])
    model, _ = recommender_model.train(store, now=NOW, scheme="exposure", publications=True)
    assert recommender_model.publications_of(model) == ["bad.example", "good.example"]
    scores = recommender_model.score_documents(
        [store.document("new")], store.embeddings(config.EMBED_KEY), model)
    assert scores["new"] > 0.5


def test_train_keeps_an_evaluation_set_out(store):
    stock(store, [doc(f"r{i}", location="archive", reading_progress=1.0) for i in range(30)]
          + [doc(f"i{i}", saved_at=iso(30)) for i in range(30)])
    _, stats = recommender_model.train(store, now=NOW, exclude={"r0", "i0"})
    assert stats["usable"] == 58


# -- what gets embedded ------------------------------------------------------


def test_the_embedded_text_carries_author_and_publication():
    """Both recur constantly in this corpus and carry a lot of the signal."""
    import recommender_embed

    text = recommender_embed.document_text(
        doc("a", title="On Stuff", author="Zvi Mowshowitz", site_name="lesswrong.com")
    )
    assert "Zvi Mowshowitz" in text
    assert "lesswrong.com" in text
    assert "On Stuff" in text


def test_a_superseded_text_recipe_does_not_linger_in_the_store(store):
    """After a version bump, anything still on the old recipe is stale, not usable."""
    stock(store, [doc("a"), doc("b")], embed=False)
    store.save_embeddings("old-recipe", {"a": np.zeros(4, dtype=np.float32)})
    store.save_embeddings(config.EMBED_KEY, {"b": np.zeros(4, dtype=np.float32)})
    assert [d["id"] for d in store.missing_embeddings(config.EMBED_KEY)] == ["a"]
    assert store.forget_other_embeddings(config.EMBED_KEY) == 1
    assert list(store.embeddings(config.EMBED_KEY)) == ["b"]


# -- features ----------------------------------------------------------------


def test_the_feature_vector_carries_no_age_signal():
    """Age leaks the label (old -> archived -> read), so it must not be a feature."""
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    old = recommender_model.features(doc("a", saved_at=iso(900)), vector)
    new = recommender_model.features(doc("a", saved_at=iso(0)), vector)
    assert np.array_equal(old, new)
    assert len(old) == config.EMBED_DIM + 1 + len(recommender_model.CATEGORIES)


# -- selection ---------------------------------------------------------------


def taste():
    return np.ones(config.EMBED_DIM, dtype=np.float32) / np.sqrt(config.EMBED_DIM)


def test_pick_leaves_a_hand_shortlisted_document_alone(store):
    stock(store, [doc(f"f{i}") for i in range(10)] + [doc("mine", tags={"shortlist": {}})])
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    assert "mine" not in {d["id"] for d, _, _ in chosen}


def test_pick_respects_the_reshow_cooldown(store):
    stock(store, [doc(f"f{i}") for i in range(10)])
    store.log_event("f0", "added", (NOW - timedelta(days=5)).date().isoformat())
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    assert "f0" not in {d["id"] for d, _, _ in chosen}


def test_pick_fills_both_pools_with_their_own_random_arm(store):
    stock(
        store,
        [doc(f"f{i}") for i in range(20)]
        + [doc(f"l{i}", location="later", saved_at=iso(400)) for i in range(20)],
    )
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    slots = [slot for _, slot, _ in chosen]
    assert len(chosen) == config.SHORTLIST_SIZE
    assert slots.count("feed") == config.FEED_SLOTS - config.FEED_RANDOM_SLOTS
    assert slots.count("feed-random") == config.FEED_RANDOM_SLOTS
    assert slots.count("later") == config.LATER_SLOTS - config.LATER_RANDOM_SLOTS
    assert slots.count("later-random") == config.LATER_RANDOM_SLOTS


def test_the_random_slots_are_never_scored(store):
    """They are the measurement arms -- ranking them defeats the whole point."""
    stock(
        store,
        [doc(f"f{i}") for i in range(20)]
        + [doc(f"l{i}", location="later") for i in range(20)],
    )
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    assert all(score is None for _, slot, score in chosen if slot.endswith("-random"))


def test_each_random_arm_is_drawn_from_its_own_pool(store):
    """Comparing a random draw from `later` against ranked picks from `feed` would
    measure age and prior selection, not the quality of the ranking."""
    stock(
        store,
        [doc(f"f{i}") for i in range(20)]
        + [doc(f"l{i}", location="later", saved_at=iso(400)) for i in range(20)],
    )
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    by_slot = {slot: {d["id"] for d, s, _ in chosen if s == slot} for slot in
               ("feed", "feed-random", "later", "later-random")}
    assert all(i.startswith("f") for i in by_slot["feed"] | by_slot["feed-random"])
    assert all(i.startswith("l") for i in by_slot["later"] | by_slot["later-random"])
    assert by_slot["feed-random"] and by_slot["later-random"]


def test_no_document_takes_two_slots_at_once(store):
    stock(store, [doc(f"f{i}") for i in range(12)] + [doc("l0", location="later")])
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    ids = [d["id"] for d, _, _ in chosen]
    assert len(ids) == len(set(ids))


def test_ranked_picks_skip_near_duplicates(store):
    """The first live dry run returned three copies of one newsletter."""
    store.upsert_documents([doc(f"dupe{i}") for i in range(6)] + [doc("other")])
    same = np.ones(config.EMBED_DIM, dtype=np.float32)
    same /= np.linalg.norm(same)
    different = np.zeros(config.EMBED_DIM, dtype=np.float32)
    different[0] = 1.0
    vectors = {f"dupe{i}": same for i in range(6)}
    vectors["other"] = different
    store.save_embeddings(config.EMBED_KEY, vectors)

    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    ranked = [d["id"] for d, slot, _ in chosen if slot == "feed"]
    assert len([i for i in ranked if i.startswith("dupe")]) == 1


def test_the_random_arm_is_not_deduplicated(store):
    """Skipping duplicates there would stop it being a uniform draw."""
    store.upsert_documents([doc(f"dupe{i}") for i in range(6)])
    same = np.ones(config.EMBED_DIM, dtype=np.float32)
    same /= np.linalg.norm(same)
    store.save_embeddings(config.EMBED_KEY, {f"dupe{i}": same for i in range(6)})
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    assert sum(1 for _, slot, _ in chosen if slot == "feed") == 1
    assert sum(1 for _, slot, _ in chosen if slot == "feed-random") == config.FEED_RANDOM_SLOTS


def test_short_reads_are_guaranteed_slots(store):
    """Unconstrained, the ranker builds a long-form monoculture: 67% of its picks
    were 2500+ words against a pool that is 57% short."""
    long_docs = [doc(f"f{i}", word_count=5000) for i in range(20)]
    short_docs = [doc(f"s{i}", word_count=300) for i in range(20)]
    stock(store, long_docs + short_docs)
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    ranked_feed = [d for d, slot, _ in chosen if slot == "feed"]
    shorts = [d for d in ranked_feed if d["word_count"] < config.SHORT_WORDS]
    assert len(shorts) >= config.FEED_SHORT_SLOTS


def test_the_quota_is_a_floor_not_a_cap(store):
    """If a pool is all short, every pick is short -- no artificial long quota."""
    stock(store, [doc(f"s{i}", word_count=300) for i in range(20)])
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    ranked = [d for d, slot, _ in chosen if slot == "feed"]
    assert len(ranked) == config.FEED_SLOTS - config.FEED_RANDOM_SLOTS


class AxisModel:
    """A stand-in ranker that likes whatever points along one embedding axis."""

    def __init__(self, axis):
        self.axis = axis

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-10 * X[:, self.axis]))
        return np.column_stack([1 - p, p])


def two_camps(store, n=12):
    """Feed docs a0.. point along axis 0, b0.. along axis 1; same for `later`."""
    docs, vectors = [], {}
    for pool, extra in (("", {}), ("l", {"location": "later", "saved_at": iso(400)})):
        for camp, axis in (("a", 0), ("b", 1)):
            for i in range(n):
                d = doc(f"{pool}{camp}{i}", **extra)
                v = np.zeros(config.EMBED_DIM, dtype=np.float32)
                v[axis] = 1.0
                v[2 + i + (n if pool else 0) + (2 * n if camp == "b" else 0)] = 0.8
                vectors[d["id"]] = v / np.linalg.norm(v)
                docs.append(d)
    store.upsert_documents(docs)
    store.save_embeddings(config.EMBED_KEY, vectors)


def test_interleaving_splits_the_ranked_slots_evenly(store):
    two_camps(store)
    chosen = shortlist_job.pick(store, AxisModel(0), None, now=NOW, candidate=AxisModel(1))
    slots = [slot for _, slot, _ in chosen]
    ranked_feed = config.FEED_SLOTS - config.FEED_RANDOM_SLOTS
    ranked_later = config.LATER_SLOTS - config.LATER_RANDOM_SLOTS
    assert slots.count("feed") == slots.count("feed-cand") == ranked_feed // 2
    assert slots.count("later") + slots.count("later-cand") == ranked_later
    assert abs(slots.count("later") - slots.count("later-cand")) <= 1
    assert slots.count("feed-random") == config.FEED_RANDOM_SLOTS
    assert len(chosen) == config.SHORTLIST_SIZE


def test_each_ranker_fills_its_slots_with_its_own_favourites(store):
    """Otherwise the arm comparison is measuring the draft, not the rankers."""
    two_camps(store)
    chosen = shortlist_job.pick(store, AxisModel(0), None, now=NOW, candidate=AxisModel(1))
    assert all(d["id"].lstrip("l").startswith("a") for d, s, _ in chosen if s in ("feed", "later"))
    assert all(d["id"].lstrip("l").startswith("b") for d, s, _ in chosen
               if s in ("feed-cand", "later-cand"))


def test_two_rankers_that_agree_never_pick_the_same_document_twice(store):
    two_camps(store)
    chosen = shortlist_job.pick(store, AxisModel(0), None, now=NOW, candidate=AxisModel(0))
    ids = [d["id"] for d, _, _ in chosen]
    assert len(ids) == len(set(ids))
    assert [s for _, s, _ in chosen].count("feed-cand") == \
        (config.FEED_SLOTS - config.FEED_RANDOM_SLOTS) // 2


def test_pick_skips_stale_feed_items(store):
    stock(store, [doc("old", saved_at=iso(60))])
    assert shortlist_job.pick(store, None, taste(), now=NOW) == []


def test_evictable_never_includes_a_document_we_did_not_add(store):
    stock(store, [doc("mine", tags={"shortlist": {}}), doc("ours")])
    store.log_event("ours", "added", "2026-09-22")
    assert shortlist_job.evictable(store) == ["ours"]


def test_evictable_forgets_documents_already_evicted(store):
    stock(store, [doc("a")])
    store.log_event("a", "added", "2026-09-20")
    store.log_event("a", "evicted", "2026-09-21")
    assert shortlist_job.evictable(store) == []


# -- the dry run -------------------------------------------------------------


class RecordingClient:
    def __init__(self):
        self.writes = []

    def check_auth(self):
        return True

    def documents(self, **kwargs):
        return iter(())

    def bulk_set_tags(self, doc_tags):
        self.writes.append(doc_tags)


def test_a_dry_run_writes_nothing(store, tmp_path, monkeypatch):
    stock(
        store,
        [doc(f"f{i}") for i in range(12)]
        + [doc(f"l{i}", location="later") for i in range(12)]
        + [doc("read", location="archive", reading_progress=0.9)],
    )
    store.close()
    recording = RecordingClient()
    monkeypatch.setattr(config, "RECOMMENDER_DB", str(tmp_path / "t.sqlite3"))
    monkeypatch.setattr(shortlist_job, "ReadwiseClient", lambda *a, **k: recording)
    monkeypatch.setattr(shortlist_job, "embed_missing",
                        lambda *a, **k: {"pending": 0, "embedded": 0,
                                         "dropped_superseded": 0})
    monkeypatch.setattr(html_status_config, "STATS_DIR", str(tmp_path / "stats"))
    monkeypatch.setattr(html_status_config, "HTML_SERVE_DIR", str(tmp_path / "html"))

    assert shortlist_job.run(dry_run=True, skip_sync=True) == 0
    assert recording.writes == []
    # a dry run still publishes, flagged as one -- the page is how you inspect it
    page = tmp_path / "html" / html_status_config.RECOMMENDER_PAGE / "index.html"
    assert page.exists()
    assert "dry run" in page.read_text()


def test_a_bulk_clear_is_not_a_bounce(store, monkeypatch):
    """A "mark all seen" burst opens everything at once; he never looked at most of it."""
    monkeypatch.setattr(config, "EXPOSURE_BLOCK", 100)
    burst = [doc(f"c{i}", saved_at=iso(40), first_opened_at="2026-05-01T10:00:0%dZ" % (i % 10))
             for i in range(config.BULK_CLEAR_OPENS)]
    burst[0]["tags"] = {config.RATE_BAD_TAG: {}}
    stock(store, burst + [doc("alone", saved_at=iso(40), first_opened_at=iso(39))])
    labels = recommender_model.derive_labels(store, now=NOW, scheme="exposure")
    assert "c1" not in labels
    assert labels["c0"][2] == "rated_bad"        # an explicit rating still counts
    assert labels["alone"][2] == "bounced"


@pytest.mark.parametrize("words, progress, read", [
    (5000, 0.02, True),     # 100 words into a long post
    (5000, 0.01, False),
    (300, 0.10, True),      # 10% of a short post
    (300, 0.09, False),
])
def test_any_real_reading_is_a_positive(store, words, progress, read):
    stock(store, [doc("d", saved_at=iso(20), first_opened_at=iso(19),
                      word_count=words, reading_progress=progress)])
    label = recommender_model.derive_labels(store, now=NOW, scheme="exposure")["d"]
    assert label[2] == ("read" if read else "bounced")
