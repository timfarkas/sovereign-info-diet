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
    assert recommender_model.derive_labels(store, now=NOW)["a"][0] == 0


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
    store.upsert_documents([doc(f"dupe{i}") for i in range(4)])
    same = np.ones(config.EMBED_DIM, dtype=np.float32)
    same /= np.linalg.norm(same)
    store.save_embeddings(config.EMBED_KEY, {f"dupe{i}": same for i in range(4)})
    chosen = shortlist_job.pick(store, None, taste(), now=NOW)
    assert sum(1 for _, slot, _ in chosen if slot == "feed-random") == 1


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
    monkeypatch.setattr(shortlist_job, "embed_missing", lambda *a, **k: 0)

    assert shortlist_job.run(dry_run=True, skip_sync=True) == 0
    assert recording.writes == []
