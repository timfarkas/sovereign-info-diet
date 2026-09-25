#!/usr/bin/env python3
"""What the nightly shortlist cycle knew about itself, as plain data.

Everything here is read-only against the store and returns dicts. It exists so
`shortlist_job.run()` stays about deciding things, and so each of these
questions -- "why was this picked", "what happened to the last batch", "does
ranking beat random" -- can be tested on its own.
"""

from datetime import datetime, timezone

import numpy as np

import config
import recommender_model
from recommender_model import _parse, _tags


def outcome(doc):
    """What became of a document we put in front of him. Three states, no more.

    Deliberately coarser than the training labels: this is the human-readable
    "did it land" column, not the thing the model fits on.
    """
    progress = doc.get("reading_progress") or 0.0
    words_read = progress * (doc.get("word_count") or 0)
    if words_read >= config.READ_WORDS or progress >= config.READ_PROGRESS:
        return "read"
    if doc.get("first_opened_at"):
        return "opened"
    return "passed"


def live_arms(store):
    """Ranked vs random, measured on what he actually did. The honest metric.

    Each arm is counted once per document, not once per event, so a document
    that came back after the cooldown does not vote twice.
    """
    seen = {}
    for event in store.events():
        if event["action"] == "added" and event["doc_id"] not in seen:
            seen[event["doc_id"]] = event["slot"] or "unknown"
    arms = {}
    for doc_id, slot in seen.items():
        doc = store.document(doc_id)
        if not doc:
            continue
        arm = arms.setdefault(slot, {"shown": 0, "opened": 0, "read": 0})
        arm["shown"] += 1
        state = outcome(doc)
        if state in ("opened", "read"):
            arm["opened"] += 1
        if state == "read":
            arm["read"] += 1
    for arm in arms.values():
        arm["open_rate"] = arm["opened"] / arm["shown"] if arm["shown"] else None
        arm["read_rate"] = arm["read"] / arm["shown"] if arm["shown"] else None
    return arms


def overnight_signals(store, since):
    """What Readwise reported as changed since the previous sync.

    Honest about what it is: the *current* state of documents Readwise touched
    in that window, not a diff against a snapshot we never took. A document he
    rated three weeks ago and re-opened last night shows up under both "opened"
    and "rate:good" -- that is the point, these are the signals tonight's fit
    saw and last night's did not.
    """
    cutoff = _parse(since)
    counts = dict(updated=0, new_docs=0, rated_good=0, rated_bad=0,
                  opened=0, read=0, archived=0)
    if not cutoff:
        return counts
    for doc in store.documents():
        updated = _parse(doc.get("updated_at"))
        if not updated or updated < cutoff:
            continue
        counts["updated"] += 1
        saved = _parse(doc.get("saved_at"))
        if saved and saved >= cutoff:
            counts["new_docs"] += 1
        tags = _tags(doc)
        if config.RATE_GOOD_TAG in tags:
            counts["rated_good"] += 1
        if config.RATE_BAD_TAG in tags:
            counts["rated_bad"] += 1
        opened = _parse(doc.get("last_opened_at"))
        if opened and opened >= cutoff:
            counts["opened"] += 1
            if outcome(doc) == "read":
                counts["read"] += 1
        if doc.get("location") == "archive":
            moved = _parse(doc.get("last_moved_at"))
            if moved and moved >= cutoff:
                counts["archived"] += 1
    return counts


def nearest_read(doc_id, embeddings, positives):
    """The thing he has already read that is closest to this pick.

    A score is not a reason a human can argue with; "this is 0.84 cosine from
    the Nanda piece you finished" is. Vectors are L2-normalized, so the dot
    product is the cosine.
    """
    vector = embeddings.get(doc_id)
    if vector is None or not positives:
        return None
    ids, matrix = positives
    similarities = matrix @ vector
    best = int(np.argmax(similarities))
    return {"id": ids[best], "similarity": float(similarities[best])}


def positive_matrix(store, embeddings, limit=2000):
    """(ids, matrix) of documents he demonstrably read, for nearest-neighbour use."""
    labels = recommender_model.derive_labels(store)
    ids = [doc_id for doc_id, (y, _, reason) in labels.items()
           if y == 1 and reason in ("rated", "favorited", "read") and doc_id in embeddings]
    ids = ids[:limit]
    if not ids:
        return None
    return ids, np.vstack([embeddings[i] for i in ids])


def describe(doc, slot, score, trace):
    """Why this document is on tonight's list, in one sentence he can check."""
    pool = "fresh feed" if slot.startswith("feed") else "resurfaced backlog"
    size = trace.get("pool_size")
    if slot.endswith("-random"):
        return (f"random control draw from the same {size} {pool} candidates the "
                f"ranked picks came out of -- unranked on purpose, this is the arm "
                f"the ranking gets measured against")
    rank = trace.get("rank")
    parts = [f"ranked #{rank} of {size} {pool} candidates" if rank else f"{pool} pick"]
    if score is not None:
        parts.append(f"score {score:.3f}")
    if trace.get("short_reserve"):
        parts.append(f"took one of the reserved short-read slots at "
                     f"{doc.get('word_count') or 0:,} words")
    return ", ".join(parts)


def build(store, *, cycle, dry_run, chosen, trace, evicted_ids, label_stats,
          holdout, trained, sync_info, embed_info, durations, problems=None):
    """Assemble the run row the status page renders."""
    embeddings = store.embeddings(config.EMBED_KEY)
    positives = positive_matrix(store, embeddings)

    picks = []
    for doc, slot, score in chosen:
        detail = trace.get(doc["id"], {})
        near = nearest_read(doc["id"], embeddings, positives)
        if near:
            neighbour = store.document(near["id"]) or {}
            near = {"title": (neighbour.get("title") or near["id"])[:90],
                    "similarity": near["similarity"]}
        picks.append({
            "id": doc["id"],
            "title": doc.get("title"),
            "author": doc.get("author"),
            "site_name": doc.get("site_name"),
            "category": doc.get("category"),
            "word_count": doc.get("word_count"),
            "url": doc.get("url"),          # read.readwise.io/read/<id>
            "slot": slot,
            "score": score,
            "rank": detail.get("rank"),
            "pool_size": detail.get("pool_size"),
            "reason": describe(doc, slot, score, detail),
            "nearest": near,
        })

    evicted = []
    last_slot = {}
    for event in store.events():
        if event["action"] == "added":
            last_slot[event["doc_id"]] = event["slot"]
    for doc_id in evicted_ids:
        doc = store.document(doc_id)
        if not doc:
            continue
        evicted.append({
            "id": doc_id,
            "title": doc.get("title"),
            "slot": last_slot.get(doc_id),
            "outcome": outcome(doc),
            "progress": doc.get("reading_progress"),
        })

    by_location = {}
    for doc in store.documents():
        by_location[doc.get("location")] = by_location.get(doc.get("location"), 0) + 1

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "dry_run": dry_run,
        "ok": True,
        "problems": problems or [],
        "corpus": {
            "total": sum(by_location.values()),
            "by_location": by_location,
            "embedded": len(embeddings),
        },
        "embed_key": config.EMBED_KEY,
        "features": config.EMBED_DIM + 1 + len(recommender_model.CATEGORIES),
        "labels": label_stats,
        "trained": trained,
        "holdout": holdout,
        # With no sync there is nothing newer to count, and printing zeros would
        # claim he did nothing overnight when the truth is nobody asked.
        "overnight": (None if sync_info.get("skipped")
                      else overnight_signals(store, sync_info.get("since"))),
        "arms": live_arms(store),
        "pools": trace.get("_pools", {}),
        "picks": picks,
        "evicted": evicted,
        "sync": sync_info,
        "embed": embed_info,
        **durations,
    }
