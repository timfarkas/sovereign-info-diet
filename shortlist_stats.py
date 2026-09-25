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


def rating(doc):
    """His explicit verdict, if he gave one. The strongest signal in the system.

    Kept apart from `outcome` on purpose: "he read it" and "he liked it" are
    different facts, and a document he read to the end and then tagged
    `rate:bad` must not be allowed to read as a win for the arm that picked it.
    """
    tags = _tags(doc)
    if config.RATE_GOOD_TAG in tags:
        return "good"
    if config.RATE_BAD_TAG in tags:
        return "bad"
    return None


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
        arm = arms.setdefault(slot, {"shown": 0, "opened": 0, "read": 0,
                                     "rated_good": 0, "rated_bad": 0})
        arm["shown"] += 1
        state = outcome(doc)
        if state in ("opened", "read"):
            arm["opened"] += 1
        if state == "read":
            arm["read"] += 1
        verdict = rating(doc)
        if verdict:
            arm[f"rated_{verdict}"] += 1
    for arm in arms.values():
        arm["open_rate"] = arm["opened"] / arm["shown"] if arm["shown"] else None
        arm["read_rate"] = arm["read"] / arm["shown"] if arm["shown"] else None
    return arms


def overnight_signals(documents, since):
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
    for doc in documents:
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


def positive_matrix(labels, embeddings, limit=2000):
    """(ids, matrix) of documents he demonstrably read, for nearest-neighbour use."""
    ids = [doc_id for doc_id, (y, _, reason) in labels.items()
           if y == 1 and reason in ("rated", "favorited", "read") and doc_id in embeddings]
    ids = ids[:limit]
    if not ids:
        return None
    return ids, np.vstack([embeddings[i] for i in ids])


def labelled_matrix(labels, embeddings, limit=8000):
    """(ids, matrix, meta) over every labelled document. meta[i] = (y, reason).

    This is the evidence base for the per-pick explanation: the model is linear
    on a frozen encoder, so "what does this look like that he has already judged"
    is the only local explanation of the topic term that means anything to a
    human. A bar per embedding dimension would be 384 bars of noise.
    """
    ids, meta = [], []
    for doc_id, (y, _, reason) in labels.items():
        if doc_id in embeddings and len(ids) < limit:
            ids.append(doc_id)
            meta.append((y, reason))
    if not ids:
        return None
    return ids, np.vstack([embeddings[i] for i in ids]), meta


def evidence(doc_id, embeddings, labelled, k=3):
    """The k labelled documents this pick most resembles, on each side.

    Returns {"like": [...], "unlike": [...]} -- nearest positives and nearest
    negatives by cosine. Read it as "it looks like these things you finished and
    these things you ignored", which is a claim you can argue with.
    """
    vector = embeddings.get(doc_id)
    if vector is None or not labelled:
        return None
    ids, matrix, meta = labelled
    similarities = matrix @ vector
    out = {}
    for side, want in (("like", 1), ("unlike", 0)):
        hits = [i for i in range(len(ids)) if meta[i][0] == want and ids[i] != doc_id]
        hits.sort(key=lambda i: -similarities[i])
        out[side] = [{"id": ids[i], "similarity": float(similarities[i]),
                      "reason": meta[i][1]} for i in hits[:k]]
    return out


def baseline_features(docs, embeddings):
    """The average document, in feature space. The origin the split is taken about.

    Attributing against zero would say "long documents score high" even when
    every candidate is long; attributing against the mean says what makes THIS
    document different from the rest of the corpus, which is the question.
    """
    if not docs or not embeddings:
        return None
    vectors = [embeddings[d["id"]] for d in docs if d["id"] in embeddings]
    if not vectors:
        return None
    mean_vector = np.mean(np.vstack(vectors), axis=0)
    extras = np.mean(
        np.vstack([recommender_model.features(d, np.zeros(0, dtype=np.float32))
                   for d in docs]),
        axis=0,
    )
    return np.concatenate([mean_vector, extras])


GROUPS = ("topic", "length", "format")


def attribution(doc, vector, model, baseline):
    """Split the logit into topic / length / format, around the average document.

    Exact for a linear model: baseline_logit + sum(contributions) == logit, so
    the bars add up to the score rather than merely gesturing at it.
    """
    if model is None or baseline is None or vector is None:
        return None
    x = recommender_model.features(doc, vector)
    w = model.coef_[0]
    if len(w) != len(x) or len(baseline) != len(x):
        return None
    delta = w * (x - baseline)
    dim = config.EMBED_DIM
    contributions = {
        "topic": float(delta[:dim].sum()),
        "length": float(delta[dim]),
        "format": float(delta[dim + 1:].sum()),
    }
    baseline_logit = float(model.intercept_[0] + w @ baseline)
    logit = baseline_logit + sum(contributions.values())
    return {
        "baseline_logit": baseline_logit,
        "baseline_probability": float(1 / (1 + np.exp(-baseline_logit))),
        "contributions": contributions,
        "logit": logit,
        "probability": float(1 / (1 + np.exp(-logit))),
        "word_count": doc.get("word_count"),
        "category": doc.get("category"),
    }


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
          holdout, trained, sync_info, embed_info, durations, model=None,
          problems=None):
    """Assemble the run row the status page renders."""
    embeddings = store.embeddings(config.EMBED_KEY)
    # One pass over the corpus, shared by everything below. Each of these used to
    # pull its own copy of 14.7k documents and re-derive the labels, which cost
    # 320 MB of peak RSS for no information.
    documents = store.documents()
    labels = recommender_model.derive_labels(store)
    positives = positive_matrix(labels, embeddings)
    labelled = labelled_matrix(labels, embeddings)
    baseline = baseline_features(documents, embeddings)

    def titled(items):
        for item in items or []:
            doc = store.document(item["id"]) or {}
            item["title"] = (doc.get("title") or item["id"])[:90]
        return items

    picks = []
    for doc, slot, score in chosen:
        detail = trace.get(doc["id"], {})
        near = nearest_read(doc["id"], embeddings, positives)
        if near:
            neighbour = store.document(near["id"]) or {}
            near = {"title": (neighbour.get("title") or near["id"])[:90],
                    "similarity": near["similarity"]}
        # The control arm gets no attribution on purpose: it was never scored,
        # and printing what the model would have said turns the measurement
        # arm into another one of the model's opinions.
        explain = None
        if not slot.endswith("-random"):
            explain = attribution(doc, embeddings.get(doc["id"]), model, baseline)
            near_by_label = evidence(doc["id"], embeddings, labelled)
            if near_by_label and explain:
                explain["like"] = titled(near_by_label["like"])
                explain["unlike"] = titled(near_by_label["unlike"])
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
            "attribution": explain,
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
            "rating": rating(doc),
            "progress": doc.get("reading_progress"),
        })

    by_location = {}
    for doc in documents:
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
                      else overnight_signals(documents, sync_info.get("since"))),
        "arms": live_arms(store),
        "pools": trace.get("_pools", {}),
        "picks": picks,
        "evicted": evicted,
        "sync": sync_info,
        "embed": embed_info,
        **durations,
    }
