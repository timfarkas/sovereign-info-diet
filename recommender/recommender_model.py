#!/usr/bin/env python3
"""Labels, features and the ranking model.

Frozen embeddings plus a logistic regression. That is a deliberate ceiling: with
hundreds-to-thousands of labels, a linear head on good embeddings is about as
much model as the data can honestly support, and anything deeper would overfit
and then hide it behind a flattering training score.

No feature here may encode *age*. Old documents are archived and archived means
read, so age leaks the label almost perfectly and would produce a model that
ranks by "is old" while looking excellent on paper. Recency belongs at selection
time, not in the model.
"""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config

CATEGORIES = ["article", "email", "rss", "pdf", "epub", "video", "tweet"]


def _parse(stamp):
    """Always returns UTC-aware. Cycle stamps are bare dates, Readwise's are not."""
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _tags(doc):
    raw = doc.get("tags")
    if isinstance(raw, str):
        return set(json.loads(raw or "[]"))
    if isinstance(raw, dict):
        return set(raw.keys())
    return set(raw or [])


def label_for(doc, evicted_unopened=False, cycle_read_fraction=None, shelved_days=(), now=None):
    """(y, weight_key) for one document, or None when it carries no usable signal.

    Ordered strongest signal first; the first rule that matches wins.

    `cycle_read_fraction` and `shelved_days` supply the day-level context behind
    the two "he just didn't have time" cases below -- callers that don't have a
    store handy (tests, mostly) can omit them and get the conservative, no-context
    answer instead of a fabricated negative.
    """
    now = now or datetime.now(timezone.utc)
    tags = _tags(doc)
    progress = doc.get("reading_progress") or 0.0
    opened = bool(doc.get("first_opened_at"))
    words_read = progress * (doc.get("word_count") or 0)

    if config.RATE_GOOD_TAG in tags:
        return 1, "rated"
    if config.RATE_BAD_TAG in tags:
        return 0, "rated"
    if tags & {"favorite", "important"}:
        return 1, "favorited"
    if words_read >= config.READ_WORDS or progress >= config.READ_PROGRESS:
        return 1, "read"
    if opened and words_read >= config.OPENED_WORDS:
        return 1, "opened"
    if evicted_unopened and not opened:
        # A day he read most of what was shown makes skipping this one a real
        # signal. A quiet day says nothing -- he just didn't have time, and
        # treating it as rejection would manufacture a negative out of his
        # schedule rather than his taste.
        if cycle_read_fraction is not None and cycle_read_fraction > config.CYCLE_READ_MAJORITY:
            return 0, "passed"
        return None
    if doc.get("location") == "archive" and not opened:
        return 0, "archived_unread"
    if opened:
        # Opened and barely read. Could be a bounce, could be a save for later --
        # genuinely ambiguous, so it stays out of training.
        return None
    saved = _parse(doc.get("saved_at"))
    if (
        doc.get("location") == "feed"
        and saved
        and saved < now - timedelta(days=config.FEED_STALE_DAYS)
    ):
        # Same logic as the shortlist case above, applied to the whole feed
        # history: only a signal on days he was actually shelving things, else
        # the firehose simply outran him.
        if saved.date().isoformat() in shelved_days:
            return 0, "ignored"
        return None
    return None


def derive_labels(store, now=None):
    """{doc_id: (y, weight, reason)} across the whole corpus."""
    matured, added_cycle = _shortlist_history(store)
    cycle_fraction = _cycle_read_fractions(store, added_cycle)
    shelved_days = _shelved_days(store)
    labels = {}
    for doc in store.documents():
        result = label_for(
            doc,
            evicted_unopened=doc["id"] in matured,
            cycle_read_fraction=cycle_fraction.get(added_cycle.get(doc["id"])),
            shelved_days=shelved_days,
            now=now,
        )
        if result is None:
            continue
        y, reason = result
        labels[doc["id"]] = (y, config.LABEL_WEIGHTS[reason], reason)
    return labels


def _shortlist_history(store):
    """(matured_evicted_doc_ids, {doc_id: first_added_cycle}) from the event log.

    A document evicted on the same day it was added was never really offered --
    that happens when the job runs twice in one night -- and reading it as a
    rejection would manufacture negatives out of our own scheduling.
    """
    added = {}
    matured = set()
    for event in store.events():
        if event["action"] == "added" and event["doc_id"] not in added:
            added[event["doc_id"]] = event["cycle"]
        elif event["action"] == "evicted":
            first_seen = added.get(event["doc_id"])
            if first_seen and event["cycle"] > first_seen:
                matured.add(event["doc_id"])
    return matured, added


def _cycle_read_fractions(store, added_cycle):
    """{cycle: fraction of that cycle's shortlist that was actually read}."""
    by_cycle = {}
    for doc_id, cycle in added_cycle.items():
        by_cycle.setdefault(cycle, []).append(doc_id)
    docs = {d["id"]: d for d in store.documents()}
    fractions = {}
    for cycle, doc_ids in by_cycle.items():
        read = 0
        for doc_id in doc_ids:
            d = docs.get(doc_id)
            if not d:
                continue
            progress = d.get("reading_progress") or 0.0
            words_read = progress * (d.get("word_count") or 0)
            if words_read >= config.READ_WORDS or progress >= config.READ_PROGRESS:
                read += 1
        fractions[cycle] = read / len(doc_ids)
    return fractions


def _shelved_days(store):
    """Calendar dates (by saved_at) with at least one feed item archived without
    ever being opened -- evidence he was actually triaging the feed that day."""
    days = set()
    for d in store.documents(location="archive"):
        if d.get("first_opened_at"):
            continue
        saved = _parse(d.get("saved_at"))
        if saved:
            days.add(saved.date().isoformat())
    return days


def features(doc, vector):
    """embedding + a couple of shape features. No age, ever -- see module docstring."""
    word_count = doc.get("word_count") or 0
    extras = [np.log1p(word_count)]
    extras += [1.0 if doc.get("category") == c else 0.0 for c in CATEGORIES]
    return np.concatenate([vector, np.asarray(extras, dtype=np.float32)])


def build_matrix(docs, embeddings):
    """(X, kept_docs) for every doc that has an embedding."""
    rows, kept = [], []
    for doc in docs:
        vector = embeddings.get(doc["id"])
        if vector is None:
            continue
        rows.append(features(doc, vector))
        kept.append(doc)
    if not rows:
        return np.zeros((0, config.EMBED_DIM + 1 + len(CATEGORIES)), dtype=np.float32), []
    return np.vstack(rows), kept


def train(store, now=None):
    """Fit on everything labelled. Returns (model, stats) or (None, stats)."""
    labels = derive_labels(store, now=now)
    embeddings = store.embeddings(config.EMBED_KEY)
    docs = [d for d in store.documents() if d["id"] in labels]
    X, kept = build_matrix(docs, embeddings)
    y = np.array([labels[d["id"]][0] for d in kept])
    weights = np.array([labels[d["id"]][1] for d in kept])
    stats = {
        "labelled": len(labels),
        "usable": len(kept),
        "positives": int(y.sum()) if len(y) else 0,
        "by_reason": _count_reasons(labels),
    }
    if len(kept) < config.MIN_LABELS_TO_TRAIN or len(set(y)) < 2:
        return None, stats
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    model.fit(X, y, sample_weight=weights)
    return model, stats


def _count_reasons(labels):
    counts = {}
    for _, _, reason in labels.values():
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def taste_vector(store):
    """Cold-start fallback: the mean embedding of what he has actually read."""
    labels = derive_labels(store)
    embeddings = store.embeddings(config.EMBED_KEY)
    positives = [
        embeddings[doc_id]
        for doc_id, (y, _, _) in labels.items()
        if y == 1 and doc_id in embeddings
    ]
    if not positives:
        return None
    mean = np.mean(positives, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else None


def score_documents(docs, embeddings, model=None, taste=None):
    """{doc_id: score in 0..1}. Uses the model when we have one, taste otherwise."""
    X, kept = build_matrix(docs, embeddings)
    if not kept:
        return {}
    if model is not None:
        scores = model.predict_proba(X)[:, 1]
    elif taste is not None:
        vectors = np.vstack([embeddings[d["id"]] for d in kept])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        scores = (vectors / norms) @ taste
        scores = (scores + 1.0) / 2.0
    else:
        raise ValueError("need either a trained model or a taste vector to score")
    return {doc["id"]: float(s) for doc, s in zip(kept, scores)}


def evaluate(store, holdout_days=30, now=None):
    """Honest gate: train on the past, score the future, compare to baselines.

    Returns None when there is not enough held-out signal to say anything.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=holdout_days)
    labels = derive_labels(store, now=now)
    embeddings = store.embeddings(config.EMBED_KEY)
    docs = [d for d in store.documents() if d["id"] in labels]

    train_docs, test_docs = [], []
    for doc in docs:
        saved = _parse(doc.get("saved_at"))
        (test_docs if saved and saved >= cutoff else train_docs).append(doc)

    X_train, kept_train = build_matrix(train_docs, embeddings)
    X_test, kept_test = build_matrix(test_docs, embeddings)
    if len(kept_train) < config.MIN_LABELS_TO_TRAIN or len(kept_test) < 10:
        return None
    y_train = np.array([labels[d["id"]][0] for d in kept_train])
    y_test = np.array([labels[d["id"]][0] for d in kept_test])
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return None

    weights = np.array([labels[d["id"]][1] for d in kept_train])
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    model.fit(X_train, y_train, sample_weight=weights)
    scores = model.predict_proba(X_test)[:, 1]

    rng = np.random.default_rng(0)
    baselines = {
        "random": rng.random(len(y_test)),
        "word_count": np.array([d.get("word_count") or 0 for d in kept_test], dtype=float),
    }
    k = min(config.SHORTLIST_SIZE, len(y_test))
    result = {
        "train_n": len(kept_train),
        "test_n": len(kept_test),
        "test_positive_rate": float(y_test.mean()),
        "auc": float(roc_auc_score(y_test, scores)),
        f"precision_at_{k}": float(y_test[np.argsort(-scores)[:k]].mean()),
    }
    for name, values in baselines.items():
        result[f"auc_{name}"] = float(roc_auc_score(y_test, values))
        result[f"precision_at_{k}_{name}"] = float(y_test[np.argsort(-values)[:k]].mean())
    return result
