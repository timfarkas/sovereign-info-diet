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


def label_for(doc, evicted_unopened=False, now=None):
    """(y, weight_key) for one document, or None when it carries no usable signal.

    Ordered strongest signal first; the first rule that matches wins.
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
        return 0, "passed"
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
        return 0, "ignored"
    return None


def derive_labels(store, now=None):
    """{doc_id: (y, weight)} across the whole corpus."""
    evicted = {
        e["doc_id"] for e in store.events() if e["action"] == "evicted"
    }
    labels = {}
    for doc in store.documents():
        result = label_for(doc, evicted_unopened=doc["id"] in evicted, now=now)
        if result is None:
            continue
        y, reason = result
        labels[doc["id"]] = (y, config.LABEL_WEIGHTS[reason], reason)
    return labels


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
