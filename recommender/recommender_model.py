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


def derive_labels(store, now=None, scheme="current"):
    """{doc_id: (y, weight, reason)} across the whole corpus."""
    if scheme == "exposure":
        return exposure_labels(store, now=now)
    if scheme != "current":
        raise ValueError(f"unknown labelling scheme {scheme!r}")
    matured = _matured_evictions(store)
    labels = {}
    for doc in store.documents():
        result = label_for(doc, evicted_unopened=doc["id"] in matured, now=now)
        if result is None:
            continue
        y, reason = result
        labels[doc["id"]] = (y, config.LABEL_WEIGHTS[reason], reason)
    return labels


def exposure_blocks(documents, shortlisted):
    """{doc_id: fraction of its arrival block he organically opened}.

    The arrival stream is everything that came in through the feed, in saved_at
    order, cut into blocks of EXPOSURE_BLOCK. Shortlisted documents belong to a
    block but do not count towards its open rate: opening something because we
    put it in front of him says nothing about whether he scanned the rest.
    """
    stream = sorted(
        (d for d in documents
         if d.get("saved_at")
         and (d.get("location") == "feed" or d.get("category") in ("rss", "email"))),
        key=lambda d: d["saved_at"],
    )
    exposure = {}
    for start in range(0, len(stream), config.EXPOSURE_BLOCK):
        block = stream[start:start + config.EXPOSURE_BLOCK]
        organic = [d for d in block if d["id"] not in shortlisted]
        rate = (sum(bool(d.get("first_opened_at")) for d in organic) / len(organic)
                if organic else 0.0)
        for d in block:
            exposure[d["id"]] = rate
    return exposure


def _words_read(doc):
    return (doc.get("reading_progress") or 0.0) * (doc.get("word_count") or 0)


def _is_read(doc):
    return (_words_read(doc) >= config.READ_WORDS
            or (doc.get("reading_progress") or 0.0) >= config.READ_PROGRESS)


def exposure_labels(store, now=None):
    """The KYRO-30 scheme: {doc_id: (y, weight, reason)}. See config.EXPOSURE_WEIGHTS.

    Positives keep their full weight. Negatives he could not have avoided seeing
    (a shortlist pick he passed on) keep theirs; every other negative is scaled
    by its block's exposure, so the October flood -- hundreds a day, almost none
    opened -- contributes almost nothing instead of thousands of 0.3 negatives.
    """
    now = now or datetime.now(timezone.utc)
    weights = config.EXPOSURE_WEIGHTS
    documents = store.documents()
    shortlisted = {e["doc_id"] for e in store.events() if e["action"] == "added"}
    matured = _matured_evictions(store)
    exposure = exposure_blocks(documents, shortlisted)
    labels = {}
    for doc in documents:
        tags = _tags(doc)
        opened = bool(doc.get("first_opened_at"))
        if config.RATE_GOOD_TAG in tags:
            labels[doc["id"]] = (1, weights["rated_good"], "rated_good")
        elif config.RATE_BAD_TAG in tags:
            labels[doc["id"]] = (0, weights["rated_bad"], "rated_bad")
        elif _is_read(doc):
            labels[doc["id"]] = (1, weights["read"], "read")
        elif tags & {"favorite", "important"} or (
                doc.get("location") == "later" and doc.get("category") in ("rss", "email")):
            # `later` only counts for feed-born documents: a hand-saved article
            # lives there by definition, and calling every one of those a
            # positive would make the backlog pool impossible to rank.
            labels[doc["id"]] = (1, weights["saved"], "saved")
        elif doc["id"] in shortlisted:
            if opened:
                labels[doc["id"]] = (0, weights["bounced"], "bounced")
            elif doc["id"] in matured:
                labels[doc["id"]] = (0, weights["passed"], "passed")
        elif doc["id"] in exposure:
            seen = exposure[doc["id"]]
            saved = _parse(doc.get("saved_at"))
            if opened:
                labels[doc["id"]] = (0, weights["bounced"] * seen, "bounced")
            elif (doc.get("location") == "feed" and saved
                  and saved < now - timedelta(days=config.FEED_STALE_DAYS) and seen > 0):
                labels[doc["id"]] = (0, weights["ignored"] * seen, "ignored")
    return labels


def _matured_evictions(store):
    """Evictions that count as evidence he passed on something.

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
    return matured


def site(doc):
    """The publication, normalised just enough that 'the Guardian' is one site."""
    return (doc.get("site_name") or "").strip().lower()


def publication_vocabulary(docs):
    """Sites common enough among labelled docs to earn their own column."""
    counts = {}
    for doc in docs:
        name = site(doc)
        if name:
            counts[name] = counts.get(name, 0) + 1
    return sorted(n for n, c in counts.items() if c >= config.PUBLICATION_MIN_DOCS)


def features(doc, vector, publications=()):
    """embedding + a couple of shape features. No age, ever -- see module docstring.

    With a publication vocabulary, also a one-hot over it plus a trailing
    "other" column, so every document lights exactly one publication column.
    """
    word_count = doc.get("word_count") or 0
    extras = [np.log1p(word_count)]
    extras += [1.0 if doc.get("category") == c else 0.0 for c in CATEGORIES]
    if publications:
        name = site(doc)
        extras += [1.0 if name == p else 0.0 for p in publications]
        extras.append(0.0 if name in publications else 1.0)
    return np.concatenate([vector, np.asarray(extras, dtype=np.float32)])


def feature_count(publications=()):
    return config.EMBED_DIM + 1 + len(CATEGORIES) + (len(publications) + 1 if publications else 0)


def build_matrix(docs, embeddings, publications=()):
    """(X, kept_docs) for every doc that has an embedding."""
    rows, kept = [], []
    for doc in docs:
        vector = embeddings.get(doc["id"])
        if vector is None:
            continue
        rows.append(features(doc, vector, publications))
        kept.append(doc)
    if not rows:
        return np.zeros((0, feature_count(publications)), dtype=np.float32), []
    return np.vstack(rows), kept


def publications_of(model):
    """The vocabulary a model was fitted with; empty for one without the feature."""
    return list(getattr(model, "publications_", ()))


def train(store, now=None, scheme="current", publications=False, exclude=()):
    """Fit on everything labelled. Returns (model, stats) or (None, stats).

    `scheme` names the labelling scheme, `publications` switches the publication one-hot
    on, `exclude` holds doc ids to keep out of training (an evaluation set).
    The fitted vocabulary rides on the model as `publications_`, so scoring and
    attribution always build exactly the columns the model was fitted on.
    """
    labels = derive_labels(store, now=now, scheme=scheme)
    embeddings = store.embeddings(config.EMBED_KEY)
    docs = [d for d in store.documents() if d["id"] in labels and d["id"] not in exclude]
    vocabulary = publication_vocabulary(docs) if publications else []
    X, kept = build_matrix(docs, embeddings, vocabulary)
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
    model.publications_ = vocabulary
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
    X, kept = build_matrix(docs, embeddings, publications_of(model))
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
