#!/usr/bin/env python3
"""Offline side of the control-vs-candidate comparison (KYRO-30).

Each labelling scheme defines its own holdout, so the nightly holdout AUC of
one scheme cannot be compared with another's -- a scheme that adds thousands of
easy negatives scores higher without ranking any better. (That is what happened
when the October RSS flood arrived: AUC climbed from 0.74 to 0.81 while a
word-count baseline on the same split climbed from 0.57 to 0.77.)

So every ranker in config.RANKERS is scored here on the SAME two sets, both kept
out of every ranker's training data, and labelled by one neutral rule:

  shown  -- documents we put on the shortlist whose outcome is known. He
            definitely saw these, so a skip is a real skip. Small, and already
            filtered by a ranker, so it can only separate big differences.
  triage -- every 5th arrival block in which he organically opened >= half, i.e.
            periods when he was demonstrably working through the feed.

Neutral label: positive when rated good, favourited, or >= OPENED_WORDS
actually read; negative otherwise, with rate:bad overriding.

This is the offline smoke test. The online interleaved arms on the refinement
page are the real verdict. To try an idea: edit config.RANKERS["candidate"],
then run this file -- it prints the comparison without touching Readwise.

    .venv-rec/bin/python recommender_refine.py
"""

import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import recommender_model
from recommender_model import _parse, _tags

TRIAGE_EXPOSURE = 0.5
TRIAGE_EVERY = 5
BOOTSTRAPS = 200


def neutral_label(doc):
    tags = _tags(doc)
    if config.RATE_BAD_TAG in tags:
        return 0
    if config.RATE_GOOD_TAG in tags or tags & {"favorite", "important"}:
        return 1
    return int(recommender_model._words_read(doc) >= config.OPENED_WORDS
               or recommender_model._is_read(doc))


def eval_sets(store, documents=None):
    """({"shown": {id: y}, "triage": {id: y}}, ids to keep out of training)."""
    documents = documents if documents is not None else store.documents()
    events = store.events()
    shortlisted = {e["doc_id"] for e in events if e["action"] == "added"}
    evicted = {e["doc_id"] for e in events if e["action"] == "evicted"}
    by_id = {d["id"]: d for d in documents}
    rated = {config.RATE_GOOD_TAG, config.RATE_BAD_TAG}
    shown = {i: neutral_label(by_id[i]) for i in shortlisted
             if i in by_id and (i in evicted or _tags(by_id[i]) & rated)}

    triage_blocks = [b for b in recommender_model.arrival_blocks(documents)
                     if recommender_model.block_exposure(b, shortlisted) >= TRIAGE_EXPOSURE]
    held = triage_blocks[::TRIAGE_EVERY]
    triage = {d["id"]: neutral_label(d) for b in held for d in b if d["id"] not in shortlisted}
    excluded = set(shown) | {d["id"] for b in held for d in b}
    return {"shown": shown, "triage": triage}, excluded


def auc_with_interval(y, scores, seed=0):
    """Point AUC and a 90% bootstrap interval, or None when one class is missing."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if len(set(y.tolist())) < 2:
        return None
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(BOOTSTRAPS):
        k = rng.integers(0, len(y), len(y))
        if len(set(y[k].tolist())) == 2:
            boots.append(roc_auc_score(y[k], scores[k]))
    low, high = np.percentile(boots, [5, 95]) if boots else (None, None)
    return {"auc": float(roc_auc_score(y, scores)), "low": float(low), "high": float(high),
            "n": int(len(y)), "positives": int(y.sum())}


def score_set(model, labels, by_id, embeddings):
    docs = [by_id[i] for i in labels if i in by_id and i in embeddings]
    scores = recommender_model.score_documents(docs, embeddings, model)
    return [labels[d["id"]] for d in docs], [scores[d["id"]] for d in docs]


def top_fresh(store, models, documents, embeddings, k=10, now=None):
    """Each ranker's top k of the fresh feed, before cooldown and dedup."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=config.FEED_CANDIDATE_DAYS)
    pool = [d for d in documents
            if d.get("location") == "feed" and not d.get("first_opened_at")
            and d["id"] in embeddings and (_parse(d.get("saved_at")) or now) >= cutoff]
    out = {}
    for name, model in models.items():
        if model is None:
            continue
        scores = recommender_model.score_documents(pool, embeddings, model)
        ranked = sorted(pool, key=lambda d: -scores[d["id"]])[:k]
        out[name] = [{"id": d["id"], "title": (d.get("title") or d["id"])[:100],
                      "site_name": d.get("site_name"), "word_count": d.get("word_count"),
                      "url": d.get("url"), "score": scores[d["id"]]} for d in ranked]
    return out


def compare(store, rankers=None, now=None):
    """The refinement-page payload: common-set AUCs per ranker, plus top picks."""
    rankers = rankers or config.RANKERS
    documents = store.documents()
    by_id = {d["id"]: d for d in documents}
    embeddings = store.embeddings(config.EMBED_KEY)
    sets, excluded = eval_sets(store, documents)

    results, models = {}, {}
    for name, spec in rankers.items():
        model, stats = recommender_model.train(store, now=now, exclude=excluded, **spec)
        models[name] = model
        entry = {"spec": spec, "train_n": stats["usable"], "train_positives": stats["positives"],
                 "by_reason": stats["by_reason"]}
        for set_name, labels in sets.items():
            if model is None:
                entry[set_name] = None
                continue
            y, scores = score_set(model, labels, by_id, embeddings)
            entry[set_name] = auc_with_interval(y, scores)
        results[name] = entry

    baselines = {}
    for set_name, labels in sets.items():
        ids = [i for i in labels if i in by_id]
        baselines[set_name] = auc_with_interval(
            [labels[i] for i in ids], [by_id[i].get("word_count") or 0 for i in ids])
    return {"rankers": results, "word_count_baseline": baselines,
            "top_fresh": top_fresh(store, models, documents, embeddings, now=now)}


def main():
    from recommender_store import Store

    store = Store(sys.argv[1] if len(sys.argv) > 1 else config.RECOMMENDER_DB)
    result = compare(store)

    def show(a):
        return "--" if not a else f"{a['auc']:.3f} [{a['low']:.3f}-{a['high']:.3f}] n={a['n']}"

    for name, r in result["rankers"].items():
        print(f"{name:10} {r['spec']}  train n={r['train_n']} pos={r['train_positives']}")
        print(f"{'':10} shown  {show(r['shown'])}")
        print(f"{'':10} triage {show(r['triage'])}")
    for set_name, a in result["word_count_baseline"].items():
        print(f"word_count baseline {set_name}: {show(a)}")
    for name, top in result["top_fresh"].items():
        print(f"\ntop fresh by {name}:")
        for t in top:
            print(f"  {t['score']:.3f}  {(t['site_name'] or '?')[:18]:18}  {t['title'][:70]}")


if __name__ == "__main__":
    main()
