#!/usr/bin/env python3
"""The nightly shortlist cycle.

sync -> embed -> train -> select -> write.

The only thing this ever writes to Readwise is the `shortlist` tag, and it only
ever removes that tag from documents it added itself (tracked in the event log),
so a hand-shortlisted document is never touched. Run with --dry-run to see the
whole decision without writing anything.
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv(os.getenv("DOTENV_PATH") or None)

import config  # noqa: E402
import recommender_embed  # noqa: E402
import recommender_model  # noqa: E402
from readwise_client import ReadwiseClient  # noqa: E402
from recommender_store import Store  # noqa: E402


def log(message):
    print(f"[shortlist] {message}", flush=True)


def sync(store, client, full=False):
    """Pull documents Readwise has touched since the last run."""
    since = None if full else store.get_meta("last_sync")
    started = datetime.now(timezone.utc).isoformat()
    total = 0
    batch = []
    for doc in client.documents(updated_after=since):
        batch.append(doc)
        if len(batch) >= 200:
            total += store.upsert_documents(batch)
            batch = []
    if batch:
        total += store.upsert_documents(batch)
    store.set_meta("last_sync", started)
    log(f"synced {total} documents ({'full backfill' if full else f'since {since}'})")
    return total


def embed_missing(store, limit=None):
    pending = store.missing_embeddings(config.EMBED_KEY)
    if limit:
        pending = pending[:limit]
    if not pending:
        log("embeddings up to date")
        return 0
    log(f"embedding {len(pending)} documents")
    done = 0
    for start in range(0, len(pending), 200):
        chunk = pending[start : start + 200]
        vectors = recommender_embed.embed_documents(chunk)
        store.save_embeddings(config.EMBED_KEY, vectors)
        done += len(vectors)
        log(f"  {done}/{len(pending)}")
    dropped = store.forget_other_embeddings(config.EMBED_KEY)
    if dropped:
        log(f"dropped {dropped} vectors from a superseded text recipe")
    return done


def _parse(stamp):
    return recommender_model._parse(stamp)


def pick(store, model, taste, now=None, rng=None):
    """Choose this cycle's shortlist. Returns [(doc, slot, score), ...]."""
    now = now or datetime.now(timezone.utc)
    rng = rng or random.Random()
    embeddings = store.embeddings(config.EMBED_KEY)
    shown = store.last_shown()
    cooldown = now - timedelta(days=config.RESHOW_COOLDOWN_DAYS)

    def recently_shown(doc_id):
        cycle = shown.get(doc_id)
        return bool(cycle and _parse(cycle) and _parse(cycle) > cooldown)

    def eligible(doc):
        return (
            not recently_shown(doc["id"])
            and config.SHORTLIST_TAG not in recommender_model._tags(doc)
            and doc["id"] in embeddings
        )

    fresh_cutoff = now - timedelta(days=config.FEED_CANDIDATE_DAYS)
    feed = [
        d
        for d in store.documents(location="feed")
        if eligible(d)
        and not d.get("first_opened_at")
        and (_parse(d.get("saved_at")) or now) >= fresh_cutoff
    ]
    later = [d for d in store.documents(location="later") if eligible(d)]

    exploit_slots = (
        config.SHORTLIST_SIZE
        - config.SHORTLIST_RESURFACE_SLOTS
        - config.SHORTLIST_RANDOM_SLOTS
    )

    chosen = []
    taken = set()

    feed_scores = recommender_model.score_documents(feed, embeddings, model, taste)
    for doc in sorted(feed, key=lambda d: -feed_scores.get(d["id"], 0.0))[:exploit_slots]:
        chosen.append((doc, "exploit", feed_scores.get(doc["id"], 0.0)))
        taken.add(doc["id"])

    # The measurement slots: same pool as the exploit picks, drawn uniformly and
    # never scored. Ranked-vs-random open rate is only meaningful because both
    # come from this one pool.
    unranked = [d for d in feed if d["id"] not in taken]
    for doc in rng.sample(unranked, min(config.SHORTLIST_RANDOM_SLOTS, len(unranked))):
        chosen.append((doc, "random", None))
        taken.add(doc["id"])

    # Resurfacing: a random draw from the backlog, then ranked within that draw,
    # so the model never gets to comb the whole backlog for its own favourites.
    sample = rng.sample(later, min(config.RESURFACE_SAMPLE_SIZE, len(later)))
    if config.SHORTLIST_RESURFACE_SLOTS > 0 and sample:
        sample_scores = recommender_model.score_documents(sample, embeddings, model, taste)
        for doc in sorted(sample, key=lambda d: -sample_scores.get(d["id"], 0.0))[
            : config.SHORTLIST_RESURFACE_SLOTS
        ]:
            chosen.append((doc, "resurface", sample_scores.get(doc["id"], 0.0)))

    return chosen


def evictable(store):
    """Documents we shortlisted in an earlier cycle and have not yet evicted."""
    state = {}
    for event in store.events():
        state[event["doc_id"]] = event["action"]
    return [doc_id for doc_id, action in state.items() if action == "added"]


def run(dry_run=False, full_sync=False, skip_sync=False, embed_limit=None):
    store = Store(config.RECOMMENDER_DB)
    client = ReadwiseClient()
    cycle = datetime.now(timezone.utc).date().isoformat()

    if not skip_sync:
        client.check_auth()
        sync(store, client, full=full_sync)
    embed_missing(store, limit=embed_limit)

    model, stats = recommender_model.train(store)
    log(f"labels: {stats}")
    if model is None:
        taste = recommender_model.taste_vector(store)
        if taste is None:
            log("no labels and no taste vector -- nothing to rank with, stopping")
            return 1
        log("below the training threshold, ranking by taste-vector similarity")
    else:
        taste = None
        metrics = recommender_model.evaluate(store)
        log(f"holdout: {metrics}" if metrics else "holdout: not enough held-out signal yet")

    old = evictable(store)
    chosen = pick(store, model, taste)
    log(f"evicting {len(old)}, adding {len(chosen)}")
    for doc, slot, score in chosen:
        pretty = f"{score:.3f}" if score is not None else "  --  "
        log(f"  [{slot:9}] {pretty}  {(doc.get('title') or '')[:70]}")

    if dry_run:
        log("dry run -- no tags written")
        return 0

    updates = {}
    for doc_id in old:
        doc = store.document(doc_id)
        if not doc:
            continue
        tags = recommender_model._tags(doc) - {config.SHORTLIST_TAG}
        updates[doc_id] = sorted(tags)
    for doc, _, _ in chosen:
        updates[doc["id"]] = sorted(
            recommender_model._tags(doc) | {config.SHORTLIST_TAG}
        )
    if updates:
        client.bulk_set_tags(updates)
    for doc_id in old:
        store.log_event(doc_id, "evicted", cycle)
    for doc, slot, score in chosen:
        store.log_event(doc["id"], "added", cycle, slot=slot, score=score)
    log(f"wrote {len(updates)} tag updates")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="decide, but write nothing")
    parser.add_argument("--full-sync", action="store_true", help="re-pull every document")
    parser.add_argument("--skip-sync", action="store_true", help="use the local store as-is")
    parser.add_argument("--embed-limit", type=int, help="cap embeddings this run")
    args = parser.parse_args()
    return run(
        dry_run=args.dry_run,
        full_sync=args.full_sync,
        skip_sync=args.skip_sync,
        embed_limit=args.embed_limit,
    )


if __name__ == "__main__":
    sys.exit(main())
