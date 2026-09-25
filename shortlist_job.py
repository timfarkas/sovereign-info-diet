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
import time
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
    """Embed whatever has no vector yet. Returns what it did, for the status page."""
    pending = store.missing_embeddings(config.EMBED_KEY)
    if limit:
        pending = pending[:limit]
    if not pending:
        log("embeddings up to date")
        return {"pending": 0, "embedded": 0, "dropped_superseded": 0}
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
    return {"pending": len(pending), "embedded": done, "dropped_superseded": dropped}


def _parse(stamp):
    return recommender_model._parse(stamp)


def _duplicates(candidate, already, embeddings):
    """Is this the same thing as something already picked?

    Vectors are L2-normalized, so the dot product is the cosine.
    """
    vector = embeddings.get(candidate["id"])
    if vector is None:
        return False
    for other in already:
        past = embeddings.get(other["id"])
        if past is not None and float(vector @ past) >= config.DEDUP_SIMILARITY:
            return True
    return False


def pick(store, model, taste, now=None, rng=None, trace=None):
    """Choose this cycle's shortlist. Returns [(doc, slot, score), ...].

    `trace`, if given, is filled in with the reasoning behind each pick --
    {doc_id: {rank, pool_size, short_reserve}} plus a "_pools" entry holding the
    candidate counts. It is an out-parameter rather than part of the return
    value so that the selection contract every test in test_shortlist.py asserts
    on stays exactly what it was.
    """
    now = now or datetime.now(timezone.utc)
    rng = rng or random.Random()
    trace = {} if trace is None else trace
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

    def fill(pool, label, total, random_slots, short_slots):
        """Rank the pool, take the top few, then draw the rest uniformly from it.

        Both arms come out of the same pool on purpose: that is what makes
        comparing their open rates a statement about the ranking. `short_slots`
        of the ranked picks are reserved for short documents, so the list always
        has something for a five-minute gap.
        """
        if not pool:
            return []
        scores = recommender_model.score_documents(pool, embeddings, model, taste)
        ranked = sorted(pool, key=lambda d: -scores.get(d["id"], 0.0))
        ranked_slots = total - random_slots
        rank_of = {d["id"]: i + 1 for i, d in enumerate(ranked)}

        def take(candidates, limit, picks, short_reserve=False):
            for candidate in candidates:
                if len(picks) >= limit:
                    break
                if candidate["id"] in {d["id"] for d, _, _ in picks}:
                    continue
                if _duplicates(candidate, [d for d, _, _ in picks], embeddings):
                    continue
                picks.append((candidate, label, scores.get(candidate["id"], 0.0)))
                trace[candidate["id"]] = {
                    "rank": rank_of.get(candidate["id"]),
                    "pool_size": len(pool),
                    "short_reserve": short_reserve,
                }
            return picks

        short = [d for d in ranked if (d.get("word_count") or 0) < config.SHORT_WORDS]
        picks = take(short, min(short_slots, ranked_slots), [], short_reserve=True)
        picks = take(ranked, ranked_slots, picks)
        taken = {d["id"] for d, _, _ in picks}
        rest = [d for d in pool if d["id"] not in taken]
        drawn = rng.sample(rest, min(random_slots, len(rest)))
        for d in drawn:
            trace[d["id"]] = {"rank": None, "pool_size": len(pool), "short_reserve": False}
        picks += [(d, f"{label}-random", None) for d in drawn]
        return picks

    # The backlog is sampled before it is ranked, so the model never gets to comb
    # all of `later` for its own favourites -- it only ranks within a random draw.
    backlog = rng.sample(later, min(config.RESURFACE_SAMPLE_SIZE, len(later)))
    trace["_pools"] = {
        "feed_candidates": len(feed),
        "later_candidates": len(later),
        "backlog_sampled": len(backlog),
        "embedded": len(embeddings),
    }

    return fill(
        feed, "feed", config.FEED_SLOTS, config.FEED_RANDOM_SLOTS, config.FEED_SHORT_SLOTS
    ) + fill(
        backlog,
        "later",
        config.LATER_SLOTS,
        config.LATER_RANDOM_SLOTS,
        config.LATER_SHORT_SLOTS,
    )


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
    started = time.monotonic()

    # Read before sync() overwrites it: this is the boundary the overnight
    # signal counts are measured against.
    previous_sync = store.get_meta("last_sync")
    clock = time.monotonic()
    synced = 0
    if not skip_sync:
        client.check_auth()
        synced = sync(store, client, full=full_sync)
    sync_info = {"documents": synced, "since": previous_sync, "full": full_sync,
                 "skipped": skip_sync, "duration_s": time.monotonic() - clock}

    clock = time.monotonic()
    embed_info = embed_missing(store, limit=embed_limit)
    embed_info["duration_s"] = time.monotonic() - clock

    clock = time.monotonic()
    model, stats = recommender_model.train(store)
    log(f"labels: {stats}")
    metrics = None
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
    train_duration = time.monotonic() - clock

    old = evictable(store)
    trace = {}
    chosen = pick(store, model, taste, trace=trace)
    log(f"evicting {len(old)}, adding {len(chosen)}")
    for doc, slot, score in chosen:
        pretty = f"{score:.3f}" if score is not None else "  --  "
        log(f"  [{slot:9}] {pretty}  {(doc.get('title') or '')[:70]}")

    def publish():
        """Record this run and re-render the status page.

        Wrapped: a bug in the stats layer must not be able to undo a cycle whose
        tags are already written to Readwise. It prints the traceback instead of
        swallowing it, so a broken page is loud in the log.
        """
        try:
            from html_status import stats_page, stats_store, shortlist_stats
            row = shortlist_stats.build(
                store, cycle=cycle, dry_run=dry_run, chosen=chosen, trace=trace,
                evicted_ids=old, label_stats=stats, holdout=metrics,
                trained=model is not None, model=model, sync_info=sync_info,
                embed_info=embed_info,
                durations={"train_duration_s": train_duration,
                           "duration_s": time.monotonic() - started},
                problems=([] if metrics else
                          ["not enough held-out signal to score the model tonight"]),
            )
            stats_store.record("shortlist", row)          # unconditional: cheap history
            written = stats_page.render_recommender_page()  # opt-in: needs HTML_SERVE_DIR
            if written:
                log(f"stats: wrote {written}")
            else:
                log("stats: recorded to data/run_stats -- page rendering is off "
                    "(set HTML_SERVE_DIR in .env to turn it on)")
        except Exception:
            import traceback
            log("stats: status page failed -- the cycle itself is unaffected:")
            traceback.print_exc()

    if dry_run:
        log("dry run -- no tags written")
        publish()
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
    publish()
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
