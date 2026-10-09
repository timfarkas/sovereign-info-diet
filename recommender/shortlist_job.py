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
import re
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
    """Pull documents Readwise has touched since the last run, with their full text.

    The html is cut to plain text into the store's fulltext table and never kept
    on the document row.
    """
    since = None if full else store.get_meta("last_sync")
    started = datetime.now(timezone.utc).isoformat()
    total = 0
    batch = []

    def flush(batch):
        store.save_fulltext({d["id"]: recommender_embed.plain_text(d.pop("html_content", None))
                             for d in batch})
        return store.upsert_documents(batch)

    for doc in client.documents(updated_after=since, html=True):
        batch.append(doc)
        if len(batch) >= 200:
            total += flush(batch)
            batch = []
    if batch:
        total += flush(batch)
    store.set_meta("last_sync", started)
    log(f"synced {total} documents ({'full backfill' if full else f'since {since}'})")
    return total


def used_embeddings():
    """Every vector key some ranker fits on. bge is always one: dedup, the taste
    vector and the stats page work on it whatever the rankers use."""
    specs = list(config.RANKERS.values()) + list(config.OFFLINE_RANKERS.values())
    return {config.EMBED_KEY} | {s["embedding"] for s in specs if "embedding" in s}


def embed_missing(store, limit=None):
    """Embed whatever has no vector yet. Returns what it did, for the status page."""
    pending = store.missing_embeddings(config.EMBED_KEY)
    if limit:
        pending = pending[:limit]
    done = 0
    if pending:
        log(f"embedding {len(pending)} documents")
    for start in range(0, len(pending), 200):
        chunk = pending[start : start + 200]
        vectors = recommender_embed.embed_documents(chunk)
        store.save_embeddings(config.EMBED_KEY, vectors)
        done += len(vectors)
        log(f"  {done}/{len(pending)}")
    info = {"pending": len(pending), "embedded": done}
    if config.MISTRAL_EMBED_KEY in used_embeddings():
        info["fulltext"] = embed_fulltext(store, limit)
    keys = used_embeddings()
    unknown = keys - {config.EMBED_KEY, config.MISTRAL_EMBED_KEY}
    if unknown:
        raise ValueError(f"a ranker wants vectors nothing here embeds: {sorted(unknown)}")
    info["dropped_superseded"] = store.forget_other_embeddings(*keys)
    if info["dropped_superseded"]:
        log(f"dropped {info['dropped_superseded']} vectors from a superseded text recipe")
    if not pending and not (info.get("fulltext") or {}).get("pending"):
        log("embeddings up to date")
    return info


def embed_fulltext(store, limit=None):
    """mistral-embed whatever has no full-text vector yet. Costs money, so it
    refuses a run bigger than config.MISTRAL_MAX_WORDS_PER_RUN."""
    pending = store.missing_embeddings(config.MISTRAL_EMBED_KEY)
    if limit:
        pending = pending[:limit]
    if not pending:
        return {"pending": 0, "embedded": 0, "words": 0}
    bodies = store.fulltext(d["id"] for d in pending)
    words = sum(len(c.split()) for d in pending
                for c in recommender_embed.fulltext_chunks(d, bodies.get(d["id"])))
    if words > config.MISTRAL_MAX_WORDS_PER_RUN:
        raise RuntimeError(
            f"{len(pending)} documents / {words:,} words to full-text embed, over "
            f"MISTRAL_MAX_WORDS_PER_RUN={config.MISTRAL_MAX_WORDS_PER_RUN:,} -- a lost table or a "
            "key bump? cap it with --embed-limit or raise the limit on purpose")
    log(f"full-text embedding {len(pending)} documents ({words:,} words, "
        f"{sum(d['id'] in bodies for d in pending)} with a body)")
    done = 0
    for start in range(0, len(pending), 200):
        chunk = pending[start : start + 200]
        vectors = recommender_embed.mistral_embed_documents(chunk, bodies)
        store.save_embeddings(config.MISTRAL_EMBED_KEY, vectors)
        done += len(vectors)
        log(f"  {done}/{len(pending)}")
    return {"pending": len(pending), "embedded": done, "words": words}


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


def damped_sample(docs, k, pool, rng):
    """k of docs without replacement, a publication with n posts in `pool`
    drawing with total mass RANDOM_PUBLICATION_MASS(n) instead of n."""
    counts = {}
    for d in pool:
        counts[recommender_model.site(d)] = counts.get(recommender_model.site(d), 0) + 1
    left = list(docs)
    weights = [config.RANDOM_PUBLICATION_MASS(counts[recommender_model.site(d)])
               / counts[recommender_model.site(d)] for d in left]
    drawn = []
    while left and len(drawn) < k:
        i = rng.choices(range(len(left)), weights=weights)[0]
        drawn.append(left.pop(i))
        weights.pop(i)
    return drawn


def pick(store, model, taste, now=None, rng=None, trace=None, candidate=None, reinsert=(), skip=()):
    """Choose this cycle's shortlist. Returns [(doc, slot, score), ...].

    With a `candidate` model, each pool's ranked slots are split between it and
    `model` (the control) by team-draft interleaving: whichever ranker has fewer
    picks so far chooses next, ties broken by coin flip, each taking its own
    best document not already taken. Candidate picks get the slot `<pool>-cand`.
    Both rankers draw from the same pool on the same night, so their open rates
    are a like-for-like comparison -- the same argument as for the random arm.

    `reinsert` is a set of doc ids that skip the already-shortlisted and cooldown
    checks, so a same-day rerun can pick the current list again. Without it they
    would be treated as served, which is right for the nightly cycle.

    `skip` is a set of doc ids kept out of every pool (the privileged digests,
    which have their own slot).

    `trace`, if given, is filled in with the reasoning behind each pick --
    {doc_id: {rank, pool_size, short_reserve}} plus a "_pools" entry holding the
    candidate counts. It is an out-parameter rather than part of the return
    value so that the selection contract every test in test_shortlist.py asserts
    on stays exactly what it was.
    """
    now = now or datetime.now(timezone.utc)
    rng = rng or random.Random()
    trace = {} if trace is None else trace
    # bge vectors do the dedup whatever a ranker scores with; each ranker scores
    # with its own. A document needs all of them to be eligible, so neither
    # ranker can win a slot just because the other could not see it.
    embeddings = store.embeddings(config.EMBED_KEY)
    vectors = {config.EMBED_KEY: embeddings}
    for ranker in (model, candidate):
        if ranker is not None:
            key = recommender_model.embedding_of(ranker)
            vectors[key] = vectors.get(key) or store.embeddings(key)
    shown = store.last_shown()
    cooldown = now - timedelta(days=config.RESHOW_COOLDOWN_DAYS)

    def recently_shown(doc_id):
        cycle = shown.get(doc_id)
        return bool(cycle and _parse(cycle) and _parse(cycle) > cooldown)

    def eligible(doc):
        served = (recently_shown(doc["id"])
                  or config.SHORTLIST_TAG in recommender_model._tags(doc))
        return (doc["id"] not in skip
                and (not served or doc["id"] in reinsert)
                and all(doc["id"] in v for v in vectors.values()))

    fresh_cutoff = now - timedelta(days=config.FEED_CANDIDATE_DAYS)
    feed = [
        d
        for d in store.documents(location="feed")
        if eligible(d)
        and not d.get("first_opened_at")
        and (_parse(d.get("saved_at")) or now) >= fresh_cutoff
    ]
    later = [d for d in store.documents(location="later") if eligible(d)]

    def fill(pool, label, total, random_slots, short_slots, damp=False):
        """Rank the pool, take the top few, then draw the rest at random from it.

        The draw is uniform unless `damp`, which weights it against flooding
        publications (see config.RANDOM_PUBLICATION_MASS).

        Both arms come out of the same pool on purpose: that is what makes
        comparing their open rates a statement about the ranking. `short_slots`
        of the ranked picks are reserved for short documents, so the list always
        has something for a five-minute gap.
        """
        if not pool:
            return []
        teams = []
        for slot, ranker, fallback in ((label, model, taste),
                                       (f"{label}-cand", candidate, None)):
            if ranker is None and fallback is None:
                continue
            own = vectors[recommender_model.embedding_of(ranker)] if ranker is not None else embeddings
            scores = recommender_model.score_documents(pool, own, ranker, fallback, now=now)
            ranked = sorted(pool, key=lambda d: -scores.get(d["id"], 0.0))
            teams.append({"slot": slot, "scores": scores, "ranked": ranked,
                          "rank_of": {d["id"]: i + 1 for i, d in enumerate(ranked)}})
        ranked_slots = total - random_slots

        def next_choice(team, picks, only_short):
            for doc in team["ranked"]:
                if only_short and (doc.get("word_count") or 0) >= config.SHORT_WORDS:
                    continue
                if doc["id"] in {d["id"] for d, _, _ in picks}:
                    continue
                if _duplicates(doc, [d for d, _, _ in picks], embeddings):
                    continue
                return doc
            return None

        def take(limit, picks, short_reserve=False):
            exhausted = set()
            while len(picks) < limit and len(exhausted) < len(teams):
                live = [t for t in teams if t["slot"] not in exhausted]
                count = {t["slot"]: sum(1 for _, s, _ in picks if s == t["slot"]) for t in live}
                fewest = min(count.values())
                tied = [t for t in live if count[t["slot"]] == fewest]
                team = tied[0] if len(tied) == 1 else rng.choice(tied)
                doc = next_choice(team, picks, short_reserve)
                if doc is None:
                    exhausted.add(team["slot"])
                    continue
                picks.append((doc, team["slot"], team["scores"].get(doc["id"], 0.0)))
                trace[doc["id"]] = {
                    "rank": team["rank_of"].get(doc["id"]),
                    "pool_size": len(pool),
                    "short_reserve": short_reserve,
                }
            return picks

        picks = take(min(short_slots, ranked_slots), [], short_reserve=True)
        picks = take(ranked_slots, picks)
        taken = {d["id"] for d, _, _ in picks}
        rest = [d for d in pool if d["id"] not in taken]
        drawn = (damped_sample(rest, random_slots, pool, rng) if damp
                 else rng.sample(rest, min(random_slots, len(rest))))
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
        feed, "feed", config.FEED_SLOTS, config.FEED_RANDOM_SLOTS, config.FEED_SHORT_SLOTS,
        damp=True,
    ) + fill(
        backlog,
        "later",
        config.LATER_SLOTS,
        config.LATER_RANDOM_SLOTS,
        config.LATER_SHORT_SLOTS,
    )


def is_digest(doc):
    return (doc.get("category") == config.DIGEST_CATEGORY
            and doc.get("author") == config.DIGEST_AUTHOR)


def privileged_digests(store):
    """The newest digest of each series, unless it is already archived.

    A series is the title without its trailing date ("AI Digest - 2026-10-08" ->
    "AI Digest"). A digest stays privileged until Tim archives it or a newer one
    of its series arrives, whichever comes first.
    """
    newest = {}
    for doc in store.documents():
        if not is_digest(doc):
            continue
        series = re.sub(r"\s*-\s*\d{4}-\d{2}-\d{2}\s*$", "", doc.get("title") or "")
        if series not in newest or doc["saved_at"] > newest[series]["saved_at"]:
            newest[series] = doc
    return [d for d in newest.values() if d["location"] != "archive"]


def tag_digests(store, client, dry_run=False):
    """Tag every digest rate:good in Reader and in the local store. Returns the count."""
    updates = {}
    for doc in store.documents():
        tags = recommender_model._tags(doc)
        if is_digest(doc) and config.RATE_GOOD_TAG not in tags:
            updates[doc["id"]] = sorted(tags | {config.RATE_GOOD_TAG})
    if updates and not dry_run:
        client.bulk_set_tags(updates)
        for doc_id, tags in updates.items():
            store.set_tags(doc_id, tags)
    return len(updates)


def evictable(store):
    """Documents we shortlisted in an earlier cycle and have not yet evicted."""
    state = {}
    for event in store.events():
        state[event["doc_id"]] = event["action"]
    return [doc_id for doc_id, action in state.items() if action == "added"]


def run(dry_run=False, full_sync=False, skip_sync=False, embed_limit=None, rerun=False):
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

    tagged = tag_digests(store, client, dry_run=dry_run)
    if tagged:
        log(f"tagged {tagged} digests {config.RATE_GOOD_TAG}")
    digests = privileged_digests(store)
    digest_ids = {d["id"] for d in digests}

    clock = time.monotonic()
    model, stats = recommender_model.train(store, **config.RANKERS["control"])
    log(f"labels: {stats}")
    candidate, candidate_stats = recommender_model.train(store, **config.RANKERS["candidate"])
    log(f"candidate labels: {candidate_stats}")
    if model is None:
        # Interleaving a model against the cold-start taste vector would be
        # measuring the cold start, not the candidate.
        candidate = None
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

    # Privileged digests stay on the list across nights, so they are neither
    # evicted nor re-added while they are still privileged.
    held = [d for d in evictable(store) if d in digest_ids]
    old = [d for d in evictable(store) if d not in digest_ids]
    new_digests = [d for d in digests if d["id"] not in held]
    trace = {}
    chosen = pick(store, model, taste, trace=trace, candidate=candidate,
                  reinsert=set(old) if rerun else (), skip=digest_ids)
    log(f"evicting {len(old)}, adding {len(chosen)} + {len(new_digests)} digests "
        f"({len(held)} digests kept)")
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
            import sys
            from pathlib import Path
            # html_status/ is a repo-root sibling of this folder, not a
            # dependency installed anywhere on sys.path -- add the root once,
            # here, rather than assuming whoever invoked this script already
            # did. shortlist_stats itself lives right here in recommender/, so
            # it needs no path help -- same-directory import, same as config.
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from html_status import stats_page, stats_store
            import recommender_refine
            import shortlist_stats
            # Offline control-vs-candidate comparison on common held-out sets;
            # trains both rankers once more with those sets excluded (~15 s).
            refine = None
            if model is not None:
                try:
                    refine = recommender_refine.compare(store)
                except Exception:
                    # The page then says "not measured"; the rest of the row stands.
                    import traceback
                    log("refine: offline comparison failed:")
                    traceback.print_exc()
            row = shortlist_stats.build(
                store, cycle=cycle, dry_run=dry_run, chosen=chosen, trace=trace,
                evicted_ids=old, label_stats=stats, holdout=metrics,
                trained=model is not None, model=model, sync_info=sync_info,
                candidate=candidate, candidate_stats=candidate_stats, refine=refine,
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
    for doc in new_digests:
        updates[doc["id"]] = sorted(
            recommender_model._tags(doc) | {config.SHORTLIST_TAG}
        )
    if updates:
        client.bulk_set_tags(updates)
    for doc_id in old:
        store.log_event(doc_id, "evicted", cycle)
    for doc, slot, score in chosen:
        store.log_event(doc["id"], "added", cycle, slot=slot, score=score)
    for doc in new_digests:
        store.log_event(doc["id"], "added", cycle, slot=config.DIGEST_SLOT)
    log(f"wrote {len(updates)} tag updates")
    publish()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="decide, but write nothing")
    parser.add_argument("--full-sync", action="store_true", help="re-pull every document")
    parser.add_argument("--skip-sync", action="store_true", help="use the local store as-is")
    parser.add_argument("--rerun", action="store_true",
                        help="put the current shortlist back in the pool instead of treating it as served")
    parser.add_argument("--embed-limit", type=int, help="cap embeddings this run")
    args = parser.parse_args()
    return run(
        dry_run=args.dry_run,
        full_sync=args.full_sync,
        skip_sync=args.skip_sync,
        embed_limit=args.embed_limit,
        rerun=args.rerun,
    )


if __name__ == "__main__":
    sys.exit(main())
