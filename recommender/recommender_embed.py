#!/usr/bin/env python3
"""Local text embeddings for the shortlist recommender.

Runs bge-small (quantized ONNX) on the CPU through fastembed. Nothing leaves the
VPS -- that is the point of doing it this way rather than calling an API.

Memory is the real constraint here, not speed: this box has 3.7 GB total and
already runs the 01:00 digest cron, and onnxruntime with default thread and
batch settings OOM'd it once (2026-09-23). So the defaults below are deliberately
small, and the model is loaded lazily so importing this module costs nothing.
"""

import os

import numpy as np

import config

_model = None


def _load():
    global _model
    if _model is None:
        os.environ.setdefault("OMP_NUM_THREADS", str(config.EMBED_THREADS))
        from fastembed import TextEmbedding

        _model = TextEmbedding(
            model_name=config.EMBED_MODEL,
            threads=config.EMBED_THREADS,
            cache_dir=config.EMBED_CACHE_DIR,
        )
    return _model


def document_text(doc):
    """What we actually embed: title, who wrote it, where it came from, summary.

    Author and site name go in on purpose -- a handful of authors and
    publications recur constantly in this corpus and carry a lot of the signal,
    and putting them in the text lets the linear head learn that without a
    separate per-source feature computed from the labels, which would leak.

    Changing anything here means bumping config.EMBED_TEXT_VERSION.
    """
    parts = [
        doc.get("title") or "",
        doc.get("author") or "",
        doc.get("site_name") or doc.get("source") or "",
        (doc.get("summary") or "")[: config.EMBED_SUMMARY_CHARS],
    ]
    return "\n".join(p for p in parts if p).strip()


def embed_texts(texts, batch_size=None):
    """Embed in small batches. Returns a float32 array, one row per text."""
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    model = _load()
    batch = batch_size or config.EMBED_BATCH_SIZE
    vectors = []
    for start in range(0, len(texts), batch):
        vectors.extend(model.embed(texts[start : start + batch], batch_size=batch))
    return np.asarray(vectors, dtype=np.float32)


def embed_documents(documents, batch_size=None):
    """{doc_id: vector} for documents that have any text worth embedding."""
    usable = [(d["id"], document_text(d)) for d in documents]
    usable = [(doc_id, text) for doc_id, text in usable if text]
    if not usable:
        return {}
    vectors = embed_texts([text for _, text in usable], batch_size=batch_size)
    return {doc_id: vectors[i] for i, (doc_id, _) in enumerate(usable)}
