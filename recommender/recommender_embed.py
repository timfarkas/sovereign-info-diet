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
    (The candidate ranker's publication one-hot is not that: it is an indicator
    per site, and its weight is fitted like any other -- no label goes into the
    feature value itself.)

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


# -- full text, mistral-embed (KYRO-30) ----------------------------------------
#
# Unlike bge above, this one leaves the box: document text goes to Mistral's API.
# The recipe must match the one the stored mistral vectors were made with
# (config.MISTRAL_EMBED_KEY), or old and new vectors live in different spaces.

import re  # noqa: E402
import time  # noqa: E402
from html.parser import HTMLParser  # noqa: E402

import requests  # noqa: E402


class _Text(HTMLParser):
    SKIP = {"script", "style"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in self.SKIP

    def handle_endtag(self, tag):
        self.skip -= tag in self.SKIP and self.skip > 0

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


# Footer markers only -- "sponsor" blocks sit mid-newsletter and are content-ish.
_FOOTER = re.compile(r"unsubscribe|manage (your )?(subscription|preferences)|you('re| are) receiving|"
                     r"update your preferences|forwarded this|sent to [\w.+-]+@", re.I)


def plain_text(html):
    """html_content -> whitespace-normalised words, newsletter footer cut.

    A footer marker only counts in the second half, so a newsletter that opens
    with "you're receiving this because..." keeps its body.
    """
    parser = _Text()
    parser.feed(html or "")
    text = " ".join(" ".join(parser.out).split())
    m = next((m for m in _FOOTER.finditer(text) if m.start() >= len(text) // 2), None)
    return text[: m.start()] if m else text


def fulltext_chunks(doc, body):
    """The pieces one document is embedded as: header + body, CHUNK_WORDS at a time."""
    words = (document_text(doc) + "\n" + (body or "")).split()
    out = []
    for start in range(0, len(words), config.MISTRAL_CHUNK_WORDS):
        chunk = " ".join(words[start : start + config.MISTRAL_CHUNK_WORDS])
        out += [chunk[j : j + config.MISTRAL_CHUNK_CHARS]
                for j in range(0, len(chunk), config.MISTRAL_CHUNK_CHARS)]
    return out or [doc["id"]]


def mistral_request(texts, session=None):
    """One embeddings call. Retries rate limits and 5xx; anything else raises."""
    session = session or requests.Session()
    headers = {"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"}
    for attempt in range(8):
        r = session.post("https://api.mistral.ai/v1/embeddings", headers=headers,
                         json={"model": "mistral-embed", "input": texts}, timeout=120)
        if r.status_code == 200:
            return [d["embedding"] for d in r.json()["data"]]
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"mistral {r.status_code}: {r.text[:300]}")
    raise RuntimeError("mistral: retries exhausted")


def mistral_embed_documents(documents, bodies, request=mistral_request):
    """{doc_id: vector}: word-weighted mean of each document's chunk vectors, L2-normed.

    A document's chunks may land in different requests; they are averaged here.
    """
    items = [(d["id"], c) for d in documents for c in fulltext_chunks(d, bodies.get(d["id"]))]
    parts = {}
    batch, words = [], 0
    for n, (doc_id, chunk) in enumerate(items):
        batch.append((doc_id, chunk))
        words += len(chunk.split())
        last = n == len(items) - 1
        if (last or len(batch) >= config.MISTRAL_BATCH_CHUNKS
                or words + len(items[n + 1][1].split()) > config.MISTRAL_BATCH_WORDS):
            vectors = request([c for _, c in batch])
            for (i, c), v in zip(batch, vectors):
                parts.setdefault(i, []).append((len(c.split()), np.asarray(v, dtype=np.float32)))
            batch, words = [], 0
    out = {}
    for doc_id, pieces in parts.items():
        weights = np.array([max(w, 1) for w, _ in pieces], dtype=np.float32)
        mean = (np.vstack([v for _, v in pieces]) * weights[:, None]).sum(0)
        out[doc_id] = (mean / np.linalg.norm(mean)).astype(np.float32)
    return out
