#!/usr/bin/env python3
"""SQLite store for the shortlist recommender.

One file, three tables: the documents as Readwise last reported them, their
embeddings, and the log of what we shortlisted and what became of it. Labels are
NOT stored -- they are derived from these tables on every run, so changing the
labelling rules never leaves stale labels behind.
"""

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT,
    author TEXT,
    source TEXT,
    site_name TEXT,
    category TEXT,
    location TEXT,
    summary TEXT,
    url TEXT,
    source_url TEXT,
    word_count INTEGER,
    published_date TEXT,
    saved_at TEXT,
    updated_at TEXT,
    last_moved_at TEXT,
    first_opened_at TEXT,
    last_opened_at TEXT,
    reading_progress REAL,
    tags TEXT,
    synced_at TEXT
);

CREATE TABLE IF NOT EXISTS embeddings (
    doc_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    embedded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shortlist_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    action TEXT NOT NULL,          -- 'added' | 'evicted'
    slot TEXT,                     -- 'exploit' | 'resurface' | 'random'
    score REAL,
    cycle TEXT NOT NULL,           -- ISO date of the run
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_location ON documents(location);
CREATE INDEX IF NOT EXISTS idx_events_doc ON shortlist_events(doc_id);
"""

_FIELDS = [
    "id", "title", "author", "source", "site_name", "category", "location",
    "summary", "url", "source_url", "word_count", "published_date", "saved_at",
    "updated_at", "last_moved_at", "first_opened_at", "last_opened_at",
    "reading_progress",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # -- documents -----------------------------------------------------------

    def upsert_documents(self, documents):
        rows = []
        for doc in documents:
            tags = doc.get("tags") or {}
            names = sorted(tags.keys() if isinstance(tags, dict) else tags)
            rows.append(
                [doc.get(f) for f in _FIELDS] + [json.dumps(names), now_iso()]
            )
        placeholders = ",".join("?" * (len(_FIELDS) + 2))
        columns = ",".join(_FIELDS + ["tags", "synced_at"])
        self.conn.executemany(
            f"INSERT OR REPLACE INTO documents ({columns}) VALUES ({placeholders})",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def documents(self, location=None, where=None):
        sql = "SELECT * FROM documents"
        clauses, params = [], []
        if location:
            clauses.append("location = ?")
            params.append(location)
        if where:
            clauses.append(where)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return [dict(r) for r in self.conn.execute(sql, params)]

    def document(self, doc_id):
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    # -- embeddings ----------------------------------------------------------

    def missing_embeddings(self, model):
        sql = """
            SELECT d.* FROM documents d
            LEFT JOIN embeddings e ON e.doc_id = d.id AND e.model = ?
            WHERE e.doc_id IS NULL
        """
        return [dict(r) for r in self.conn.execute(sql, (model,))]

    def save_embeddings(self, model, vectors):
        """vectors: {doc_id: np.ndarray(float32)}"""
        stamp = now_iso()
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (doc_id, model, dim, vec, embedded_at)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (doc_id, model, len(vec), np.asarray(vec, dtype=np.float32).tobytes(), stamp)
                for doc_id, vec in vectors.items()
            ],
        )
        self.conn.commit()

    def embeddings(self, model):
        out = {}
        for row in self.conn.execute(
            "SELECT doc_id, vec FROM embeddings WHERE model = ?", (model,)
        ):
            out[row["doc_id"]] = np.frombuffer(row["vec"], dtype=np.float32)
        return out

    # -- shortlist log -------------------------------------------------------

    def log_event(self, doc_id, action, cycle, slot=None, score=None):
        self.conn.execute(
            "INSERT INTO shortlist_events (doc_id, action, slot, score, cycle, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, action, slot, score, cycle, now_iso()),
        )
        self.conn.commit()

    def events(self, doc_id=None):
        sql = "SELECT * FROM shortlist_events"
        params = []
        if doc_id:
            sql += " WHERE doc_id = ?"
            params.append(doc_id)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY id", params)]

    def last_shown(self):
        """{doc_id: cycle} for the most recent cycle each doc was shortlisted in."""
        sql = """
            SELECT doc_id, MAX(cycle) AS cycle FROM shortlist_events
            WHERE action = 'added' GROUP BY doc_id
        """
        return {r["doc_id"]: r["cycle"] for r in self.conn.execute(sql)}

    # -- meta ----------------------------------------------------------------

    def get_meta(self, key, default=None):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
        )
        self.conn.commit()
