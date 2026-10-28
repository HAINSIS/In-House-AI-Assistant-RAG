"""Exact-repeat answer cache: the same question, asked in the same scope, with
the same settings and index, is answered from SQLite with zero tokens.

The key includes everything that could change the answer or what the user
may see: the normalised question, the role's scope (filter + tools), provider
and model, a fingerprint of the answer-shaping settings, and the index
version. So a cached answer is never served across scopes (no access leak),
and editing a prompt or re-ingesting invalidates the cache by itself.
Only first-turn questions are cached: a follow-up depends on its history.
Paraphrases are not matched (a semantic cache can return a wrong answer for
a similar-looking question; not worth the risk here).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    key TEXT PRIMARY KEY, created TEXT NOT NULL, answer TEXT NOT NULL, extra TEXT NOT NULL
);
"""


def normalise(question: str) -> str:
    return " ".join(question.casefold().split())


def make_key(question: str, scope: dict, provider: str, model: str, fingerprint: str, index_version: str) -> str:
    payload = json.dumps([normalise(question), scope, provider, model, fingerprint, index_version],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint(settings: dict) -> str:
    """Hash of every setting that shapes an answer."""
    parts = {k: settings.get(k) for k in ("model", "retrieval", "tools")}
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class AnswerCache:
    def __init__(self, db: Path, ttl_hours: float):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.ttl = dt.timedelta(hours=ttl_hours)

    def get(self, key: str) -> tuple[str, dict] | None:
        row = self.conn.execute("SELECT created, answer, extra FROM answer_cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        if dt.datetime.now() - dt.datetime.fromisoformat(row[0]) > self.ttl:
            self.conn.execute("DELETE FROM answer_cache WHERE key=?", (key,))
            self.conn.commit()
            return None
        return row[1], json.loads(row[2])

    def put(self, key: str, answer: str, extra: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO answer_cache VALUES (?, ?, ?, ?)",
                          (key, dt.datetime.now().isoformat(timespec="seconds"), answer,
                           json.dumps(extra, ensure_ascii=False, default=str)))
        self.conn.commit()
