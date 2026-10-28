"""One trace row per answered question, in SQLite (data/inhouse.sqlite).

Everything needed to explain an answer after the fact: who asked, which path
answered it, what was retrieved, which tools ran with which arguments, the
model's request id, tokens, cost, latency, and errors. Local only: the
table holds questions and answers about the private library.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from pathlib import Path

COLUMNS = [
    ("id", "TEXT PRIMARY KEY"), ("ts", "TEXT"), ("source", "TEXT"), ("user", "TEXT"), ("role", "TEXT"),
    ("question", "TEXT"), ("path", "TEXT"), ("provider", "TEXT"), ("model", "TEXT"),
    ("cache_hit", "INTEGER"), ("budget_blocked", "INTEGER"), ("skipped_model", "INTEGER"),
    ("top_similarity", "REAL"), ("retrieved", "TEXT"), ("tool_calls", "TEXT"), ("model_turns", "INTEGER"),
    ("input_tokens", "INTEGER"), ("cached_tokens", "INTEGER"), ("cache_write_tokens", "INTEGER"),
    ("output_tokens", "INTEGER"), ("cost_usd", "REAL"), ("ref_cost_usd", "REAL"),
    ("latency_ms", "INTEGER"), ("stop_reason", "TEXT"), ("request_id", "TEXT"), ("error", "TEXT"),
    ("answer", "TEXT"),
]


class Tracer:
    def __init__(self, db: Path):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db), check_same_thread=False)
        cols = ", ".join(f"{n} {t}" for n, t in COLUMNS)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS traces ({cols})")
        self.conn.execute("CREATE INDEX IF NOT EXISTS traces_ts ON traces(ts)")
        self.conn.commit()

    def record(self, **fields) -> str:
        fields.setdefault("id", uuid.uuid4().hex[:12])
        fields.setdefault("ts", dt.datetime.now().isoformat(timespec="seconds"))
        for key in ("retrieved", "tool_calls"):
            if fields.get(key) is not None and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], ensure_ascii=False, default=str)
        names = [n for n, _ in COLUMNS if n in fields]
        self.conn.execute(f"INSERT INTO traces ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                          [fields[n] for n in names])
        self.conn.commit()
        return fields["id"]
