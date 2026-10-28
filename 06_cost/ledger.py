"""Per-user token ledger and daily budgets, in SQLite (data/inhouse.sqlite).

A budget is checked before any model call: a user over their role's daily
limit gets a refusal that costs nothing. Days are local calendar days.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_ledger (
    day TEXT NOT NULL, user TEXT NOT NULL, tokens INTEGER NOT NULL, cost_usd REAL,
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_ledger_day_user ON usage_ledger(day, user);
"""


class Ledger:
    def __init__(self, db: Path):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def spent_today(self, user: str) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(tokens), 0) FROM usage_ledger WHERE day=? AND user=?",
                                (dt.date.today().isoformat(), user)).fetchone()
        return int(row[0])

    def record(self, user: str, tokens: int, cost_usd: float | None) -> None:
        if tokens <= 0:
            return
        now = dt.datetime.now()
        self.conn.execute("INSERT INTO usage_ledger VALUES (?, ?, ?, ?, ?)",
                          (now.date().isoformat(), user, tokens, cost_usd, now.isoformat(timespec="seconds")))
        self.conn.commit()

    def over_budget(self, user: str, daily_tokens: int | None) -> tuple[bool, int]:
        """-> (over the limit?, tokens spent today). None = unlimited."""
        spent = self.spent_today(user)
        return (daily_tokens is not None and spent >= daily_tokens), spent
