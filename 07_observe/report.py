"""Read the traces: totals, breakdowns, and the full story of one answer.

    python 07_observe/report.py                    # everything
    python 07_observe/report.py --since 2026-10-28 --source cli
    python 07_observe/report.py --trace <id>       # one request, end to end
    python 07_observe/report.py --last 5           # the latest requests, one line each
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_shared"))
from settings import load_settings  # noqa: E402


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def table(rows: list[list], header: list[str]) -> str:
    cells = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in cells[1:]])


def breakdown(conn, where: str, args: list, column: str) -> str:
    rows = []
    for key, in conn.execute(f"SELECT DISTINCT {column} FROM traces WHERE {where} ORDER BY 1", args):
        r = conn.execute(f"""SELECT COUNT(*), SUM(cache_hit), SUM(skipped_model), SUM(budget_blocked),
                                    SUM(error IS NOT NULL), SUM(input_tokens + cached_tokens), SUM(output_tokens),
                                    SUM(cached_tokens), SUM(ref_cost_usd)
                             FROM traces WHERE {where} AND {column} IS ?""", args + [key]).fetchone()
        lat = [x for x, in conn.execute(f"SELECT latency_ms FROM traces WHERE {where} AND {column} IS ? "
                                        "AND error IS NULL", args + [key])]
        n, hits, skipped, blocked, errors, tin, tout, cached, ref = (v or 0 for v in r)
        rows.append([key, n, hits, skipped, blocked, errors, f"{tin:,}", f"{cached / tin:.0%}" if tin else "-",
                     f"{tout:,}", f"{(tin + tout) / n:,.0f}", f"${ref:.3f}",
                     f"{pct(lat, .5) / 1000:.1f}s", f"{pct(lat, .95) / 1000:.1f}s"])
    return table(rows, [column, "requests", "cache hits", "no-model", "budget", "errors", "tokens in",
                        "cached %", "tokens out", "tokens/req", "≈ on ref model", "p50", "p95"])


def show_trace(conn, trace_id: str) -> None:
    conn.row_factory = sqlite3.Row
    r = conn.execute("SELECT * FROM traces WHERE id=?", (trace_id,)).fetchone()
    if not r:
        print(f"no trace {trace_id}")
        return
    for key in r.keys():
        value = r[key]
        if key in ("retrieved", "tool_calls") and value:
            value = json.dumps(json.loads(value), indent=2, ensure_ascii=False)
        print(f"{key:>18}: {value}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="ISO date, e.g. 2026-10-28")
    ap.add_argument("--source", choices=["cli", "eval", "mcp"])
    ap.add_argument("--trace", help="show one trace in full")
    ap.add_argument("--last", type=int, help="list the latest N requests")
    args = ap.parse_args()

    db = ROOT / load_settings()["observe"]["db"]
    if not db.exists():
        print("no traces yet")
        return 0
    conn = sqlite3.connect(str(db))
    if args.trace:
        show_trace(conn, args.trace)
        return 0

    where, params = "1=1", []
    if args.since:
        where += " AND ts >= ?"
        params.append(args.since)
    if args.source:
        where += " AND source = ?"
        params.append(args.source)

    if args.last:
        rows = conn.execute(f"""SELECT id, ts, user, path, model, input_tokens + cached_tokens, output_tokens,
                                       latency_ms, COALESCE(error, substr(replace(answer, char(10), ' '), 1, 60))
                                FROM traces WHERE {where} ORDER BY ts DESC LIMIT ?""", params + [args.last]).fetchall()
        print(table([list(r) for r in rows], ["id", "ts", "user", "path", "model", "in", "out", "ms", "answer / error"]))
        return 0

    n, first, last = conn.execute(f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM traces WHERE {where}", params).fetchone()
    print(f"{n} requests, {first} to {last}\n")
    for column in ("source", "path", "role", "model"):
        print(breakdown(conn, where, params, column) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
