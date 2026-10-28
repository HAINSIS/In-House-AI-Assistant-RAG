"""Run the eval set through the app's pipeline and score it.

    python evals/run_eval.py --split dev                 # tune on this
    python evals/run_eval.py --split test                # once per layer
    python evals/run_eval.py --split dev --provider zhipu --limit 5

Resumable: results go to results/<date>_<layer>_<provider>_<split>.jsonl, one
row per question; re-running the same command skips questions that already
have an answer and retries the ones that errored (free tiers return 429s).
Throttled: waits `min_interval_s` (provider setting, or --min-interval)
between calls. At the end prints and appends a summary line to results/summary.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "app")]

import yaml  # noqa: E402

import _pathsetup  # noqa: E402,F401
import pipeline  # noqa: E402
from llm import LLMError, make_client  # noqa: E402
from sample_cards import papers_dir  # noqa: E402
from score import Library, score  # noqa: E402
from settings import load_secrets, load_settings, pick_provider  # noqa: E402

KINDS = ("lookup", "multi", "count", "unanswerable", "forbidden")


def load_library(settings: dict) -> Library:
    c = settings["corpus"]
    root = HERE.parent / c["root"]
    return Library.load(papers_dir(), (root / c["researches_dir"]).resolve() if c.get("researches_dir") else None)


def stored_count(row: dict) -> int | None:
    """The expected count recorded when a count row was scored (older rows
    only have it inside passed_detail, "expected N")."""
    if row.get("kind") != "count":
        return None
    if row.get("expected_count") is not None:
        return row["expected_count"]
    m = re.match(r"expected (\d+)", row.get("passed_detail") or "")
    return int(m.group(1)) if m else None


def ref_price(settings: dict) -> dict | None:
    c = settings["cost"]
    return c["prices"].get(c["reference_model"]) if c.get("reference_model") else None


def scope_of(item: dict, settings: dict):
    """The researches the question's user may see (None = all), per settings' access policy."""
    from access import resolve
    return resolve(settings, item.get("user")).researches


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def summarise(rows: list[dict], ref_price: dict | None = None) -> dict:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    answered = [r for r in rows if not r.get("error")]
    tokens = lambda key: sum(r["usage"][key] for r in answered)  # noqa: E731
    return {
        "n": len(rows),
        "errors": len(rows) - len(answered),
        "pass": sum(r["pass"] for r in answered),
        "by_kind": {k: (sum(r["pass"] for r in by_kind[k] if not r.get("error")),
                        sum(1 for r in by_kind[k] if not r.get("error")))
                    for k in KINDS if by_kind[k]},
        "input": tokens("input_tokens"),
        "cached": tokens("cache_read_input_tokens"),
        "output": tokens("output_tokens"),
        "ref_cost": (sum(r["usage"]["input_tokens"] * ref_price["input"]
                         + r["usage"]["cache_read_input_tokens"] * ref_price["cached_input"]
                         + r["usage"]["output_tokens"] * ref_price["output"] for r in answered) / 1e6
                     if ref_price else None),
        "avg_latency_s": round(sum(r["latency_ms"] for r in answered) / max(len(answered), 1) / 1000, 1),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--provider")
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--min-interval", type=float, help="seconds between calls")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="resume a run from another day")
    ap.add_argument("--questions", default=str(HERE / "questions.yaml"),
                    help="questions file; results go to a results/ folder next to it")
    ap.add_argument("--allow-test-rerun", action="store_true",
                    help="run the test split again for a layer/provider that already has a test run")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score all stored answers with the current scorer and rebuild summary.md; no API calls")
    args = ap.parse_args()

    questions = Path(args.questions).resolve()
    results = questions.parent / "results"   # results live next to their questions
    if args.rescore:
        settings = load_settings()
        spec = yaml.safe_load(open(questions, encoding="utf-8"))
        write_summary(results, {i["id"]: i for i in spec["items"]}, load_library(settings), settings)
        print((results / "summary.md").read_text(encoding="utf-8"))
        return 0

    load_secrets()
    settings = load_settings()
    name = pick_provider(settings, args.provider)
    pcfg = settings["providers"][name]
    client = make_client(name, pcfg, settings["model"], settings["client"])
    interval = args.min_interval if args.min_interval is not None else pcfg.get("min_interval_s", 0)

    spec = yaml.safe_load(open(questions, encoding="utf-8"))
    items = [i for i in spec["items"] if args.split == "all" or i["split"] == args.split]
    if args.limit:
        items = items[: args.limit]
    lib = load_library(settings)

    out_dir = results
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.date}_{pipeline.LAYER}_{name}_{args.split}.jsonl"
    if args.split == "test" and not args.allow_test_rerun:
        # Test runs once per layer and provider. On 2026-10-29 a retry loop
        # crossed midnight, the default --date changed, and a second full
        # test run started in a new file. Resume an earlier run with --date.
        earlier = [p for p in out_dir.glob(f"*_{pipeline.LAYER}_{name}_test*.jsonl") if p != out]
        if earlier:
            print(f"refusing: a test run for {pipeline.LAYER}/{name} already exists ({earlier[0].name}). "
                  f"Resume it with --date {earlier[0].name[:10]}, or pass --allow-test-rerun.")
            return 2
    rows = load_rows(out)
    todo = [i for i in items if i["id"] not in rows or rows[i["id"]].get("error")]
    print(f"{pipeline.LAYER} | {name}/{client.model} | {args.split}: {len(items)} questions, "
          f"{len(items) - len(todo)} already done | library {len(lib.cards)} cards | -> {out.name}")

    last_call = 0.0
    for n, item in enumerate(todo, 1):
        row = {"id": item["id"], "kind": item["kind"], "split": item["split"], "q": item["q"],
               "user": item.get("user", settings["access"]["default_user"]),
               "layer": pipeline.LAYER, "provider": name, "settings_model": settings["model"],
               "settings_retrieval": {k: v for k, v in settings["retrieval"].items() if k != "system_prompt"},
               "questions_version": spec["version"]}
        try:
            wait = interval - (time.monotonic() - last_call)
            if wait > 0:
                time.sleep(wait)
            reply = pipeline.answer(client, item["q"], settings, user=item.get("user"),
                                    source="eval", use_cache=False, enforce_budget=False)
            if not reply.extra.get("skipped_model"):
                last_call = time.monotonic()   # throttle only real API calls
            passed, detail = score(item, reply.text, lib, scope_of(item, settings))
            row.update(answer=reply.text, model=reply.model, stop_reason=reply.stop_reason,
                       usage=vars(reply.usage), latency_ms=reply.latency_ms, passed_detail=detail,
                       skipped_model=bool(reply.extra.get("skipped_model")),
                       top_similarity=reply.extra.get("top_similarity"),
                       retrieved=reply.extra.get("retrieved"), path=reply.extra.get("path"),
                       tool_calls=reply.extra.get("tool_calls"), model_turns=reply.extra.get("model_turns"),
                       cost_usd=reply.extra.get("cost_usd"), ref_cost_usd=reply.extra.get("ref_cost_usd"),
                       trace_id=reply.extra.get("trace_id"),
                       expected_count=stored_count({"kind": item["kind"], "passed_detail": detail}),
                       **{"pass": passed})
            mark = "PASS" if passed else "fail"
        except LLMError as e:
            row.update(error=str(e), **{"pass": False})
            mark = "ERR "
        rows[item["id"]] = row
        print(f"  [{n}/{len(todo)}] {item['id']} {mark} {row.get('passed_detail', row.get('error', ''))[:90]}")
        with open(out, "w", encoding="utf-8") as f:      # rewrite: keeps one row per id
            for r in rows.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    done = [rows[i["id"]] for i in items if i["id"] in rows]
    print("\n" + summary_line(out.stem, done, ref_price(settings)))
    errors = sum(1 for r in done if r.get("error"))
    if errors:
        print(f"{errors} questions errored; re-run the same command to retry them.")
    if not args.limit:
        write_summary(out_dir, {i["id"]: i for i in spec["items"]}, lib, settings)
    return 0


def summary_line(run: str, rows: list[dict], ref_price: dict | None = None) -> str:
    s = summarise(rows, ref_price)
    kinds = ", ".join(f"{k} {p}/{t}" for k, (p, t) in s["by_kind"].items())
    return (f"| {run} | {s['pass']}/{s['n'] - s['errors']} ({kinds}) | {s['errors']} | "
            f"{s['input']} + {s['cached']} cached / {s['output']} | "
            f"{'$%.3f' % s['ref_cost'] if s['ref_cost'] is not None else 'n/a'} | {s['avg_latency_s']} s |")


def write_summary(out_dir: Path, items: dict[str, dict], lib: Library, settings: dict) -> None:
    """Rebuild summary.md from every result file, re-scoring stored answers with
    the current scorer (no API calls). Generated, never hand-edited."""
    lines = ["| run (date_layer_provider_split) | pass (by kind) | errors | "
             "tokens in + cached / out | ≈ cost on reference model | avg latency |", "|---|---|---|---|---|---|"]
    for path in sorted(out_dir.glob("*.jsonl")):
        rows = list(load_rows(path).values())
        for r in rows:
            if not r.get("error") and r["id"] in items:
                r["pass"], r["passed_detail"] = score(items[r["id"]], r["answer"], lib, scope_of(items[r["id"]], settings),
                                                      expected_count=stored_count(r))
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        lines.append(summary_line(path.stem, rows, ref_price(settings)))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
