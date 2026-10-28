"""Build or update the index from the paper library.

    python 02_ingest/ingest.py               # incremental: new/changed cards only
    python 02_ingest/ingest.py --limit 200   # smoke test on the first 200 cards
    python 02_ingest/ingest.py --dry-run     # plan + chunk stats, no model, no writes
    python 02_ingest/ingest.py --rebuild     # drop the index and start over

Safe to stop and re-run: cards are written in batches with their content
hash, so a re-run skips everything already embedded. Prints throughput and
an ETA after each batch, and appends a run record to <index>/ingest_log.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "_shared")]

from cards import parse  # noqa: E402
from chunker import chunk_card  # noqa: E402
from research_map import build as build_research_map, research_of  # noqa: E402
from settings import load_settings  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N cards (sorted by file name)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    s = load_settings()
    ix = s["index"]
    papers = (ROOT / s["corpus"]["root"] / s["corpus"]["papers_dir"]).resolve()
    index_dir = (ROOT / ix["dir"]).resolve()
    files = sorted(papers.glob("*.md"))
    if args.limit:
        files = files[: args.limit]

    if args.rebuild and index_dir.exists() and not args.dry_run:
        shutil.rmtree(index_dir)

    t0 = time.perf_counter()
    if args.dry_run:
        chunks = [c for f in files for c in chunk_card(parse(f), ix["max_chunk_words"])]
        print(f"{len(files)} cards -> {len(chunks)} chunks, ~{sum(len(c.text) for c in chunks)//4:,} tokens "
              f"({time.perf_counter() - t0:.1f}s). Nothing written.")
        return 0

    from embedder import Embedder
    from store import Store

    emb = Embedder(ix["embed_model"], ix["device"], ix["batch_size"], ix.get("hf_endpoint"))
    store = Store(index_dir, emb.dim)
    print(f"model {ix['embed_model']} on {emb.device} (dim {emb.dim}) | index {index_dir}")

    have = store.indexed()
    current = {f.name: f for f in files}
    removed = [] if args.limit else [f for f in have if f not in current]
    cards = [parse(f) for f in files]
    by_synthesis, slugs = build_research_map((ROOT / s["corpus"]["root"] / s["corpus"]["researches_dir"]).resolve())
    research = {c.file: research_of(c.file, current[c.file].read_text(encoding="utf-8"), by_synthesis, slugs)
                for c in cards}
    todo = [c for c in cards if have.get(c.file) != c.sha]
    changed = [c.file for c in todo if c.file in have]
    store.delete_files(removed + changed)
    print(f"{len(files)} cards: {len(files) - len(todo)} unchanged, {len(todo) - len(changed)} new, "
          f"{len(changed)} changed, {len(removed)} removed")

    done_chunks = 0
    t_embed = time.perf_counter()
    step = ix["cards_per_write"]
    for i in range(0, len(todo), step):
        batch = todo[i:i + step]
        chunks = [(c, ch) for c in batch for ch in chunk_card(c, ix["max_chunk_words"])]
        vectors = emb.embed([ch.text for _, ch in chunks])
        store.add([{
            "id": ch.id, "file": c.file, "sha": c.sha, "title": c.meta["title"],
            "authors": c.meta["authors"], "year": c.meta["year"], "venue": c.meta["venue"],
            "doi": c.meta["doi"], "subtheme": c.meta["subtheme"], "research": research[c.file],
            "section": ch.section,
            "text": ch.text, "vector": v,
        } for (c, ch), v in zip(chunks, vectors)])
        done_chunks += len(chunks)
        rate = done_chunks / (time.perf_counter() - t_embed)
        left = (len(todo) - i - len(batch)) * (done_chunks / (i + len(batch))) / rate
        print(f"  {i + len(batch)}/{len(todo)} cards, {done_chunks} chunks, {rate:.1f} chunks/s, "
              f"ETA {left / 60:.1f} min", flush=True)

    store.build_fts()   # seconds at this size; always rebuilt so an interrupted run can't leave it stale
    record = {
        "date": dt.datetime.now().isoformat(timespec="seconds"), "model": ix["embed_model"],
        "device": emb.device, "cards": len(files), "embedded_cards": len(todo),
        "removed": len(removed), "chunks_embedded": done_chunks, "rows_total": store.count(),
        "seconds": round(time.perf_counter() - t0, 1), "limit": args.limit,
    }
    with open(index_dir / "ingest_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"done: {record['rows_total']} chunks in the index ({record['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
