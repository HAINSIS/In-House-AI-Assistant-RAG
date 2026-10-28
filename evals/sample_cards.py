"""Reproducible card sample for writing eval questions.

    python evals/sample_cards.py --n 40 --seed 20261028 > sample.txt

Prints cards (file, year, subtheme, title, abstract start) from the paper
library in a fixed shuffled order, skipping cards without a real abstract.
The question author reads this and writes questions into questions.yaml;
the seed is recorded there so the sample can be regenerated.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "_shared"))
from settings import load_settings  # noqa: E402


def papers_dir() -> Path:
    s = load_settings()
    return (ROOT / s["corpus"]["root"] / s["corpus"]["papers_dir"]).resolve()


def read_card(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    def field(key: str) -> str:
        m = re.search(rf'^{key}:\s*"?([^"\n]*)', text, re.M)
        return m.group(1).strip() if m else ""

    m = re.search(r"## What the paper claims.*?\n>\s*(.+?)\n\n", text, re.S)
    abstract = m.group(1).replace("\n> ", " ") if m else ""
    return {
        "file": path.name,
        "title": field("title"),
        "year": field("year"),
        "venue": field("venue"),
        "doi": field("doi"),
        "subtheme": field("subtheme"),
        "abstract": abstract,
    }


def has_abstract(card: dict) -> bool:
    return len(card["abstract"]) >= 300 and not card["abstract"].startswith("Logged by")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N usable cards (next batch)")
    args = ap.parse_args()

    files = sorted(papers_dir().glob("*.md"))
    random.Random(args.seed).shuffle(files)
    shown = usable = 0
    for f in files:
        card = read_card(f)
        if not has_abstract(card):
            continue
        usable += 1
        if usable <= args.skip:
            continue
        print(f"[{args.skip + shown}] {card['file']} | {card['year']} | {card['subtheme']}")
        print(f"  T: {card['title']}")
        print(f"  A: {card['abstract'][:420]}\n")
        shown += 1
        if shown == args.n:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
