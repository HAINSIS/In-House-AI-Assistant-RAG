"""Build the public demo corpus: paper cards from OpenAlex, in two groups.

    python demo/build_demo_corpus.py              # ~300 cards into data/demo/
    python demo/build_demo_corpus.py --per-group 50

OpenAlex metadata is CC0 (public domain). Nothing is committed: the cards are
downloaded on demand into data/demo/ (gitignored), in the same card format the
indexer reads. Each group becomes a folder in data/demo/groups/ and the
group name is written into its cards, which is how access control assigns
papers to groups (see 02_ingest/research_map.py).

Then:  python 02_ingest/ingest.py  and  python demo/make_demo_questions.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "demo"
API = "https://api.openalex.org/works"

# group name -> OpenAlex search. Two unrelated fields, so access control is visible.
GROUPS = {
    "pedestrian-safety": "pedestrian crossing safety",
    "ev-charging": "electric vehicle charging scheduling",
}
FIELDS = "id,doi,title,publication_year,primary_location,authorships,primary_topic,abstract_inverted_index"


def fetch(search: str, n: int) -> list[dict]:
    works, page = [], 1
    while len(works) < n:
        query = urllib.parse.urlencode({
            "search": search, "per_page": 100, "page": page, "select": FIELDS,
            "filter": "has_abstract:true,type:article,from_publication_date:2015-01-01",
        })
        with urllib.request.urlopen(f"{API}?{query}", timeout=60) as r:
            batch = json.load(r)["results"]
        if not batch:
            break
        works += batch
        page += 1
        time.sleep(0.2)   # stay well inside OpenAlex's rate limit
    return works[:n]


def abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    words = sorted((pos, w) for w, ps in inverted.items() for pos in ps)
    return " ".join(w for _, w in words)


def slug(text: str, n: int) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"a", "an", "the", "of", "and", "for", "in", "on", "with", "to", "by", "from", "using", "based"}
    return "-".join([w for w in words if w not in stop][:n]) or "untitled"


def yaml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def card(w: dict, group: str) -> tuple[str, str] | None:
    text = abstract(w.get("abstract_inverted_index"))
    if len(text.split()) < 40 or not w.get("title"):
        return None
    authors = [a["author"]["display_name"] for a in w.get("authorships") or [] if a.get("author")]
    first = (authors[0].split()[-1] if authors else "anon")
    year = str(w.get("publication_year") or "")
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    topic = (w.get("primary_topic") or {}).get("display_name") or ""
    name = f"{year}_{slug(first, 1)}_{slug(w['title'], 4)}.md"
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    body = "\n".join([
        "---",
        f"title: {yaml_str(w['title'])}",
        f"authors: {yaml_str(', '.join(authors[:8]))}",
        f"year: {year}",
        f"venue: {yaml_str(venue)}",
        f"doi: {yaml_str(doi)}",
        f"url: {yaml_str(w['id'])}",
        f"subtheme: {yaml_str(topic)}",
        f"research: {yaml_str(group)}",
        'source: "OpenAlex (CC0)"',
        "---",
        "",
        f"# {w['title']}",
        "",
        "## What the paper claims",
        "",
        f"> {text}",
        "",
    ])
    return name, body


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=150)
    args = ap.parse_args()

    papers = OUT / "papers"
    papers.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for group, search in GROUPS.items():
        (OUT / "groups" / group).mkdir(parents=True, exist_ok=True)
        written = 0
        for w in fetch(search, args.per_group * 2):      # some lack usable abstracts
            made = card(w, group)
            if not made or made[0] in seen:
                continue
            seen.add(made[0])
            (papers / made[0]).write_text(made[1], encoding="utf-8")
            written += 1
            if written == args.per_group:
                break
        print(f"{group}: {written} cards")
    print(f"done: {len(seen)} cards in {papers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
