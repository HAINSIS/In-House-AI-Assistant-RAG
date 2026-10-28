"""Which group ("research") each paper card belongs to.

Access control (05_access) scopes users to researches, so every chunk in the
index carries a `research` value. Rule, in order:
1. a synthesis under researches/<slug>/01_landscape/syntheses/ lists the card
   in its `paper_files` frontmatter -> <slug>;
2. the card's text names a group folder (a `research: <slug>` field, as in the
   demo corpus, or a log line that mentions it) -> <slug>;
3. otherwise "unassigned" (only roles with `researches: all` see those).
`researches_dir` holds one folder per group; the folder names are the slugs.
"""
from __future__ import annotations

import re
from pathlib import Path

UNASSIGNED = "unassigned"


def build(researches_dir: Path) -> tuple[dict[str, str], list[str]]:
    """-> (card file name -> research slug from syntheses, all research slugs)."""
    slugs = sorted(p.name for p in researches_dir.iterdir() if p.is_dir()) if researches_dir.is_dir() else []
    by_synthesis: dict[str, str] = {}
    for slug in slugs:
        for syn in (researches_dir / slug / "01_landscape" / "syntheses").glob("*.md"):
            front = syn.read_text(encoding="utf-8").split("---", 2)
            block = front[1] if len(front) == 3 else ""
            for f in re.findall(r"library/papers/([^\s\"']+\.md)", block):
                by_synthesis.setdefault(f, slug)
    return by_synthesis, slugs


def research_of(file: str, raw_text: str, by_synthesis: dict[str, str], slugs: list[str]) -> str:
    if file in by_synthesis:
        return by_synthesis[file]
    for slug in slugs:
        if slug in raw_text:
            return slug
    return UNASSIGNED
