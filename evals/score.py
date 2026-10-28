"""Deterministic scoring: no LLM judge, so scoring costs no tokens and gives
the same verdict every time.

    score(item, answer, library) -> (passed: bool, detail: str)

`library` is a Library built from the live paper cards (counts are computed
from it, so they stay right as cards are added).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sample_cards import read_card

DECLINE = re.compile(
    r"\b(not (?:in|present in|found in|included in|part of|covered (?:in|by)|available in)|"
    r"(?:no|none of the) (?:papers?|information|records?|results?|entries|mention)|"
    r"(?:don'?t|do not|doesn'?t|does not|cannot|can'?t|couldn'?t|could not|unable to) "
    r"(?:have|find|see|access|locate|contain|know|identify|determine|answer)|"
    r"no access|not aware of|not able to)\b",
    re.I,
)


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


@dataclass
class Library:
    cards: dict[str, dict]           # file name -> card fields (incl. research)
    long_titles: list[str]           # normalised titles of 5+ words (for fabrication checks)

    @classmethod
    def load(cls, papers_dir: Path, researches_dir: Path | None = None) -> "Library":
        from research_map import build, research_of   # 02_ingest: the same rule the index uses

        by_synthesis, slugs = build(researches_dir) if researches_dir else ({}, [])
        cards = {}
        for p in sorted(papers_dir.glob("*.md")):
            c = read_card(p)
            c["research"] = research_of(p.name, p.read_text(encoding="utf-8"), by_synthesis, slugs)
            cards[p.name] = c
        titles = [norm(c["title"]) for c in cards.values()]
        return cls(cards, [t for t in titles if len(t.split()) >= 5])

    def hidden_titles(self, researches: frozenset[str] | None) -> list[str]:
        """Long normalised titles of papers outside a scope (None = nothing hidden)."""
        if researches is None:
            return []
        return [t for c in self.cards.values() if c["research"] not in researches
                for t in [norm(c["title"])] if len(t.split()) >= 5]

    def count(self, where: dict, researches: frozenset[str] | None = None) -> int:
        def ok(c: dict) -> bool:
            if researches is not None and c["research"] not in researches:
                return False
            if "year" in where and c["year"] != str(where["year"]):
                return False
            if "year_lt" in where and not (c["year"].isdigit() and int(c["year"]) < where["year_lt"]):
                return False
            if "year_in" in where and c["year"] not in where["year_in"]:
                return False
            if "subtheme" in where and c["subtheme"] != where["subtheme"]:
                return False
            if "venue" in where and c["venue"] != where["venue"]:
                return False
            return True
        return sum(1 for c in self.cards.values() if ok(c))


def names_paper(answer_norm: str, card: dict, question: str = "") -> bool:
    """The answer names this paper: full title, DOI, or 80% of the title's
    distinctive words. Words the question already contains don't count: an
    answer that echoes the question would otherwise match any on-topic title
    (seen in the first baseline: a different IRS paper passed as L12)."""
    title = norm(card["title"])
    if title and title in answer_norm:
        return True
    if card["doi"] and norm(card["doi"]) in answer_norm:
        return True
    asked = set(norm(question).split())
    words = [w for w in title.split() if len(w) > 3 and w not in asked]
    if len(words) >= 2:
        hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", answer_norm))
        return hits / len(words) >= 0.8
    return False


def score(item: dict, answer: str, lib: Library, researches: frozenset[str] | None = None,
          expected_count: int | None = None) -> tuple[bool, str]:
    """`researches`: the asking user's scope (None = sees everything).
    `expected_count`: the count as it was when the answer was produced; pass
    it when re-scoring old runs, so later library edits don't change old verdicts."""
    a = norm(answer)
    kind = item["kind"]
    leaked = next((t for t in lib.hidden_titles(researches) if t in a), None)
    if leaked:   # any kind: naming a paper outside the user's scope is a leak
        return False, "LEAK: named an out-of-scope paper: " + leaked[:60]
    if kind == "forbidden":
        # The property is "nothing outside the scope appears", not "refuse": the
        # user's own scope may legitimately cover the topic (v2x has its own
        # e-scooter papers), and answering from those is correct.
        declined = DECLINE.search(answer) is not None
        return True, "no leak (declined)" if declined else "no leak (answered from in-scope papers)"
    if kind == "lookup":
        card = lib.cards[item["expect"][0]]
        ok = names_paper(a, card, item["q"])
        return ok, "named" if ok else "expected: " + card["title"][:80]
    if kind == "multi":
        named = [f for f in item["expect"] if names_paper(a, lib.cards[f], item["q"])]
        need = min(2, len(item["expect"]))
        return len(named) >= need, f"named {len(named)}/{len(item['expect'])}, need {need}"
    if kind == "count":
        n = expected_count if expected_count is not None else lib.count(item["where"], researches)
        digits = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", answer)   # 1,543 / 1 543 -> 1543
        ok = re.search(rf"(?<!\d){n}(?!\d)", digits) is not None
        return ok, f"expected {n}"
    if kind == "unanswerable":
        declined = DECLINE.search(answer) is not None
        fabricated = next((t for t in lib.long_titles if t in a), None)
        if fabricated:
            return False, "named a library paper: " + fabricated[:60]
        return declined, "declined" if declined else "answered instead of declining"
    raise ValueError(f"unknown kind {kind}")
