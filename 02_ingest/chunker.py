"""Clean card -> chunks small enough to embed and cheap to send.

Each chunk is "<header>\n<section>: <text>": the header (title, year, venue)
travels with every chunk so a retrieved chunk is self-explanatory, and the
answer can cite the paper without a second lookup. Sections are packed
together up to `max_words`; a section longer than that is split on paragraph
then sentence boundaries. bge-m3 reads up to 8192 tokens, but long chunks
cost more to send at answer time; 300 words is ~400 tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from cards import Card


@dataclass
class Chunk:
    id: str           # "<file>#<n>"
    file: str
    section: str      # heading(s) the text came from
    text: str         # header + section text: what gets embedded and sent


def _units(text: str, max_words: int) -> list[str]:
    """Paragraphs/sentences; anything still too long (no sentence breaks) is
    cut by word count."""
    out = []
    for unit in re.split(r"\n\n+|(?<=[.!?])\s+(?=[A-Z])", text):
        words = unit.split()
        for i in range(0, max(len(words), 1), max_words):
            out.append(" ".join(words[i:i + max_words]))
    return [u for u in out if u]


def _split_long(text: str, max_words: int) -> list[str]:
    pieces, cur = [], []
    for unit in _units(text, max_words):
        n = len(unit.split())
        if cur and sum(len(u.split()) for u in cur) + n > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(unit)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_card(card: Card, max_words: int = 300) -> list[Chunk]:
    units: list[tuple[str, str]] = []
    for heading, text in card.sections:
        for piece in _split_long(text, max_words):
            units.append((heading, piece))

    chunks: list[Chunk] = []
    cur: list[tuple[str, str]] = []

    def flush() -> None:
        if not cur:
            return
        body = "\n".join(f"{h}: {t}" for h, t in cur)
        sections = " | ".join(dict.fromkeys(h for h, _ in cur))
        n = len(chunks)
        chunks.append(Chunk(f"{card.file}#{n}", card.file, sections, f"{card.header}\n{body}"))
        cur.clear()

    for heading, text in units:
        if cur and sum(len(t.split()) for _, t in cur) + len(text.split()) > max_words:
            flush()
        cur.append((heading, text))
    flush()
    if not chunks:  # no usable section: the title alone still identifies the paper
        chunks.append(Chunk(f"{card.file}#0", card.file, "title", card.header))
    return chunks
