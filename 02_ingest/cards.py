"""Paper card -> metadata + clean text sections.

Cards come from the paper library (read-only). Breadth-pass cards hold
one real section (the author abstract) plus placeholders ("not assessed
beyond the abstract", "Not extracted (breadth pass)"), sweep logs and HTML
comments. Indexing those would make every retrieved chunk longer and noisier,
so they are dropped here: only text that says something about the paper is kept.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

META_KEYS = ("title", "authors", "year", "venue", "doi", "url", "subtheme")

# A line is boilerplate if it only says that something wasn't assessed.
_BOILERPLATE = re.compile(
    r"not assessed|not extracted|not specified in the abstract|breadth[- ]pass|"
    r"full text not read|method not independently re-derived|"
    r"any numbers are only those inside the abstract|see that synthesis, once written",
    re.I,
)
# Sections that are process logs about the card, not about the paper.
_SWEEP_NOTE = re.compile(r"^\s*>?\s*Logged by the `", re.M)


@dataclass
class Card:
    file: str
    sha: str                                  # content hash: re-ingest only when it changes
    meta: dict
    sections: list[tuple[str, str]] = field(default_factory=list)   # (heading, clean text)

    @property
    def header(self) -> str:
        """One line that identifies the paper; prefixed to every chunk."""
        m = self.meta
        bits = [m["title"]]
        tail = ", ".join(x for x in (m["year"], m["venue"]) if x)
        if tail:
            bits.append(f"({tail})")
        return " ".join(bits)


def _field(frontmatter: str, key: str) -> str:
    m = re.search(rf'^{key}:\s*"?(.*?)"?\s*$', frontmatter, re.M)
    return m.group(1).strip() if m else ""


def _clean(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = text.replace("\\n", " ")                    # escaped newlines from harvest APIs
    kept = []
    for line in text.splitlines():
        if _BOILERPLATE.search(line):
            continue
        line = re.sub(r"^\s*>\s?", "", line)          # unquote abstracts
        kept.append(line.rstrip())
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # A list whose items were all boilerplate leaves only labels behind.
    return "" if re.fullmatch(r"[\s\-*:]*", text) else text


def parse(path: Path) -> Card:
    raw = path.read_text(encoding="utf-8")
    sha = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    parts = raw.split("---", 2)
    front, body = (parts[1], parts[2]) if raw.startswith("---") and len(parts) == 3 else ("", raw)
    meta = {k: _field(front, k).replace("\\n", " ") for k in META_KEYS}
    if not meta["title"]:
        m = re.search(r"^# (.+)$", body, re.M)
        meta["title"] = m.group(1).strip() if m else path.stem

    card = Card(file=path.name, sha=sha, meta=meta)
    for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", body, re.S | re.M):
        heading, text = m.group(1).strip(), m.group(2)
        if _SWEEP_NOTE.search(text):
            continue
        text = _clean(text)
        if text:
            card.sections.append((heading, text))
    return card
