"""Search hits -> the prompt the model sees, and the model's numbered citations
-> the titles the user sees.

The model cites sources as [n] instead of writing titles: ~15 output tokens
saved per citation, and no chance of a misspelled or invented title. The app
appends the titles of the cited sources only, so a source the model did not
use never appears in the answer.
"""
from __future__ import annotations

import re

from retriever import Hit

NOT_FOUND = "Not in the library."


def build_prompt(question: str, hits: list[Hit]) -> str:
    sources = "\n\n".join(f"[{n}] {h.text}" for n, h in enumerate(hits, 1))
    return f"Sources:\n{sources}\n\nQuestion: {question}"


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    nums: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]", answer):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[–-]", part):
                a, b = (int(x) for x in re.split(r"\s*[–-]\s*", part))
                nums.extend(range(a, b + 1))
            else:
                nums.append(int(part))
    return [n for n in dict.fromkeys(nums) if 1 <= n <= n_sources]


def sources_block(answer: str, hits: list[Hit]) -> str:
    cited = cited_numbers(answer, len(hits))
    if not cited:
        return ""
    lines = []
    for n in cited:
        h = hits[n - 1]
        tail = ", ".join(x for x in (h.year, h.venue) if x)
        lines.append(f"[{n}] {h.title}" + (f" ({tail})" if tail else ""))
    return "\n\nSources:\n" + "\n".join(lines)
