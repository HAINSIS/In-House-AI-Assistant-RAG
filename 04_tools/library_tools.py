"""Tools over the paper library, shared by the app's tool loop and the MCP server.

Deterministic where possible: counting and listing never touch a model, they
read the index's metadata (one row per paper). Only `search_papers` uses the
embedding model. Outputs are short JSON strings: every tool result is sent
back to the model, so it costs tokens.

Metadata comes from the index (as of the last 02_ingest run), not the live
library: re-run ingest after cards are added or edited.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import lancedb

FIELDS = ("year", "venue", "subtheme")

# Kept short on purpose: tool specs are resent with every tool-path request.
SPECS = [
    {
        "name": "count_papers",
        "description": "Count papers in the library matching all given filters. "
                       "Venue and subtheme must be exact names (case-insensitive); use list_values if unsure.",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "year_before": {"type": "integer", "description": "strictly earlier than this year"},
                "year_after": {"type": "integer", "description": "strictly later than this year"},
                "venue": {"type": "string"},
                "subtheme": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_values",
        "description": "List the exact values of a metadata field with paper counts, most common first.",
        "parameters": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": list(FIELDS)},
                "contains": {"type": "string", "description": "only values containing this text"},
                "limit": {"type": "integer", "description": "default 20"},
            },
            "required": ["field"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_papers",
        "description": "Search the library by meaning and keywords; returns the best-matching papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "default 5, max 10"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_card",
        "description": "Read one paper's card (abstract, method, results) by its file name from search_papers.",
        "parameters": {
            "type": "object",
            "properties": {"file": {"type": "string"}},
            "required": ["file"],
            "additionalProperties": False,
        },
    },
]


class ToolError(ValueError):
    """Bad arguments; the message goes back to the model so it can retry."""


class OpenScope:
    """No restrictions. Any object with the same three members works as a scope
    (05_access's Principal does); this layer doesn't know about roles."""
    where = None

    def sees(self, research: str) -> bool:
        return True

    def may_use(self, tool: str) -> bool:
        return True


OPEN = OpenScope()


def specs_for(scope=OPEN) -> list[dict]:
    """Only the tools this scope may use: the model never sees the others."""
    return [s for s in SPECS if scope.may_use(s["name"])]


class LibraryTools:
    """`parse_card` (02_ingest's card parser) and `retriever_factory` are passed
    in by app/, the only place allowed to wire layers together."""

    def __init__(self, index_dir: Path, papers_dir: Path, parse_card, retriever_factory=None):
        self.parse_card = parse_card
        table = lancedb.connect(str(index_dir)).open_table("chunks")
        rows = table.search().select(["file", "title", "year", "venue", "subtheme", "research"]) \
            .limit(table.count_rows()).to_arrow().to_pylist()
        self.papers = {r["file"]: r for r in rows}          # one entry per paper
        self.scope = OPEN                                   # set per call by call()
        self.papers_dir = papers_dir
        self._retriever_factory = retriever_factory          # lazy: loads the embedding model
        self._retriever = None

    # --- tools (each sees only the papers the current scope allows) ----------

    def _visible(self) -> list[dict]:
        return [p for p in self.papers.values() if self.scope.sees(p["research"])]

    def count_papers(self, year: int | None = None, year_before: int | None = None,
                     year_after: int | None = None, venue: str | None = None,
                     subtheme: str | None = None) -> dict:
        def ok(p: dict) -> bool:
            y = int(p["year"]) if p["year"].isdigit() else None
            if year is not None and y != year:
                return False
            if year_before is not None and (y is None or y >= year_before):
                return False
            if year_after is not None and (y is None or y <= year_after):
                return False
            if venue and p["venue"].casefold() != venue.casefold():
                return False
            if subtheme and p["subtheme"].casefold() != subtheme.casefold():
                return False
            return True

        n = sum(ok(p) for p in self._visible())
        filters = {k: v for k, v in dict(year=year, year_before=year_before, year_after=year_after,
                                         venue=venue, subtheme=subtheme).items() if v is not None}
        out = {"count": n, "filters": filters}
        if n == 0:
            # Say whether the name exists at all (in this scope) and offer the
            # closest real names, so the model needn't search turn after turn.
            for field, name in (("venue", venue), ("subtheme", subtheme)):
                if not name:
                    continue
                values = {p[field] for p in self._visible() if p[field]}
                if name.casefold() in {v.casefold() for v in values}:
                    continue
                words = {w for w in name.casefold().replace("-", " ").replace("/", " ").split() if len(w) > 3}
                need = min(2, len(words))     # one shared word ("vehicle") is not "close"
                overlap = {v: sum(w in v.casefold() for w in words) for v in values}
                close = [v for v, n in sorted(overlap.items(), key=lambda x: (-x[1], x[0]))
                         if need and n >= need][:5]
                out[f"no_such_{field}"] = name
                out["closest_names"] = close
                out["note"] = (f"there is no {field} named that in the library; the count is 0"
                               + ("; if the question meant one of closest_names, count that one" if close else ""))
        return out

    def list_values(self, field: str, contains: str | None = None, limit: int = 20) -> dict:
        if field not in FIELDS:
            raise ToolError(f"field must be one of {FIELDS}")
        counts = Counter(p[field] for p in self._visible() if p[field])
        if contains:
            counts = Counter({v: n for v, n in counts.items() if contains.casefold() in v.casefold()})
        top = counts.most_common(max(1, min(limit or 20, 50)))
        return {"field": field, "values": [[v, n] for v, n in top], "distinct": len(counts)}

    def search_papers(self, query: str, k: int = 5) -> dict:
        if self._retriever is None:
            if self._retriever_factory is None:
                raise ToolError("search is not available in this context")
            self._retriever = self._retriever_factory()
        hits = self._retriever.search(query, max(1, min(k or 5, 10)), where=self.scope.where)
        return {"papers": [{"file": h.file, "title": h.title, "year": h.year, "venue": h.venue,
                            "similarity": round(h.similarity, 3)} for h in hits]}

    def read_card(self, file: str, max_words: int = 600) -> dict:
        name = Path(file).name                               # confine to the library folder
        # Out of scope looks exactly like nonexistent: same error, no hint.
        if name not in self.papers or not self.scope.sees(self.papers[name]["research"]):
            raise ToolError(f"unknown file {file!r}; use a file name returned by search_papers")
        card = self.parse_card(self.papers_dir / name)
        text = "\n".join(f"{h}: {t}" for h, t in card.sections)
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + " …"
        return {"file": name, "title": card.meta["title"], "year": card.meta["year"],
                "venue": card.meta["venue"], "doi": card.meta["doi"], "text": text}

    # --- dispatch ----------------------------------------------------------

    def call(self, name: str, args: dict, scope=OPEN) -> str:
        """Run a tool by name within `scope`; always returns a JSON string (errors included)."""
        fn = {"count_papers": self.count_papers, "list_values": self.list_values,
              "search_papers": self.search_papers, "read_card": self.read_card}.get(name)
        self.scope = scope
        try:
            if fn is None or not scope.may_use(name):
                raise ToolError(f"unknown tool {name!r}")   # a forbidden tool looks unknown
            if not isinstance(args, dict):
                raise ToolError("arguments must be a JSON object")
            spec = next(s for s in SPECS if s["name"] == name)["parameters"]
            unknown = set(args) - set(spec["properties"])
            missing = set(spec.get("required", [])) - set(args)
            if unknown or missing:
                raise ToolError(f"unknown args {sorted(unknown)}; missing {sorted(missing)}")
            for key, val in args.items():
                if spec["properties"][key]["type"] == "integer" and val is not None:
                    try:
                        args[key] = int(val)
                    except (TypeError, ValueError):
                        raise ToolError(f"{key} must be an integer") from None
            result = fn(**args)
        except ToolError as e:
            result = {"error": str(e)}
        return json.dumps(result, ensure_ascii=False)
