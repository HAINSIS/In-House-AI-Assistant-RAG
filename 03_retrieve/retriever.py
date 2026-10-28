"""Hybrid search over the 02_ingest index: question -> the best chunk of each
of the top-k papers.

Vector search (bge-m3) finds paraphrases; BM25 finds exact terms (dataset
names, acronyms). Their rankings are merged by reciprocal rank fusion, which
needs no score calibration between the two. Only the best chunk per paper is
kept: a second chunk of the same paper rarely changes the answer but always
costs ~400 tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import lancedb

from embedder import Embedder

RRF_K = 60  # standard constant; damps the weight of top ranks


@dataclass
class Hit:
    file: str
    title: str
    year: str
    venue: str
    doi: str
    text: str           # the chunk: header line + section text
    score: float        # fused RRF score
    similarity: float   # best cosine similarity of this paper's chunks (0 if only BM25 found it)


class Retriever:
    def __init__(self, index_dir: Path, embedder: Embedder, pool: int = 40):
        self.table = lancedb.connect(str(index_dir)).open_table("chunks")
        self.embedder = embedder
        self.pool = pool  # candidates taken from each search before fusion

    def search(self, question: str, k: int = 5, where: str | None = None,
               relative_cutoff: float | None = None) -> list[Hit]:
        """`where` is an SQL filter on the index columns (e.g. an access scope),
        applied before ranking: rows it excludes are never candidates.
        `relative_cutoff`: of the top k, keep only papers whose similarity is
        within this margin of the best one (at least one is kept). A clear
        single match then costs one chunk instead of k."""
        hits = self._search(question, k, where)
        if relative_cutoff is not None and hits:
            best = max(h.similarity for h in hits)
            hits = [h for h in hits if h.similarity >= best - relative_cutoff] or hits[:1]
        return hits

    def _search(self, question: str, k: int, where: str | None) -> list[Hit]:
        vec = self.embedder.embed([question])[0]
        q = self.table.search(vec).metric("cosine")
        if where:
            q = q.where(where, prefilter=True)
        by_vector = q.limit(self.pool).to_list()
        terms = re.sub(r"[^\w\s-]", " ", question)
        try:
            q = self.table.search(terms, query_type="fts")
            if where:
                q = q.where(where, prefilter=True)
            by_text = q.limit(self.pool).to_list()
        except Exception:  # an all-stopword query can fail in the FTS parser
            by_text = []

        fused: dict[str, float] = {}
        rows: dict[str, dict] = {}
        sims: dict[str, float] = {}
        for ranking, is_vec in ((by_vector, True), (by_text, False)):
            for rank, row in enumerate(ranking):
                fused[row["id"]] = fused.get(row["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
                rows[row["id"]] = row
                if is_vec:
                    sims[row["file"]] = max(sims.get(row["file"], 0.0), 1.0 - row["_distance"])

        best: dict[str, tuple[float, dict]] = {}   # file -> (score, best chunk row)
        for cid, score in fused.items():
            row = rows[cid]
            if row["file"] not in best or score > best[row["file"]][0]:
                best[row["file"]] = (score, row)
        top = sorted(best.values(), key=lambda x: -x[0])[:k]
        return [Hit(r["file"], r["title"], r["year"], r["venue"], r["doi"], r["text"], s,
                    sims.get(r["file"], 0.0)) for s, r in top]

    def top_similarity(self, question: str) -> float:
        """Unscoped; for offline analysis only, never in a user's answer path."""
        vec = self.embedder.embed([question])[0]
        rows = self.table.search(vec).metric("cosine").limit(1).to_list()
        return 1.0 - rows[0]["_distance"] if rows else 0.0
