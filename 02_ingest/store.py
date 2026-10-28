"""The index: one LanceDB table of chunks with their vectors and paper metadata.

Embedded (a folder on disk, no server). Incremental: each row carries its
card's content hash, so a re-run only re-embeds cards that changed or are new,
and drops rows of cards that disappeared. See docs/decisions/0001-vector-store.md.
"""
from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

TABLE = "chunks"


def schema(dim: int) -> pa.Schema:
    return pa.schema([
        ("id", pa.string()), ("file", pa.string()), ("sha", pa.string()),
        ("title", pa.string()), ("authors", pa.string()), ("year", pa.string()),
        ("venue", pa.string()), ("doi", pa.string()), ("subtheme", pa.string()),
        ("research", pa.string()),   # the card's group; 05_access scopes on it
        ("section", pa.string()), ("text", pa.string()),
        ("vector", pa.list_(pa.float32(), dim)),
    ])


class Store:
    def __init__(self, index_dir: Path, dim: int):
        index_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(index_dir))
        if TABLE in self.db.table_names():
            self.table = self.db.open_table(TABLE)
        else:
            self.table = self.db.create_table(TABLE, schema=schema(dim))

    def indexed(self) -> dict[str, str]:
        """file -> content hash, for every card already in the index."""
        if self.table.count_rows() == 0:
            return {}
        n = self.table.count_rows()
        t = self.table.search().select(["file", "sha"]).limit(n).to_arrow()
        return dict(zip(t.column("file").to_pylist(), t.column("sha").to_pylist()))

    def delete_files(self, files: list[str]) -> None:
        for i in range(0, len(files), 500):
            batch = files[i:i + 500]
            quoted = ", ".join("'" + f.replace("'", "''") + "'" for f in batch)
            self.table.delete(f"file IN ({quoted})")

    def add(self, rows: list[dict]) -> None:
        if rows:
            self.table.add(rows)

    def build_fts(self) -> None:
        """Full-text (BM25) index over chunk text, for hybrid search in 03_retrieve."""
        self.table.create_fts_index("text", replace=True)

    def count(self) -> int:
        return self.table.count_rows()
