# 0001 — LanceDB as the local vector store

Date: 2026-10-28 · Component: 02_ingest · Status: accepted

## Context
The index holds ~13k chunks now (1024-dim vectors) and may grow ~20× when PDFs
are added. Retrieval needs vector search, keyword (BM25) search for exact terms
like dataset names and acronyms, and metadata filters (year, venue, subtheme).
It must run on one Windows laptop with nothing to host, so a fresh clone can
run the demo.

## Options
| Option | For | Against |
|---|---|---|
| LanceDB (embedded) | A folder on disk, no server; vector search, built-in full-text index and SQL-style filters in one table; handles the PDF-scale corpus without changing approach | One more dependency |
| NumPy matrix + SQLite FTS5 | No extra dependencies; brute-force search of 13k vectors takes milliseconds | Hand-written persistence, incremental updates and hybrid merge; brute force slows down at PDF scale |
| Chroma | Popular, simple API | Full-text search is weaker than BM25; hybrid search means merging results by hand |
| Server databases (Qdrant, Milvus, pgvector) | Production features | A service to run and host; overkill for one user |

## Decision
I use LanceDB, with one table that holds vectors, text and paper metadata, plus a BM25 index on the text.

## Evidence
Scale: 12,939 chunks from 11,476 cards ([architecture](../architecture.md#index-02_ingest)).
Speed and quality: see [results](../results.md).

## Consequences
Hybrid search and filters need no glue code. If LanceDB's Windows support becomes a
problem, NumPy + SQLite FTS5 is the fallback: same table shape, about 100 lines.
