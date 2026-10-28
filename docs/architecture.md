# Architecture

One request, end to end, as implemented in `app/pipeline.py` (the only module
that wires components together; each component takes and returns plain data).

```
question, user
  │
  ├─ 05_access    resolve(user) → Principal(role, researches, tools)
  ├─ 06_cost      over the role's daily token budget? → refusal (no model call)
  ├─ 06_cost      same question, same scope, same settings and index? → cached answer (0 tokens)
  │
  ├─ counting question ("how many …")?
  │     04_tools  model + tools allowed for the role, ≤ 4 turns; last turn answers
  │
  └─ otherwise
        03_retrieve  hybrid search filtered to the scope → best chunk per paper
                     best similarity < 0.55 → "Not in the library." (no model call)
                     else papers within 0.08 of the best → numbered sources + question
        01_wrapper   any provider → answer citing [n]; app appends the cited titles
  │
  ├─ 06_cost      tokens and cost into the per-user ledger; answer into the cache
  └─ 07_observe   one trace row (or an error row) per request
```

## Model client (`01_wrapper/`)

- `make_client()` builds an `LLMClient` from one provider entry in
  `settings.yaml`; providers are configuration, not code.
- Two adapters. **Anthropic** uses the native SDK because explicit prompt
  caching and the server-side refusal fallback exist only there. **OpenAI-
  compatible** covers everything else (DeepSeek, Qwen, Gemini, GLM, Moonshot,
  OpenRouter, Ollama, OpenAI); provider-specific switches pass through
  `extra_body`, and domestic or local endpoints can bypass a proxy.
- Always streams. Usage is normalised to uncached input, cached input, cache
  write and output, whatever the provider reports.
- Tool calling is provider-neutral: assistant turns carry `tool_calls`, results
  come back as `role: "tool"` messages, each adapter translates. Within a tool
  loop the Anthropic adapter returns the assistant's own content blocks
  unchanged (thinking blocks must go back as they were).
- History is text only: the shape every provider accepts, and earlier
  reasoning is never re-sent.
- A stream that ends empty with a non-standard finish reason raises instead of
  returning an empty answer, so callers retry.

## Index (`02_ingest/`)

- `cards.py` parses frontmatter and `##` sections and drops placeholder lines,
  HTML comments and process logs; only text about the paper is kept.
- `chunker.py` packs sections into chunks of ≤ 300 words (~400 tokens). Every
  chunk starts with "Title (year, venue)", so a retrieved chunk is
  self-explanatory and citable.
- `embedder.py` runs bge-m3 locally (multilingual, 1024-dim, normalised
  vectors). Once the model is cached it never touches the network.
- `store.py` keeps one LanceDB table: vector, text, title, authors, year,
  venue, DOI, subtheme, group, section, file, content hash; plus a BM25 index.
  Why LanceDB: [decisions/0001](decisions/0001-vector-store.md).
- `ingest.py` is incremental (content hash per card), resumable, and logs
  each run; the log's last line is the index version used by the answer cache.

## Retrieval and citations (`03_retrieve/`)

- Vector search and BM25 (top 40 each) merged by reciprocal rank fusion, the
  best chunk per paper, top k = 5 papers, then only those within 0.08
  similarity of the best.
- The access scope is applied as a prefilter on both searches, so excluded
  rows are never candidates.
- Sources go in the user turn; the system prompt stays byte-stable for prefix
  caching. The model cites `[n]`, `[1, 3]` or `[2-4]`; the app appends the
  titles of the cited sources only.

## Tools and MCP (`04_tools/`)

- `count_papers`, `list_values`, `search_papers`, `read_card`: deterministic
  except search, compact JSON results, arguments validated, errors returned to
  the model so it can correct itself. A zero count on a name that doesn't exist
  says so and offers the closest real names.
- `specs_for(scope)` shows the model only the tools its role may use; the
  executor refuses the rest as unknown. `read_card` answers an out-of-scope file
  exactly like a nonexistent one.
- `mcp_server.py` exposes the same four tools over stdio (MCP Python SDK 2.x),
  read-only, as one fixed user.

## Access control (`05_access/`)

Users map to roles; a role lists the paper groups and the tools it may use.
Enforcement is in the data path, never in the prompt: search prefilters, tool
allowlists, scoped counts and lists. Because an out-of-scope question finds
nothing similar inside the scope, it usually ends at "Not in the library."
without a model call, and a restricted user can't learn that a paper exists.

## Cost control (`06_cost/`)

- `pricing.py`: USD from the four token kinds; unknown models are "unknown",
  never guessed. Each answer also gets a reference cost on a chosen paid model.
- `ledger.py`: per-user daily token totals; checked before any model call.
- `answer_cache.py`: exact-repeat cache. The key is the normalised question,
  the role's scope (filter and tools), provider, model, a fingerprint of the
  answer-shaping settings and the index version, so it can't serve across
  scopes and invalidates itself when prompts or the index change. First turns
  only; no paraphrase matching (a semantic cache can return a wrong answer for
  a similar-looking question).

## Tracing (`07_observe/`)

`tracer.py` writes one row per request to SQLite: user, role, path, provider,
model, cache hit, budget block, skipped model, top similarity, retrieved
files, tool calls with arguments, model turns, the four token counts, cost,
reference cost, latency, stop reason, provider request id, error, answer.
`report.py` prints breakdowns by source, path, role and model, the latest
requests, or one request in full. Eval results carry their trace id.

## Configuration

`_shared/config/settings.yaml` holds the public defaults (the demo corpus,
demo users and roles, providers, token budget, prompts).
`_shared/config/settings.local.yaml` (gitignored) replaces whole sections for
a private deployment. API keys live in `_shared/config/secrets.env`
(gitignored).
