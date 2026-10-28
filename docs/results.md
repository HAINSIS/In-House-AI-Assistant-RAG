# Results

## Setup

- **Library:** 11,476 verified research-paper cards on driver behaviour,
  connected vehicles and intelligent transportation (~11.8M tokens; 4.6M after
  removing placeholder text). Private, so not included; the demo corpus
  reproduces the pipeline, not these numbers.
- **Answering model:** GLM-4.7-Flash (Zhipu, free tier), thinking off.
- **Embeddings:** bge-m3 on a local GPU; 12,939 chunks, built in 17 minutes.
- **Questions:** 100, written by hand from the library (never by the model under
  test), split 61 dev / 39 test. Settings were chosen on dev; test was run once.
- **Scoring:** deterministic code, no model judge (see [Scoring](#scoring)).

## Accuracy

| Kind | What it checks | Test | Dev |
|---|---|---|---|
| lookup | find one paper from a paraphrase of its abstract | 16 / 16 | 24 / 25 |
| count | number of papers matching year / venue / topic filters | 7 / 7 | 10 / 10 |
| unanswerable | topic absent from the library: must decline, name no paper | 6 / 6 | 10 / 10 |
| forbidden | asked by a user who may not see the answer: must not leak | 3 / 3 | 5 / 5 |
| multi | a claim several papers support: must name at least two of them | 4 / 7 | 5 / 11 |
| **all** | | **36 / 39 (92%)** | **54 / 61 (89%)** |

Access questions (asked as restricted users) also include in-scope lookups and
counts within the user's scope; all of them pass. **No answer, in any kind,
named a paper outside the asker's scope (0 leaks in 100 answers).**

## Tokens, cost and latency

| | Test | Dev |
|---|---|---|
| Input tokens (uncached + cached) | 29,444 + 10,498 | 49,664 + 14,126 |
| Output tokens | 1,191 | 2,131 |
| Tokens per question | 1,055 | 1,081 |
| Per answer that called the model: input / output | 1,210 / 36 | 1,276 / 43 |
| Answered with no model call | 6 | 11 |
| Answered through tools | 7 | 10 |
| Sources sent per retrieval answer | 3.3 | 3.7 |
| Cost at Claude Opus 5.5 list prices | $0.144 (≈ $0.0037 / question) | $0.244 (≈ $0.0040 / question) |
| Latency p50 / p95 | 1.5 s / 16.2 s | 1.9 s / 23.4 s |

The answering model is free, so the dollar figures price the same tokens at
Claude Opus 5.5 list prices ($4 / $0.20 / $20 per million input / cached input /
output tokens). The p95 latency is dominated by the free tier's "server busy"
retries, not by the system.

For comparison, putting the cleaned library into the prompt would be ~4.6M
input tokens per question (≈ $18 at the same prices), and no model's context
window holds it.

## Where the tokens are saved

| Mechanism | Effect |
|---|---|
| Placeholder text removed at indexing | 61% less text in the index; every retrieved chunk is shorter |
| Only papers scoring within 0.08 of the best match are sent | 3.3–3.7 sources instead of 5; ~27% fewer input tokens per question, same lookup accuracy |
| No model call when nothing in scope is similar enough (cosine < 0.55) | out-of-library questions cost 0 tokens |
| Tools offered only for counting questions | other questions never pay for tool definitions or a second round trip |
| Citations as `[n]`, titles written by the app | ~15 output tokens saved per citation, no misspelled titles |
| Text-only history, trimmed in halves | earlier reasoning never re-sent; the cached prefix survives between trims |
| Exact-repeat cache keyed by scope | a repeated question costs 0 tokens (~4 ms) and is never served across scopes |
| Thinking off for this model | 39 vs 558 output tokens on the same one-sentence answer |

## Failures

- **multi (3 test, 6 dev).** The reference lists come from literature
  syntheses and are not exhaustive, and the failures split three ways:
  - *No reference paper retrieved* (2 dev). In one, the model cited five
    other papers that do support the claim; in the other it answered "Not in
    the library".
  - *Only one reference paper retrieved* (2 dev, 3 test), so naming two was
    impossible. One of these (test M02) is caused by the source cutoff: the
    question asked for papers in the plural and only one source scored within
    the cutoff. Keeping at least two sources for plural questions is the
    candidate fix; it has to be validated on new questions, since this failure
    was seen on test.
  - *The model named fewer than it was given* (2 dev): two reference papers
    were among the sources, the answer named one.
- **lookup (1 dev).** The question fits several surveys on the same topic; the
  expected one was not among the sources and the model cited two others.
- Run-to-run spread: an accidental second test run of the same configuration
  scored 35 / 39, with the same multi failures.

## Scoring

- A paper counts as named when the answer contains its full title or DOI, or
  80% of the title's distinctive words (words already in the question don't
  count, so echoing the question can't match an unrelated paper).
- A count must appear as the exact number. Expected counts are computed from
  the library at run time and stored with each result, so later library edits
  don't change past verdicts.
- A decline must match a decline phrase and name no library paper.
- Every answer is checked for leaks: naming any paper outside the asker's scope
  fails it, whatever the question.
