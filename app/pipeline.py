"""How one question becomes one answer, for every layer.

The CLI and the eval runner both call `answer()`, so the eval measures exactly
what the user gets. Layers are wired here, never inside each other.

    who = resolve(user)                       05_access
    over daily budget?   -> refusal, no model call          06_cost
    same question, same scope, same settings answered before? -> cached, 0 tokens
    "how many ..."       -> tool path (count/list/search/read)          04_tools
    else                 -> hybrid search in scope (03_retrieve);
                            nothing similar -> "Not in the library.", no model call;
                            else close-scoring top papers + question -> cited answer
    record tokens + cost in the ledger, store in the cache   06_cost
    one trace row per answer (or error)                      07_observe
"""
from __future__ import annotations

import re
import time
from typing import Callable

import _pathsetup  # noqa: F401
from access import Principal, resolve
from context import NOT_FOUND, build_prompt, sources_block
from llm import LLMClient, LLMError, Reply, Usage

LAYER = "07_observe"  # the highest layer wired in; recorded with every eval run

FINAL_TURN = "Answer the original question now, from the tool results above. Do not call any more tools."
# Tool calls some models write as text when no tools are offered (GLM's native format).
TOOL_MARKUP = re.compile(r"<tool_call>.*?(</tool_call>|$)", re.S)

_cache: dict[str, object] = {}


def _once(name: str, build):
    if name not in _cache:
        _cache[name] = build()
    return _cache[name]


def retriever(settings: dict):
    """Loaded on first use: the embedding model takes ~10 s and 2 GB of GPU memory."""
    def build():
        from embedder import Embedder
        from retriever import Retriever

        ix = settings["index"]
        emb = Embedder(ix["embed_model"], ix["device"], ix["batch_size"], ix.get("hf_endpoint"))
        return Retriever(_pathsetup.ROOT / ix["dir"], emb)
    return _once("retriever", build)


def library_tools(settings: dict):
    def build():
        from cards import parse
        from library_tools import LibraryTools

        corpus = settings["corpus"]
        papers = (_pathsetup.ROOT / corpus["root"] / corpus["papers_dir"]).resolve()
        return LibraryTools(_pathsetup.ROOT / settings["index"]["dir"], papers,
                            parse_card=parse, retriever_factory=lambda: retriever(settings))
    return _once("tools", build)


def ledger(settings: dict):
    from ledger import Ledger
    return _once("ledger", lambda: Ledger(_pathsetup.ROOT / settings["cost"]["db"]))


def answer_cache(settings: dict):
    from answer_cache import AnswerCache
    return _once("answer_cache", lambda: AnswerCache(_pathsetup.ROOT / settings["cost"]["db"],
                                                     settings["cost"]["answer_cache"]["ttl_hours"]))


def tracer(settings: dict):
    from tracer import Tracer
    return _once("tracer", lambda: Tracer(_pathsetup.ROOT / settings["observe"]["db"]))


def index_version(settings: dict) -> str:
    """Changes whenever 02_ingest runs: the last line of its run log."""
    log = _pathsetup.ROOT / settings["index"]["dir"] / "ingest_log.jsonl"
    try:
        return log.read_text(encoding="utf-8").strip().splitlines()[-1]
    except (OSError, IndexError):
        return "none"


def wants_tools(question: str, settings: dict, who: Principal) -> bool:
    from library_tools import specs_for

    return bool(specs_for(who)) and re.search(settings["tools"]["route"], question, re.I) is not None


def answer(
    client: LLMClient,
    question: str,
    settings: dict,
    history: list[dict] | None = None,
    on_text: Callable[[str], None] | None = None,
    user: str | None = None,
    source: str = "cli",
    use_cache: bool = True,
    enforce_budget: bool = True,
) -> Reply:
    """`use_cache` / `enforce_budget` off for evals: they measure the model,
    not the cache, and must not be cut off by a budget."""
    from answer_cache import fingerprint, make_key
    from library_tools import specs_for
    from pricing import cost_usd, total_tokens

    who = resolve(settings, user)
    ccfg = settings["cost"]
    start = time.perf_counter()
    trace = dict(source=source, user=who.user, role=who.role, question=question)

    def finish(reply: Reply) -> Reply:
        reply.extra.update(user=who.user, role=who.role)
        cost = cost_usd(reply.usage, reply.model, ccfg["prices"])
        ref = cost_usd(reply.usage, ccfg["reference_model"], ccfg["prices"]) if ccfg["reference_model"] else None
        reply.extra.update(cost_usd=cost, ref_cost_usd=ref)
        ledger(settings).record(who.user, total_tokens(reply.usage), cost)
        x = reply.extra
        reply.extra["trace_id"] = tracer(settings).record(
            **trace, path=x.get("path"), provider=reply.provider, model=reply.model,
            cache_hit=int(bool(x.get("cache_hit"))), budget_blocked=int(bool(x.get("budget_blocked"))),
            skipped_model=int(bool(x.get("skipped_model"))), top_similarity=x.get("top_similarity"),
            retrieved=x.get("retrieved"), tool_calls=x.get("tool_calls"), model_turns=x.get("model_turns"),
            input_tokens=reply.usage.input_tokens, cached_tokens=reply.usage.cache_read_input_tokens,
            cache_write_tokens=reply.usage.cache_creation_input_tokens, output_tokens=reply.usage.output_tokens,
            cost_usd=cost, ref_cost_usd=ref, latency_ms=reply.latency_ms, stop_reason=reply.stop_reason,
            request_id=reply.request_id, answer=reply.text)
        return reply

    def local(text: str, **extra) -> Reply:
        if on_text:
            on_text(text)
        return Reply(text=text, provider="local", model=extra.pop("model", "local"), stop_reason="end_turn",
                     usage=Usage(), latency_ms=int((time.perf_counter() - start) * 1000), extra=extra)

    # 06_cost: budget, before anything that costs tokens
    limit = ccfg["daily_tokens"].get(who.role)
    if enforce_budget:
        over, spent = ledger(settings).over_budget(who.user, limit)
        if over:
            return finish(local(f"Daily token budget reached for role {who.role} "
                                f"({spent:,} of {limit:,} tokens). It resets tomorrow.",
                                model="budget", path="budget", budget_blocked=True))

    # 06_cost: exact-repeat answer cache (first turns only; scope is part of the key)
    key = None
    if use_cache and ccfg["answer_cache"]["enabled"] and not history:
        scope = {"where": who.where, "tools": sorted(s["name"] for s in specs_for(who))}
        key = make_key(question, scope, client.provider, client.model, fingerprint(settings), index_version(settings))
        hit = answer_cache(settings).get(key)
        if hit:
            text, extra = hit
            reply = local(text, model="answer-cache", **{**extra, "cache_hit": True})
            reply.extra["sources_text"] = ""   # already part of the cached text
            return finish(reply)

    try:
        if wants_tools(question, settings, who):
            reply = _answer_with_tools(client, question, settings, history, on_text, who)
        else:
            reply = _answer_with_retrieval(client, question, settings, history, on_text, who)
    except LLMError as e:
        tracer(settings).record(**trace, error=str(e), latency_ms=int((time.perf_counter() - start) * 1000))
        raise

    if key and reply.stop_reason == "end_turn" and reply.text:
        keep = {k: reply.extra.get(k) for k in ("path", "retrieved", "top_similarity", "skipped_model")}
        answer_cache(settings).put(key, reply.text, keep)
    return finish(reply)


def _answer_with_retrieval(client, question, settings, history, on_text, who: Principal) -> Reply:
    rcfg = settings["retrieval"]
    start = time.perf_counter()
    hits = retriever(settings).search(question, rcfg["k"], where=who.where,
                                      relative_cutoff=rcfg.get("relative_cutoff"))

    if not hits or max(h.similarity for h in hits) < rcfg["min_similarity"]:
        if on_text:
            on_text(NOT_FOUND)
        return Reply(text=NOT_FOUND, provider="local", model="retrieval-threshold",
                     stop_reason="end_turn", usage=Usage(),
                     latency_ms=int((time.perf_counter() - start) * 1000),
                     extra={"path": "retrieval", "skipped_model": True,
                            "top_similarity": max((h.similarity for h in hits), default=0.0)})

    messages = (history or []) + [{"role": "user", "content": build_prompt(question, hits)}]
    reply = client.stream(messages, system=settings["retrieval"]["system_prompt"].strip(), on_text=on_text)
    block = sources_block(reply.text, hits)
    reply.text += block
    reply.extra.update(path="retrieval", sources_text=block, retrieved=[h.file for h in hits],
                       top_similarity=max(h.similarity for h in hits))
    reply.latency_ms = int((time.perf_counter() - start) * 1000)
    return reply


def _answer_with_tools(client, question, settings, history, on_text, who: Principal) -> Reply:
    from library_tools import specs_for

    specs = specs_for(who)
    tcfg = settings["tools"]
    tools = library_tools(settings)
    start = time.perf_counter()
    messages = (history or []) + [{"role": "user", "content": question}]
    total = Usage()
    calls_made: list[dict] = []
    reply = None
    for step in range(tcfg["max_steps"]):
        last = step == tcfg["max_steps"] - 1
        # The last turn gets no tools and an explicit instruction, so the loop
        # always ends in an answer (without it, GLM once wrote a tool call as text).
        turn = messages + ([{"role": "user", "content": FINAL_TURN}] if last and step else [])
        reply = client.stream(turn, system=tcfg["system_prompt"].strip(),
                              on_text=on_text, tools=None if last else specs)
        reply.text = TOOL_MARKUP.sub("", reply.text).strip() or reply.text
        total = total + reply.usage
        if not reply.tool_calls or reply.stop_reason in ("max_tokens", "refusal"):
            break
        messages.append(reply.assistant_message())
        for call in reply.tool_calls:
            if call.arguments is None:
                result = '{"error": "arguments were not valid JSON; call the tool again"}'
            else:
                result = tools.call(call.name, dict(call.arguments), scope=who)
            calls_made.append({"name": call.name, "arguments": call.arguments, "result": result[:300]})
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})

    reply.usage = total
    reply.latency_ms = int((time.perf_counter() - start) * 1000)
    reply.extra.update(path="tools", tool_calls=calls_made, model_turns=step + 1)
    return reply
