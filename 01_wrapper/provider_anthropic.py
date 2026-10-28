"""Claude through the native Anthropic SDK.

Native rather than OpenAI-compatible because two token and reliability
features only exist here: explicit prompt caching (cache reads cost ~0.1x)
and the server-side refusal fallback.
"""
from __future__ import annotations

import time
from typing import Callable

import anthropic

from llm import LLMError, Reply, ToolCall, Usage

# Server-side refusal fallback: on a policy decline the API re-runs the same
# request on a fallback model inside the same call, routed by refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _to_anthropic(messages: list[dict]) -> list[dict]:
    """Neutral history -> Messages API turns. Within a tool loop the assistant
    turn goes back as its own content blocks (thinking included, unchanged);
    consecutive tool results become one user turn of tool_result blocks."""
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            blocks = m.get("provider_raw") or (
                ([{"type": "text", "text": m["content"]}] if m.get("content") else [])
                + [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"] or {}}
                   for c in m["tool_calls"]])
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class AnthropicClient:
    def __init__(self, provider, model, api_key, max_tokens, max_retries, timeout_s,
                 use_proxy=True, effort="low", cache=True, fallbacks=True):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.cache = cache
        self.fallbacks = fallbacks
        kwargs = dict(api_key=api_key, max_retries=max_retries, timeout=timeout_s)
        if not use_proxy:
            kwargs["http_client"] = anthropic.DefaultHttpxClient(trust_env=False)
        self._client = anthropic.Anthropic(**kwargs)

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except anthropic.APIError as e:
            raise LLMError(f"{self.provider}: could not list models: {e}") from e

    def stream(self, messages: list[dict], system: str | None = None,
               on_text: Callable[[str], None] | None = None,
               tools: list[dict] | None = None) -> Reply:
        params: dict = {"model": self.model, "max_tokens": self.max_tokens,
                        "messages": _to_anthropic(messages)}
        if tools:
            # tool_choice stays "auto": Opus 5.5 rejects forced tool choice.
            # Inputs are validated by the tool executor (04_tools), not here.
            params["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
        if self.effort:
            # Effort is the main lever on thinking tokens; Opus 5.5 rejects
            # disabling thinking, so low effort is how to keep it short.
            params["output_config"] = {"effort": self.effort}
        if system:
            params["system"] = system
        if self.cache:
            # Automatic caching of the last cacheable block: the stable prefix
            # (system + earlier turns) is billed as a cache read next turn.
            params["cache_control"] = {"type": "ephemeral"}
        if self.fallbacks:
            api = self._client.beta.messages
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        else:
            api = self._client.messages

        start = time.perf_counter()
        try:
            with api.stream(**params) as stream:
                for text in stream.text_stream:
                    if on_text:
                        on_text(text)
                final = stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic rejected the API key (401).") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"The Anthropic key lacks permission: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Unknown Anthropic model or endpoint ({self.model}): {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Anthropic rate limit after retries; wait and try again.") from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"Anthropic bad request: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code} after retries: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach api.anthropic.com. Is the proxy running?") from e
        latency_ms = int((time.perf_counter() - start) * 1000)

        u = final.usage
        refusal_category = None
        if final.stop_reason == "refusal" and getattr(final, "stop_details", None):
            refusal_category = final.stop_details.category
        return Reply(
            text="".join(b.text for b in final.content if b.type == "text"),
            provider=self.provider,
            model=final.model,
            stop_reason=final.stop_reason,
            usage=Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
            request_id=getattr(final, "_request_id", None),
            refusal_category=refusal_category,
            fell_back=any(getattr(e, "type", None) == "fallback_message"
                          for e in (getattr(u, "iterations", None) or [])),
            tool_calls=[ToolCall(b.id, b.name, b.input if isinstance(b.input, dict) else None)
                        for b in final.content if b.type == "tool_use"],
            provider_raw=[b.model_dump(exclude_none=True) for b in final.content],
        )
