"""Any OpenAI-compatible chat-completions API through the `openai` SDK.

One adapter covers OpenAI, DeepSeek, Qwen (DashScope), Gemini's OpenAI
endpoint, Moonshot, Zhipu, OpenRouter and local Ollama: they differ only in
base_url, key and model name. Provider-specific knobs (reasoning effort,
thinking switches) go in the provider's `extra_body` in settings.yaml.

Caching on these providers is automatic prefix caching: keeping the system
prompt and early history byte-stable is what earns it. Cached input is
reported in prompt_tokens_details.cached_tokens (DeepSeek reports
prompt_cache_hit_tokens instead; both are read).
"""
from __future__ import annotations

import json
import time
from typing import Callable

import openai

from llm import LLMError, Reply, ToolCall, Usage

_STOP = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal",
         "tool_calls": "tool_use"}


def _to_openai(messages: list[dict]) -> list[dict]:
    """Neutral history -> OpenAI chat messages (tool calls and tool results)."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"],
                                                     "arguments": json.dumps(c["arguments"] or {})}}
                                       for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class OpenAICompatibleClient:
    def __init__(self, provider, model, api_key, max_tokens, max_retries, timeout_s,
                 use_proxy=True, base_url=None, extra_body=None):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        kwargs = dict(api_key=api_key or "unused", base_url=base_url,
                      max_retries=max_retries, timeout=timeout_s)
        if not use_proxy:
            # Domestic and local endpoints shouldn't go through the VPN proxy
            # (it has a data limit and adds latency).
            kwargs["http_client"] = openai.DefaultHttpxClient(trust_env=False)
        self._client = openai.OpenAI(**kwargs)

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except openai.APIError as e:
            raise LLMError(f"{self.provider}: could not list models: {e}") from e

    def stream(self, messages: list[dict], system: str | None = None,
               on_text: Callable[[str], None] | None = None,
               tools: list[dict] | None = None) -> Reply:
        msgs = ([{"role": "system", "content": system}] if system else []) + _to_openai(messages)
        params: dict = dict(
            model=self.model,
            messages=msgs,
            max_tokens=self.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            params["tools"] = [{"type": "function", "function": t} for t in tools]
            params["tool_choice"] = "auto"
        if self.extra_body:
            params["extra_body"] = self.extra_body

        start = time.perf_counter()
        parts: list[str] = []
        calls: dict[int, dict] = {}          # stream index -> {id, name, args}
        finish = None
        usage = None
        served_by = self.model
        request_id = None
        try:
            stream = self._client.chat.completions.create(**params)
            request_id = getattr(stream, "_request_id", None)
            for chunk in stream:
                served_by = chunk.model or served_by
                if chunk.usage:
                    usage = chunk.usage
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta and delta.content:
                        parts.append(delta.content)
                        if on_text:
                            on_text(delta.content)
                    for tc in (delta.tool_calls or []) if delta else []:
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                    if choice.finish_reason:
                        finish = choice.finish_reason
        except openai.AuthenticationError as e:
            raise LLMError(f"{self.provider} rejected the API key (401).") from e
        except openai.PermissionDeniedError as e:
            raise LLMError(f"{self.provider} key lacks permission: {e.message}") from e
        except openai.NotFoundError as e:
            raise LLMError(f"{self.provider}: unknown model or endpoint ({self.model}): {e.message}") from e
        except openai.RateLimitError as e:
            raise LLMError(f"{self.provider} rate limit or quota exhausted after retries: {e.message}") from e
        except openai.BadRequestError as e:
            raise LLMError(f"{self.provider} bad request: {e.message}") from e
        except openai.APIStatusError as e:
            raise LLMError(f"{self.provider} API error {e.status_code} after retries: {e.message}") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"Could not reach {self.provider}. Check base_url and use_proxy.") from e
        latency_ms = int((time.perf_counter() - start) * 1000)
        if not parts and not calls and finish not in _STOP:
            # e.g. Zhipu's finish_reason "network_error" after a 180 s stall:
            # an empty answer would be scored and shown as if it were real.
            raise LLMError(f"{self.provider} ended the stream with finish_reason={finish!r} "
                           "and no content; retry")

        tool_calls = []
        for i in sorted(calls):
            c = calls[i]
            try:
                args = json.loads(c["args"]) if c["args"].strip() else {}
            except json.JSONDecodeError:
                args = None
            tool_calls.append(ToolCall(c["id"] or f"call_{i}", c["name"], args, c["args"]))

        cached = 0
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", None) if details else None) \
                or getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        prompt = (usage.prompt_tokens or 0) if usage else 0
        stop = _STOP.get(finish, finish)
        if tool_calls and stop not in ("max_tokens", "refusal"):
            stop = "tool_use"     # some providers report "stop" even with tool calls
        return Reply(
            text="".join(parts),
            provider=self.provider,
            model=served_by,
            stop_reason=stop,
            usage=Usage(
                input_tokens=max(prompt - cached, 0),
                output_tokens=(usage.completion_tokens or 0) if usage else 0,
                cache_read_input_tokens=cached,
            ),
            latency_ms=latency_ms,
            request_id=request_id,
            tool_calls=tool_calls,
            extra={} if usage else {"usage_missing": True},
        )
