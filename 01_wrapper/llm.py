"""The one place that calls a language model, for any provider.

Every later layer (retrieval, tools, cost, tracing) goes through an
`LLMClient` from `make_client()` instead of touching a provider SDK, so
provider choice, retries, errors and usage accounting live in this folder.

Plain data in, plain data out: `stream()` takes a provider-neutral history
(a list of {"role": "user" | "assistant", "content": str}) and returns a
`Reply`. History is text only on purpose: it is the shape every provider
accepts, and it never replays reasoning tokens from earlier turns.

Tool use, provider-neutral (each adapter translates):
  tools:       [{"name", "description", "parameters": <JSON schema>}]
  reply:       Reply.tool_calls = [ToolCall(id, name, arguments)]
  history:     {"role": "assistant", "content": str, "tool_calls": [...],
                "provider_raw": <adapter-private, optional>}
               {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Within one tool loop the assistant turn is passed back with `provider_raw`
so a provider that needs its own blocks back (Claude's thinking blocks before
tool results) gets them unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Protocol


class LLMError(RuntimeError):
    """A failure the user can act on, with a readable message."""


@dataclass
class Usage:
    input_tokens: int = 0              # uncached input
    output_tokens: int = 0             # includes reasoning tokens where the provider bills them
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None      # None when the model's JSON couldn't be parsed
    raw_arguments: str = ""

    def as_message(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Reply:
    text: str
    provider: str
    model: str                         # the model that served the turn (can differ after a fallback)
    stop_reason: str | None            # normalised: end_turn | max_tokens | refusal | other provider value
    usage: Usage
    latency_ms: int
    request_id: str | None = None
    refusal_category: str | None = None
    fell_back: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider_raw: object = None       # adapter-private copy of the assistant turn
    extra: dict = field(default_factory=dict)

    def assistant_message(self) -> dict:
        """This turn as a history message, for continuing a tool loop."""
        msg: dict = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [c.as_message() for c in self.tool_calls]
            msg["provider_raw"] = self.provider_raw
        return msg

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class LLMClient(Protocol):
    provider: str
    model: str

    def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        on_text: Callable[[str], None] | None = None,
        tools: list[dict] | None = None,
    ) -> Reply: ...

    def list_models(self) -> list[str]: ...


def make_client(provider: str, provider_cfg: dict, model_cfg: dict, client_cfg: dict,
                require_model: bool = True) -> LLMClient:
    """Build the client for one provider entry from settings.yaml.

    provider_cfg: {kind, api_key_env, base_url?, model, use_proxy?, extra_body?}
    model_cfg:    {max_tokens, effort, cache, fallbacks}
    client_cfg:   {max_retries, timeout_s}
    """
    kind = provider_cfg.get("kind")
    model = provider_cfg.get("model")
    if not model and require_model:
        raise LLMError(
            f"Provider '{provider}' has no model set in _shared/config/settings.yaml. "
            f"See what it offers with: app/main.py --provider {provider} --list-models"
        )

    key_env = provider_cfg.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key and kind != "openai_compatible_local":
        raise LLMError(
            f"No API key for '{provider}'. Set {key_env} in the environment or in "
            "_shared/config/secrets.env."
        )

    common = dict(
        provider=provider,
        model=model,
        api_key=api_key,
        max_tokens=model_cfg["max_tokens"],
        # Busy free tiers need more patience than the global default.
        max_retries=provider_cfg.get("max_retries", client_cfg["max_retries"]),
        timeout_s=client_cfg["timeout_s"],
        use_proxy=provider_cfg.get("use_proxy", True),
    )
    if kind == "anthropic":
        from provider_anthropic import AnthropicClient

        return AnthropicClient(
            **common,
            effort=model_cfg.get("effort"),
            cache=model_cfg.get("cache", True),
            fallbacks=model_cfg.get("fallbacks", True),
        )
    if kind in ("openai_compatible", "openai_compatible_local"):
        from provider_openai import OpenAICompatibleClient

        return OpenAICompatibleClient(
            **common,
            base_url=provider_cfg.get("base_url"),
            extra_body=provider_cfg.get("extra_body") or {},
        )
    raise LLMError(f"Provider '{provider}' has unknown kind '{kind}'.")
