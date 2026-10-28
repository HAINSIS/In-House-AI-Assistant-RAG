"""Usage -> dollars.

Prices are USD per million tokens, per model, from settings (`cost.prices`).
A model without a price returns None ("unknown"), never a guess. Every answer
also gets a reference cost: what the same tokens would cost on
`cost.reference_model`, so free-provider runs still show what they'd cost on
a paid model.
"""
from __future__ import annotations


def cost_usd(usage, model: str, prices: dict) -> float | None:
    p = prices.get(model)
    if not p:
        return None
    cache_write = p.get("cache_write", p["input"])
    return (usage.input_tokens * p["input"]
            + usage.cache_read_input_tokens * p["cached_input"]
            + usage.cache_creation_input_tokens * cache_write
            + usage.output_tokens * p["output"]) / 1_000_000


def total_tokens(usage) -> int:
    return (usage.input_tokens + usage.cache_read_input_tokens
            + usage.cache_creation_input_tokens + usage.output_tokens)
