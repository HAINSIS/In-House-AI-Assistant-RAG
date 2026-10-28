"""Loads config/settings.yaml, deep-merged over defaults so the file can be
hand-edited and left partial, then config/settings.local.yaml if it exists
(gitignored): each top-level section in it REPLACES that section, so a private
deployment (its own corpus, users and roles) never mixes with the public demo
defaults. Also loads config/secrets.env into the environment (never
overriding a variable that is already set).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"

_DEFAULTS: dict = {
    "active_provider": "auto",
    "providers": {},
    "model": {"max_tokens": 4000, "effort": "low", "cache": True, "fallbacks": True},
    "client": {"max_retries": 3, "timeout_s": 600},
    "history": {"max_turns": 6},
    "retrieval": {"k": 5, "min_similarity": 0.55, "relative_cutoff": None, "system_prompt": ""},
    "cost": {"db": "data/inhouse.sqlite", "prices": {}, "reference_model": None,
             "answer_cache": {"enabled": False, "ttl_hours": 168}, "daily_tokens": {}},
    "observe": {"db": "data/inhouse.sqlite"},
    "tools": {"route": r"\b(how many|number of|count)\b", "max_steps": 4, "system_prompt": ""},
    "access": {"default_user": "owner", "users": {"owner": "owner"},
               "roles": {"owner": {"researches": "all", "tools": "all"}}},
    "index": {"dir": "data/index", "embed_model": "BAAI/bge-m3", "device": "auto", "batch_size": 16,
              "cards_per_write": 500, "max_chunk_words": 300, "hf_endpoint": ""},
    "system_prompt": "",
    "corpus": {"root": "data/demo", "papers_dir": "", "researches_dir": "", "include": [], "exclude": []},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(path: Path | None = None, local: Path | None = None) -> dict:
    path = path or CONFIG_DIR / "settings.yaml"
    settings = copy.deepcopy(_DEFAULTS)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            settings = _merge(settings, yaml.safe_load(f) or {})
    local = local or CONFIG_DIR / "settings.local.yaml"
    if local.exists():
        with open(local, encoding="utf-8") as f:
            for section, value in (yaml.safe_load(f) or {}).items():
                base = _DEFAULTS.get(section)
                settings[section] = _merge(base, value) if isinstance(base, dict) and isinstance(value, dict) else value
    return settings


def pick_provider(settings: dict, requested: str | None = None) -> str:
    """The provider to use: `requested`, else active_provider, else (auto) the
    first configured provider with a model set and its API key present."""
    providers = settings["providers"]
    name = requested or settings["active_provider"]
    if name != "auto":
        if name not in providers:
            raise KeyError(f"Unknown provider '{name}'. Configured: {', '.join(providers)}")
        return name
    for name, cfg in providers.items():
        # Local servers need no key, so they're only used when asked for by name.
        if cfg.get("kind") == "openai_compatible_local" or not cfg.get("model"):
            continue
        if os.environ.get(cfg.get("api_key_env", "")):
            return name
    raise KeyError(
        "No provider has an API key set. Put one of these in _shared/config/secrets.env: "
        + ", ".join(c["api_key_env"] for c in providers.values()
                    if c.get("api_key_env") and c.get("kind") != "openai_compatible_local")
    )


def load_secrets(path: Path | None = None) -> None:
    """KEY=VALUE lines from secrets.env into os.environ; '#' starts a comment."""
    path = path or CONFIG_DIR / "secrets.env"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
