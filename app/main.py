"""inhouse-ai CLI.

    python app/main.py                         interactive chat
    python app/main.py "question"              one-shot answer, then exit
    python app/main.py --provider deepseek ... use a specific provider

In chat: /reset starts a new conversation, /quit (or Ctrl+C) exits.
After each answer a status line shows provider/model, tokens (incl. cache),
latency, and the session's running token total.
"""
from __future__ import annotations

import argparse
import sys

import _pathsetup  # noqa: F401  (must run first: adds layer folders to sys.path)
import pipeline
from llm import LLMClient, LLMError, Reply, Usage, make_client
from settings import load_secrets, load_settings, pick_provider


class Conversation:
    """Text-only history with a turn cap.

    When the cap is exceeded the oldest half is dropped in one go: dropping
    one turn per question would change the prompt prefix every time and
    defeat prompt caching.
    """

    def __init__(self, max_turns: int):
        self.max_turns = max(1, max_turns)
        self.messages: list[dict] = []

    def add(self, question: str, answer: str) -> None:
        self.messages += [{"role": "user", "content": question},
                          {"role": "assistant", "content": answer}]
        turns = len(self.messages) // 2
        if turns > self.max_turns:
            keep = max(1, self.max_turns // 2)
            self.messages = self.messages[-2 * keep:]

    def clear(self) -> None:
        self.messages.clear()


def status_line(reply: Reply, total: Usage) -> str:
    u = reply.usage
    parts = [
        f"{reply.provider}/{reply.model}",
        f"in {u.input_tokens} + cached {u.cache_read_input_tokens}",
        f"out {u.output_tokens}",
        f"{reply.latency_ms / 1000:.1f}s",
        f"session {total.input_tokens + total.cache_read_input_tokens + total.cache_creation_input_tokens}"
        f" in / {total.output_tokens} out",
    ]
    if u.cache_creation_input_tokens:
        parts.insert(2, f"cache write {u.cache_creation_input_tokens}")
    if reply.fell_back:
        parts.append("fallback")
    if reply.stop_reason not in ("end_turn", None):
        parts.append(f"stop: {reply.stop_reason}")
    if reply.extra.get("usage_missing"):
        parts.append("provider sent no usage")
    if reply.extra.get("path") == "tools":
        called = ", ".join(f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in (c['arguments'] or {}).items())})"
                           for c in reply.extra["tool_calls"]) or "none"
        parts.append(f"{reply.extra['model_turns']} model turns, tools: {called}")
    if reply.extra.get("cost_usd") is not None or reply.extra.get("ref_cost_usd") is not None:
        cost = reply.extra.get("cost_usd")
        ref = reply.extra.get("ref_cost_usd")
        parts.append(f"${cost:.4f}" if cost is not None else "cost unknown")
        if ref is not None and reply.provider != "local":
            parts.append(f"≈${ref:.4f} on Opus 5.5")
    if reply.extra.get("cache_hit"):
        parts = ["answer cache: 0 tokens", f"{reply.latency_ms / 1000:.1f}s"]
    if reply.extra.get("budget_blocked"):
        parts = ["daily budget reached: no model call"]
    if reply.extra.get("skipped_model"):
        parts = [f"no model call: nothing in the library is similar enough "
                 f"(top similarity {reply.extra['top_similarity']:.2f})", f"{reply.latency_ms / 1000:.1f}s"]
    return "[" + " | ".join(parts) + "]"


def ask(client: LLMClient, convo: Conversation, settings: dict, question: str, total: list[Usage],
        user: str | None = None) -> Reply:
    reply = pipeline.answer(client, question, settings, history=convo.messages,
                            on_text=lambda t: print(t, end="", flush=True), user=user)
    print(reply.extra.get("sources_text", ""))
    if reply.refused:
        print(f"(declined{', category: ' + reply.refusal_category if reply.refusal_category else ''})")
    else:
        convo.add(question, reply.text)
    total[0] = total[0] + reply.usage
    print(status_line(reply, total[0]))
    return reply


def main(argv: list[str]) -> int:
    # Windows consoles default to cp1252; answers contain Chinese and symbols.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="inhouse-ai")
    parser.add_argument("--provider", help="provider name from settings.yaml (default: active_provider)")
    parser.add_argument("--user", help="ask as this user (access.users in settings.yaml; default access.default_user)")
    parser.add_argument("--list-models", action="store_true",
                        help="print the model ids the provider offers, then exit")
    parser.add_argument("question", nargs="*", help="ask once and exit")
    args = parser.parse_args(argv)

    load_secrets()
    settings = load_settings()
    try:
        from access import resolve
        who = resolve(settings, args.user)   # fail fast on an unknown user
        name = pick_provider(settings, args.provider)
        client = make_client(name, settings["providers"][name], settings["model"], settings["client"],
                             require_model=not args.list_models)
        if args.list_models:
            print("\n".join(client.list_models()))
            return 0
    except (KeyError, ValueError, LLMError) as e:
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 1

    convo = Conversation(settings["history"]["max_turns"])
    total = [Usage()]
    if args.question:
        try:
            ask(client, convo, settings, " ".join(args.question), total, args.user)
        except LLMError as e:
            print(f"\nerror: {e}", file=sys.stderr)
            return 1
        return 0

    print(f"inhouse-ai ({name}/{client.model}, user {who.user} [{who.role}]). /reset, /quit")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in ("/quit", "/exit"):
            return 0
        if question == "/reset":
            convo.clear()
            print("(new conversation)")
            continue
        try:
            ask(client, convo, settings, question, total, args.user)
        except LLMError as e:
            print(f"\nerror: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(interrupted)")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
