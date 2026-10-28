"""Who may see what: user -> role -> scope (researches + tools).

Enforced in data access, never in the prompt: the scope becomes a filter on
the index (search and tools only ever see allowed papers) and an allowlist of
tool specs (the model is never shown a tool its role can't use; the executor
refuses it anyway). A paper outside the scope is indistinguishable from a
paper that doesn't exist: the answer is "Not in the library.", so a restricted
user can't even learn that it exists.
"""
from __future__ import annotations

from dataclasses import dataclass

ALL = "all"


class AccessError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    user: str
    role: str
    researches: frozenset[str] | None   # None = every research, including unassigned
    tools: frozenset[str] | None        # None = every tool

    def sees(self, research: str) -> bool:
        return self.researches is None or research in self.researches

    def may_use(self, tool: str) -> bool:
        return self.tools is None or tool in self.tools

    @property
    def where(self) -> str | None:
        """SQL filter for the index (LanceDB `where`), or None for no filter."""
        if self.researches is None:
            return None
        if not self.researches:
            return "false"
        quoted = ", ".join("'" + r.replace("'", "''") + "'" for r in sorted(self.researches))
        return f"research IN ({quoted})"


def resolve(settings: dict, user: str | None = None) -> Principal:
    acfg = settings["access"]
    user = user or acfg["default_user"]
    if user not in acfg["users"]:
        raise AccessError(f"unknown user {user!r}; known: {', '.join(acfg['users'])}")
    role = acfg["users"][user]
    if role not in acfg["roles"]:
        raise AccessError(f"user {user!r} has unknown role {role!r}")
    r = acfg["roles"][role]

    def scope(value) -> frozenset[str] | None:
        return None if value == ALL else frozenset(value or [])

    return Principal(user, role, scope(r.get("researches", [])), scope(r.get("tools", [])))
