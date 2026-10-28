"""MCP server: the library tools for any MCP client (Claude Code, a
research agent, Claude Desktop), over stdio.

    python 04_tools/mcp_server.py

Same functions as the app's tool loop (library_tools.py), so both see the same
counts and search results. Search loads the embedding model on first use
(~10 s, ~2 GB GPU). Read-only: nothing here writes to the library or the index.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.getLogger("httpx").setLevel(logging.WARNING)   # quiet per-request logs on stderr

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "app")]

import _pathsetup  # noqa: E402,F401  (adds the layer folders, like the app does)
import pipeline  # noqa: E402
from mcp.server import MCPServer  # noqa: E402
from settings import load_settings  # noqa: E402

settings = load_settings()
tools = pipeline.library_tools(settings)
# The server runs as one user: INHOUSE_USER, else access.default_user. Its
# scope applies to every call (a client can't widen it).
who = pipeline.resolve(settings, os.environ.get("INHOUSE_USER"))
mcp = MCPServer("inhouse-ai-library")


def _call(name: str, **args) -> dict:
    return json.loads(tools.call(name, {k: v for k, v in args.items() if v is not None}, scope=who))


@mcp.tool()
def count_papers(year: int | None = None, year_before: int | None = None, year_after: int | None = None,
                 venue: str | None = None, subtheme: str | None = None) -> dict:
    """Count papers in the library matching all given filters. Venue and subtheme
    must be exact names (case-insensitive); use list_values if unsure."""
    return _call("count_papers", year=year, year_before=year_before, year_after=year_after,
                 venue=venue, subtheme=subtheme)


@mcp.tool()
def list_values(field: str, contains: str | None = None, limit: int = 20) -> dict:
    """List exact values of a metadata field (year, venue or subtheme) with paper counts."""
    return _call("list_values", field=field, contains=contains, limit=limit)


@mcp.tool()
def search_papers(query: str, k: int = 5) -> dict:
    """Search the paper library by meaning and keywords; returns the best-matching papers."""
    return _call("search_papers", query=query, k=k)


@mcp.tool()
def read_card(file: str) -> dict:
    """Read one paper's card (abstract, method, results) by file name from search_papers."""
    return _call("read_card", file=file)


if __name__ == "__main__":
    mcp.run()
