"""Start the MCP server over stdio the way an MCP client does, list its tools
and call two of them. Exit 1 if anything fails.

    python 04_tools/check_mcp.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

SERVER = Path(__file__).resolve().parent / "mcp_server.py"


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with Client(params, read_timeout_seconds=120) as client:
        listed = await client.list_tools()
        names = sorted(t.name for t in listed.tools)
        print("tools:", names)
        count = await client.call_tool("count_papers", {"year": 2019})
        count_text = " ".join(getattr(c, "text", "") for c in count.content)
        print("count_papers(year=2019):", count.structured_content or count_text)
        found = await client.call_tool("search_papers", {"query": "pedestrian crossing intent PIE JAAD", "k": 3})
        found_text = " ".join(getattr(c, "text", "") for c in found.content)
        print("search_papers:", (str(found.structured_content) if found.structured_content else found_text)[:400])
        # Any library works: a count must be a number, search must return titles.
        count_body = (count_text or str(count.structured_content)).replace("'", '"')
        ok = names == ["count_papers", "list_values", "read_card", "search_papers"] \
            and not count.is_error and not found.is_error \
            and re.search(r'"count": \d+', count_body) is not None \
            and "title" in (found_text or str(found.structured_content))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(asyncio.run(main()))
