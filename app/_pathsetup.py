"""Adds each numbered layer folder (and _shared) to sys.path so their modules
import by plain name (e.g. `import llm`). Folder names like "01_wrapper" are
not valid Python identifiers, so they can't be imported as packages.

Import this before any layer import. Add a layer here when its folder exists.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for _layer in ("01_wrapper", "02_ingest", "03_retrieve", "04_tools", "05_access", "06_cost",
               "07_observe", "_shared"):
    _path = str(ROOT / _layer)
    if _path not in sys.path:
        sys.path.insert(0, _path)
