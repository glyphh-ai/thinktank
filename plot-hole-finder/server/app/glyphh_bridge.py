"""Bridge to the standalone glyphh engine (../../engine).

The engine is a sibling package, not pip-installed, so we add it to sys.path
once here and re-export it. Keeping this in one module means callers just do
``from .glyphh_bridge import glyphh``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ENGINE_PATH = Path(__file__).resolve().parents[2] / "engine"
if str(_ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(_ENGINE_PATH))

import glyphh  # noqa: E402

__all__ = ["glyphh"]
