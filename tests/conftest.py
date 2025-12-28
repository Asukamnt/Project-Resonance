"""Pytest configuration.

Pytest 9 defaults to importlib-based importing, which does not automatically add the
repository root (where `train.py` lives) onto `sys.path`.

Some tests intentionally import the top-level `train` module, so we prepend the
project root to `sys.path` to make those imports stable across pytest versions.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_syspath() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_project_root_on_syspath()


