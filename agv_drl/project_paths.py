from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RWARE_ROOT = ROOT / "robotic-warehouse"


def add_local_repos_to_path() -> None:
    """Make the bundled RWARE repository importable without install."""
    for path in (RWARE_ROOT,):
        value = str(path)
        if path.exists() and value not in sys.path:
            sys.path.insert(0, value)
