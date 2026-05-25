from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def create_run_dir(
    kind: str,
    runs_root: Path,
    run_name: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """Create a fresh output directory for one training/evaluation/demo run."""

    if run_dir is not None:
        return _create_unique(Path(run_dir))
    prefix = _safe_name(run_name or kind)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _create_unique(Path(runs_root) / f"{prefix}_{stamp}")


def checkpoint_owner_dir(checkpoint: Path) -> Path:
    """Return the train run directory that owns a checkpoint path."""

    checkpoint = Path(checkpoint)
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def default_eval_root(checkpoint: Optional[Path], runs_root: Path) -> Path:
    if checkpoint is None:
        return Path(runs_root)
    return checkpoint_owner_dir(checkpoint) / "evals"


def find_latest_checkpoint(runs_root: Path) -> Optional[Path]:
    root = Path(runs_root)
    candidates = list(root.glob("*/checkpoints/ltfp_latest.pt"))
    legacy = root / "checkpoints" / "ltfp_latest.pt"
    if legacy.exists():
        candidates.append(legacy)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _create_unique(base: Path) -> Path:
    parent = base.parent
    parent.mkdir(parents=True, exist_ok=True)
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base.name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _safe_name(value: str) -> str:
    allowed = []
    for char in value.strip():
        if char.isalnum() or char in ("-", "_"):
            allowed.append(char)
        else:
            allowed.append("_")
    result = "".join(allowed).strip("_")
    return result or "run"
