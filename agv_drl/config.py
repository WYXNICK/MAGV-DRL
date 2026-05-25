from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


COURSE_MAP_PATH = Path(__file__).resolve().parents[1] / "maps" / "course_map.txt"
TRAIN_MAPS_DIR = Path(__file__).resolve().parents[1] / "maps" / "train_maps"


@dataclass(frozen=True)
class WarehouseConfig:
    """Fixed experiment setting used for fair 5000-step comparisons."""

    n_agents: int = 30
    horizon: int = 5000
    seed: int = 0
    request_queue_size: int = 10
    sensor_range: int = 2
    dynamic_cost_weight: float = 0.75
    layout_file: Optional[Path] = COURSE_MAP_PATH
    max_inactivity_steps: Optional[int] = None


@dataclass(frozen=True)
class RewardConfig:
    pickup: float = 0.15
    delivered_to_goal: float = 0.40
    completed_return: float = 1.00
    progress_scale: float = 0.015
    blocked_forward: float = -0.02
    wrong_toggle: float = -0.01
    step: float = -0.001


DEFAULT_RESULTS_DIR = Path("artifacts") / "agv_runs"
