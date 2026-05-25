from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..env.task_env import AGVTaskEnv, Phase, Coord, manhattan

from rware.warehouse import Direction  # noqa: E402


PATH_RADIUS = 5
PATH_DIM = (2 * PATH_RADIUS + 1) ** 2
FEATURE_DIM = 33 + PATH_DIM


def build_agent_features(
    task_env: AGVTaskEnv,
    reference_paths: Sequence[Sequence[Coord]] | None = None,
) -> np.ndarray:
    env = task_env.env
    height, width = env.grid_size
    max_dist = float(height + width)
    rows: List[np.ndarray] = []
    for agent in env.agents:
        task = task_env.manager.task_for_agent(agent.id)
        phase = task.phase if task is not None else Phase.IDLE
        # Keep the follower local, as in Learn-to-Follow: the policy should
        # follow the planner-provided path overlay instead of learning a
        # separate global target-to-target navigation rule.
        rel_x = rel_y = target_dx = target_dy = target_dist = on_target = 0.0

        direction = one_hot(agent.dir.value, 4)
        phase_vec = one_hot(phase.value, 4)
        local = local_occupancy(task_env, agent.id)
        path = [] if reference_paths is None else list(reference_paths[agent.id - 1])
        waypoint = next_waypoint(path)
        wx = wy = wdist = on_waypoint = 0.0
        if waypoint is not None:
            wx = (waypoint[0] - agent.x) / max(1.0, width - 1)
            wy = (waypoint[1] - agent.y) / max(1.0, height - 1)
            wdist = manhattan((agent.x, agent.y), waypoint) / max_dist
            on_waypoint = float((agent.x, agent.y) == waypoint)
        path_overlay = local_path_overlay(agent.x, agent.y, path)
        row = np.array(
            [
                rel_x,
                rel_y,
                float(agent.carrying_shelf is not None),
                float(env._is_highway(agent.x, agent.y)),
                target_dx,
                target_dy,
                target_dist,
                on_target,
                len(task_env.manager.pending) / max(1.0, task_env.config.request_queue_size),
                wx,
                wy,
                wdist,
                on_waypoint,
            ],
            dtype=np.float32,
        )
        rows.append(
            np.concatenate([row, direction, phase_vec, local, path_overlay]).astype(np.float32)
        )
    result = np.stack(rows)
    if result.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Feature dimension mismatch: got {result.shape[1]}, expected {FEATURE_DIM}")
    return result


def next_waypoint(path: Sequence[Coord]) -> Coord | None:
    if len(path) >= 2:
        return path[1]
    if len(path) == 1:
        return path[0]
    return None


def local_path_overlay(agent_x: int, agent_y: int, path: Sequence[Coord]) -> np.ndarray:
    # Same signal used by the original Learn-to-Follow preprocessor: nearby path
    # cells are injected into the local observation and path[1] is the next subgoal.
    overlay = np.zeros((2 * PATH_RADIUS + 1, 2 * PATH_RADIUS + 1), dtype=np.float32)
    for idx, (x, y) in enumerate(path):
        dx = x - agent_x
        dy = y - agent_y
        if abs(dx) > PATH_RADIUS or abs(dy) > PATH_RADIUS:
            continue
        value = 1.0 if idx == 1 else 0.5
        overlay[dy + PATH_RADIUS, dx + PATH_RADIUS] = max(overlay[dy + PATH_RADIUS, dx + PATH_RADIUS], value)
    return overlay.reshape(-1)


def one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def local_occupancy(task_env: AGVTaskEnv, agent_id: int) -> np.ndarray:
    env = task_env.env
    agent = env.agents[agent_id - 1]
    checks = [
        agent.dir,
        turn_left(agent.dir),
        turn_right(agent.dir),
        opposite(agent.dir),
    ]
    values = []
    for direction in checks:
        dx, dy = direction_delta(direction)
        x, y = agent.x + dx, agent.y + dy
        out = not (0 <= x < env.grid_size[1] and 0 <= y < env.grid_size[0])
        has_agent = 0.0 if out else float(env.grid[0, y, x] > 0)
        has_shelf = 0.0 if out else float(env.grid[1, y, x] > 0)
        values.extend([float(out), has_agent, has_shelf])
    return np.array(values, dtype=np.float32)


def direction_delta(direction: Direction):
    if direction == Direction.UP:
        return 0, -1
    if direction == Direction.DOWN:
        return 0, 1
    if direction == Direction.LEFT:
        return -1, 0
    return 1, 0


def turn_left(direction: Direction) -> Direction:
    order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    return order[(order.index(direction) - 1) % len(order)]


def turn_right(direction: Direction) -> Direction:
    order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    return order[(order.index(direction) + 1) % len(order)]


def opposite(direction: Direction) -> Direction:
    order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    return order[(order.index(direction) + 2) % len(order)]
