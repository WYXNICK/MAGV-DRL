from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..env.task_env import AGVTaskEnv, Coord, Phase, manhattan

from rware.warehouse import Action, Agent, Direction  # noqa: E402


DIR_TO_DELTA = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

DELTA_TO_DIR = {v: k for k, v in DIR_TO_DELTA.items()}
TURN_ORDER = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]


@dataclass
class Reservation:
    cells: Set[Coord]
    moves: Set[Tuple[Coord, Coord]]


class PrioritizedAStarController:
    """Long-horizon planner and warm-start oracle for the LTF-PPO local follower."""

    def __init__(self, env: AGVTaskEnv, reserve_dynamic: bool = True):
        self.task_env = env
        self.last_paths: Dict[int, List[Coord]] = {}
        self.dynamic_costs: Dict[int, np.ndarray] = {}
        self.dynamic_targets: Dict[int, Coord] = {}
        self.dynamic_weight = float(env.config.dynamic_cost_weight)
        self.reserve_dynamic = reserve_dynamic

    def act(self) -> List[int]:
        manager = self.task_env.manager
        manager.assign_idle_agents()
        reservation = Reservation(cells=set(), moves=set())
        actions = [Action.NOOP.value for _ in self.task_env.env.agents]
        self.last_paths = {}
        ordered_agents = sorted(
            self.task_env.env.agents,
            key=lambda a: self._priority(a),
            reverse=True,
        )
        for agent in ordered_agents:
            self._update_dynamic_cost(agent)
            active_reservation = reservation if self.reserve_dynamic else Reservation(cells=set(), moves=set())
            action = self._action_for_agent(agent, active_reservation)
            actions[agent.id - 1] = action.value
            if self.reserve_dynamic:
                self._reserve(agent, action, reservation)
        return actions

    def reference_actions(self) -> List[int]:
        return self.act()

    def reference_paths(self) -> List[List[Coord]]:
        """Return the last path reference aligned by agent id."""
        return [self.last_paths.get(agent.id, []) for agent in self.task_env.env.agents]

    def _priority(self, agent: Agent) -> Tuple[int, int]:
        task = self.task_env.manager.task_for_agent(agent.id)
        if task is None:
            return 0, 0
        target = self.task_env.manager.target_for(agent)
        dist = 0 if target is None else manhattan((agent.x, agent.y), target)
        phase_weight = {
            Phase.RETURNING: 4,
            Phase.TO_DELIVER: 3,
            Phase.TO_PICKUP: 2,
            Phase.IDLE: 1,
        }[task.phase]
        return phase_weight, -dist

    def _action_for_agent(self, agent: Agent, reservation: Reservation) -> Action:
        task = self.task_env.manager.task_for_agent(agent.id)
        if task is None:
            self.last_paths[agent.id] = []
            return Action.NOOP

        target = self.task_env.manager.target_for(agent)
        if target is None:
            self.last_paths[agent.id] = []
            return Action.NOOP

        current = (agent.x, agent.y)
        if current == target:
            self.last_paths[agent.id] = [current]
            if task.phase == Phase.TO_PICKUP and agent.carrying_shelf is None:
                return Action.TOGGLE_LOAD
            if task.phase == Phase.RETURNING and agent.carrying_shelf is not None:
                return Action.TOGGLE_LOAD
            return Action.NOOP

        path = self._astar(agent, target, reservation.cells)
        self.last_paths[agent.id] = path
        if len(path) < 2:
            return self._fallback_turn(agent)

        next_cell = path[1]
        if next_cell in reservation.cells:
            return Action.NOOP
        if (next_cell, current) in reservation.moves:
            return Action.NOOP

        desired = direction_between(current, next_cell)
        if desired is None:
            return Action.NOOP
        if agent.dir != desired:
            return turn_towards(agent.dir, desired)
        return Action.FORWARD

    def _reserve(self, agent: Agent, action: Action, reservation: Reservation) -> None:
        start = (agent.x, agent.y)
        end = start
        if action == Action.FORWARD:
            dx, dy = DIR_TO_DELTA[agent.dir]
            end = (agent.x + dx, agent.y + dy)
        reservation.cells.add(end)
        reservation.moves.add((start, end))

    def _fallback_turn(self, agent: Agent) -> Action:
        task = self.task_env.manager.task_for_agent(agent.id)
        target = self.task_env.manager.target_for(agent) if task else None
        if target is None:
            return Action.NOOP
        desired = coarse_direction((agent.x, agent.y), target)
        if desired is None or desired == agent.dir:
            return Action.NOOP
        return turn_towards(agent.dir, desired)

    def _astar(self, agent: Agent, target: Coord, reserved: Set[Coord]) -> List[Coord]:
        start = (agent.x, agent.y)
        if start == target:
            return [start]
        frontier: List[Tuple[int, int, Coord]] = []
        heappush(frontier, (0, 0, start))
        came_from: Dict[Coord, Optional[Coord]] = {start: None}
        cost_so_far: Dict[Coord, int] = {start: 0}
        counter = 0
        while frontier:
            _, _, current = heappop(frontier)
            if current == target:
                break
            for nxt in self._neighbors(agent, current, target, reserved):
                new_cost = cost_so_far[current] + self._move_cost(agent, nxt)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    counter += 1
                    priority = new_cost + manhattan(nxt, target)
                    heappush(frontier, (priority, counter, nxt))
                    came_from[nxt] = current
        if target not in came_from:
            return []
        path = [target]
        while path[-1] != start:
            parent = came_from[path[-1]]
            if parent is None:
                break
            path.append(parent)
        path.reverse()
        return path

    def _update_dynamic_cost(self, agent: Agent) -> None:
        target = self.task_env.manager.target_for(agent)
        if target is None:
            self.dynamic_costs.pop(agent.id, None)
            self.dynamic_targets.pop(agent.id, None)
            return
        height, width = self.task_env.env.grid_size
        if self.dynamic_targets.get(agent.id) != target:
            self.dynamic_costs[agent.id] = np.zeros((height, width), dtype=np.float32)
            self.dynamic_targets[agent.id] = target
        costs = self.dynamic_costs[agent.id]
        radius = max(1, int(self.task_env.config.sensor_range))
        for other in self.task_env.env.agents:
            if other.id == agent.id:
                continue
            if abs(other.x - agent.x) > radius or abs(other.y - agent.y) > radius:
                continue
            costs[other.y, other.x] += 1.0

    def _move_cost(self, agent: Agent, cell: Coord) -> float:
        costs = self.dynamic_costs.get(agent.id)
        if costs is None:
            return 1.0
        return 1.0 + self.dynamic_weight * float(costs[cell[1], cell[0]])

    def _neighbors(
        self,
        agent: Agent,
        current: Coord,
        target: Coord,
        reserved: Set[Coord],
    ) -> Sequence[Coord]:
        width = self.task_env.env.grid_size[1]
        height = self.task_env.env.grid_size[0]
        result = []
        for dx, dy in DIR_TO_DELTA.values():
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt in reserved and nxt != target:
                continue
            if agent.carrying_shelf is not None and self._has_blocking_shelf(agent, nxt, target):
                continue
            result.append(nxt)
        return result

    def _has_blocking_shelf(self, agent: Agent, cell: Coord, target: Coord) -> bool:
        shelf_id = self.task_env.env.grid[1, cell[1], cell[0]]
        if shelf_id == 0:
            return False
        if agent.carrying_shelf is not None and shelf_id == agent.carrying_shelf.id:
            return False
        return cell != target


def direction_between(src: Coord, dst: Coord) -> Optional[Direction]:
    return DELTA_TO_DIR.get((dst[0] - src[0], dst[1] - src[1]))


def coarse_direction(src: Coord, dst: Coord) -> Optional[Direction]:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return Direction.RIGHT if dx > 0 else Direction.LEFT
    if dy != 0:
        return Direction.DOWN if dy > 0 else Direction.UP
    return None


def turn_towards(current: Direction, desired: Direction) -> Action:
    ci = TURN_ORDER.index(current)
    di = TURN_ORDER.index(desired)
    right_steps = (di - ci) % len(TURN_ORDER)
    left_steps = (ci - di) % len(TURN_ORDER)
    return Action.RIGHT if right_steps <= left_steps else Action.LEFT
