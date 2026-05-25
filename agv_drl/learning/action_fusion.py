from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..env.task_env import AGVTaskEnv, Coord
from ..planning.astar import direction_between, turn_towards

from rware.warehouse import Action


MOVE_WAIT = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4

MOVE_TO_DELTA = {
    MOVE_WAIT: (0, 0),
    MOVE_UP: (0, -1),
    MOVE_DOWN: (0, 1),
    MOVE_LEFT: (-1, 0),
    MOVE_RIGHT: (1, 0),
}


@dataclass
class ActionFusionStats:
    policy_actions: int = 0
    forced_toggles: int = 0

    def add(self, other: "ActionFusionStats") -> None:
        self.policy_actions += other.policy_actions
        self.forced_toggles += other.forced_toggles


def reference_movement_intents(reference_paths: Sequence[Sequence[Coord]]) -> np.ndarray:
    """Convert planner paths into Learn-to-Follow style grid-move labels."""

    intents = []
    for path in reference_paths:
        if len(path) < 2:
            intents.append(MOVE_WAIT)
            continue
        dx = path[1][0] - path[0][0]
        dy = path[1][1] - path[0][1]
        if (dx, dy) == (0, -1):
            intents.append(MOVE_UP)
        elif (dx, dy) == (0, 1):
            intents.append(MOVE_DOWN)
        elif (dx, dy) == (-1, 0):
            intents.append(MOVE_LEFT)
        elif (dx, dy) == (1, 0):
            intents.append(MOVE_RIGHT)
        else:
            intents.append(MOVE_WAIT)
    return np.asarray(intents, dtype=np.int64)


def movement_intents_to_rware_actions(
    task_env: AGVTaskEnv,
    reference_actions: Sequence[int],
    movement_intents: Sequence[int],
) -> tuple[list[int], ActionFusionStats]:
    """Translate grid-move follower actions into RWARE primitive actions.

    The learned policy follows the original Learn-to-Follow action semantics:
    choose a neighboring grid cell or wait. RWARE additionally has orientation
    and load/unload primitives, so this adapter turns toward the desired cell,
    moves forward when already aligned, and leaves load/unload to the task
    state machine.
    """

    stats = ActionFusionStats()
    actions: list[int] = []
    toggle = Action.TOGGLE_LOAD.value
    for agent, reference_action, intent in zip(task_env.env.agents, reference_actions, movement_intents):
        reference_action = int(reference_action)
        if reference_action == toggle:
            actions.append(toggle)
            stats.forced_toggles += 1
            continue

        dx, dy = MOVE_TO_DELTA.get(int(intent), (0, 0))
        if (dx, dy) == (0, 0):
            actions.append(Action.NOOP.value)
            stats.policy_actions += 1
            continue

        current = (agent.x, agent.y)
        desired = direction_between(current, (agent.x + dx, agent.y + dy))
        if desired is None:
            actions.append(Action.NOOP.value)
        elif agent.dir == desired:
            actions.append(Action.FORWARD.value)
        else:
            actions.append(turn_towards(agent.dir, desired).value)
        stats.policy_actions += 1
    return actions, stats
