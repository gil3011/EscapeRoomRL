"""PatrolGridWorld — Room 4's grid with a moving patrol enemy.

A subclass of GridWorld (so Rooms 1-3 and the DP solver are untouched) that adds a
hazard which moves one cell along a fixed cyclic path every agent step. Colliding with
it ends the episode in failure.

The key idea: once a hazard *moves*, the agent's cell alone is no longer a Markov state
— the same cell is safe or deadly depending on where the patroller currently is. So the
state here is augmented to ``(agent_cell, phase)``, where ``phase`` indexes the enemy's
position in its deterministic cycle. That's all the extra information needed to make the
future predictable again, and it's the room's second lesson (a dynamic hazard forces you
to grow the state). Q-learning is model-free, so no augmented transition_model() is
needed — only reset/step/is_terminal/all_states/legal_actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.grid_world import GridWorld


@dataclass
class PatrolGridWorld(GridWorld):
    # the full cyclic sequence of cells the enemy visits (the ping-pong trip out *and*
    # back), so period = len(patrol_path) and enemy_cell(phase) = patrol_path[phase].
    patrol_path: list = field(default_factory=list)
    enemy_reward: float = -100.0  # one-time penalty on collision (failure-terminal)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.period = len(self.patrol_path)
        self.state = (self.start, 0)

    # --- state plumbing (states are (agent_cell, phase)) ------------------------
    def reset(self):
        self.state = (self.start, 0)
        return self.state

    def initial_state(self):
        return (self.start, 0)

    def enemy_cell(self, phase: int) -> tuple[int, int]:
        return self.patrol_path[phase]

    def is_goal(self, state) -> bool:
        return state[0] == self.goal

    def is_terminal(self, state) -> bool:
        cell, phase = state
        return cell == self.goal or cell == self.patrol_path[phase]

    def all_states(self):
        cells = super().all_states()  # every non-wall cell
        return [(cell, phase) for cell in cells for phase in range(self.period)]

    def legal_actions(self, state):
        cell, _phase = state
        return super().legal_actions(cell)

    # --- dynamics ---------------------------------------------------------------
    def step(self, action: int):
        if self.is_terminal(self.state):
            raise ValueError("step() called on a terminal state; call reset() first")

        agent_cell, phase = self.state
        enemy_before = self.patrol_path[phase]
        next_cell = self._sample_next_cell(agent_cell, action)  # slip + move + shortcut
        next_phase = (phase + 1) % self.period
        enemy_after = self.patrol_path[next_phase]

        # caught if they share a cell after both move, or if they swapped past each
        # other (agent into the enemy's old cell while the enemy took the agent's).
        caught = next_cell == enemy_after or (
            next_cell == enemy_before and agent_cell == enemy_after
        )
        self.state = (next_cell, next_phase)
        if caught:
            return self.state, self.enemy_reward, True, {"outcome": "caught"}

        reward, done = self._reward_for(next_cell)  # goal ends it; trap costs, continues
        info = {"outcome": "goal"} if done else {}
        return self.state, reward, done, info
