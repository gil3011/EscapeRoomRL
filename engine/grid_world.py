"""Configurable 10x10 grid-world MDP shared by Rooms 1-4.

One class serves all four grid rooms. What makes the model "known" or "unknown" to a
room is which methods that room's algorithm calls: Room 1 (Dynamic Programming) is the
only one that calls transition_model(); Rooms 2-4 only ever call reset()/step().
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

UP, DOWN, LEFT, RIGHT = range(4)
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
ACTION_ARROWS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}


@dataclass
class GridWorld:
    size: int = 10
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (9, 9)
    walls: frozenset = field(default_factory=frozenset)
    slippery: frozenset = field(default_factory=frozenset)
    traps: frozenset = field(default_factory=frozenset)
    slip_prob: float = 0.2
    step_reward: float = -0.04
    goal_reward: float = 1.0
    trap_reward: float = -1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.state = self.start

    def reset(self) -> tuple[int, int]:
        self.state = self.start
        return self.state

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return state == self.goal

    def all_states(self) -> list[tuple[int, int]]:
        return [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if (i, j) not in self.walls
        ]

    def step(self, action: int):
        if self.is_terminal(self.state):
            raise ValueError("step() called on a terminal state; call reset() first")

        actual_action = action
        if self.state in self.slippery and self._rng.random() < self.slip_prob:
            alternatives = [a for a in self._legal_actions(self.state) if a != action]
            if alternatives:
                actual_action = self._rng.choice(alternatives)
            # no legal alternative to slip into (e.g. walled in on every other side):
            # the intended action just proceeds, slip_prob has nowhere to go this time.

        next_state = self._move(self.state, actual_action)
        reward, done = self._reward_for(next_state)
        self.state = next_state
        return next_state, reward, done, {}

    def transition_model(self):
        """P(s'|s,a) and R(s,a,s') for every non-terminal state. Room 1 (DP) only.

        Only offers actions that actually move the agent: an action that would bump
        into a wall or off the edge of the board is left out entirely, so a policy
        derived from this model (argmax over the offered actions) can never choose
        to walk into a wall. Slip is restricted the same way — it only ever
        substitutes another *legal* direction, never one that would bump a wall.
        """
        model = {}
        for s in self.all_states():
            if self.is_terminal(s):
                continue
            for a in self._legal_actions(s):
                model[(s, a)] = self._outcomes(s, a)
        return model

    def _legal_actions(self, state: tuple[int, int]) -> list[int]:
        """Actions that actually move the agent from this state — excludes any
        action that would just bump into a wall or off the edge of the board."""
        return [a for a in ACTIONS if self._move(state, a) != state]

    def _outcomes(self, s: tuple[int, int], a: int):
        """List of (probability, next_state, reward, done) for taking action a in state s."""
        if s not in self.slippery:
            s2 = self._move(s, a)
            r, done = self._reward_for(s2)
            return [(1.0, s2, r, done)]

        alternatives = [x for x in self._legal_actions(s) if x != a]
        if not alternatives:
            s2 = self._move(s, a)
            r, done = self._reward_for(s2)
            return [(1.0, s2, r, done)]

        outcomes = []
        for actual in [a] + alternatives:
            prob = (1 - self.slip_prob) if actual == a else self.slip_prob / len(alternatives)
            s2 = self._move(s, actual)
            r, done = self._reward_for(s2)
            outcomes.append((prob, s2, r, done))
        return outcomes

    def _move(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        di, dj = ACTION_DELTAS[action]
        i, j = state[0] + di, state[1] + dj
        if i < 0 or i >= self.size or j < 0 or j >= self.size or (i, j) in self.walls:
            return state  # bumped into a wall/boundary: no movement
        return (i, j)

    def _reward_for(self, state: tuple[int, int]):
        if state == self.goal:
            return self.goal_reward, True
        if state in self.traps:
            return self.trap_reward, False  # costly, but not terminal
        return self.step_reward, False


def run_episode(grid: GridWorld, policy: dict, max_steps: int = 200):
    """Roll out a deterministic policy from grid.reset() until a terminal state or
    max_steps. Returns (path, rewards, steps_taken, success); path includes the
    start state and every state visited after it, rewards is the reward received at
    each step (so len(rewards) == steps_taken, one shorter than path). success means
    the goal specifically was reached (not a timeout). Shared by Rooms 1-4's escape
    attempts; rewards lets the caller compute G = discounted_return(rewards, gamma).
    """
    state = grid.reset()
    path = [state]
    rewards = []
    for _ in range(max_steps):
        state, reward, done, _info = grid.step(policy[state])
        path.append(state)
        rewards.append(reward)
        if done:
            return path, rewards, len(rewards), state == grid.goal
    return path, rewards, max_steps, False


def random_layout(
    size: int = 10,
    start: tuple[int, int] = (0, 0),
    goal: tuple[int, int] = (9, 9),
    n_slippery: int = 8,
    n_traps: int = 0,
    n_walls: int = 0,
    seed: int | None = None,
):
    """Scatter slippery/trap/wall cells over the grid, keeping start and goal clear.

    Shared by Rooms 1-4 so each room's sidebar can regenerate its own layout from the
    same generator with different counts, instead of four bespoke grid builders.
    """
    rng = random.Random(seed)
    forbidden = {start, goal}
    cells = [(i, j) for i in range(size) for j in range(size) if (i, j) not in forbidden]
    rng.shuffle(cells)

    walls = frozenset(cells[:n_walls])
    cells = cells[n_walls:]
    traps = frozenset(cells[:n_traps])
    cells = cells[n_traps:]
    slippery = frozenset(cells[:n_slippery])

    return walls, slippery, traps
