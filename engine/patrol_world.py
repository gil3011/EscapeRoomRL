"""PatrolGridWorld — Room 4's grid with a moving patrol enemy.

A subclass of GridWorld (so Rooms 1-3 and the DP solver are untouched) that adds a
hazard which moves one cell every agent step. Colliding with it ends the episode in
failure.

The key idea: once a hazard *moves*, the agent's cell alone is no longer a Markov state
— the same cell is safe or deadly depending on where the patroller currently is. So the
state here is augmented to ``(agent_cell, phase)``, where ``phase`` indexes the enemy's
position. That's all the extra information needed to make the future predictable again,
and it's the room's second lesson (a dynamic hazard forces you to grow the state).

The enemy moves in one of two modes:

* deterministic (``random_enemy=False``): it walks ``patrol_path`` as a fixed cycle, so
  ``phase`` indexes the cycle and the crossing can be *timed*.
* random (``random_enemy=True``, Room 4's setting): it random-walks over the DISTINCT
  cells of ``patrol_path`` — each step it stays put or steps to a grid-adjacent patrol
  cell, uniformly at random — so ``phase`` indexes the enemy's *current cell*. Its
  position is now a random variable, so there's nothing to time; the agent has to learn
  a policy that reacts to where the enemy is and keeps a margin against where it might go.

Q-learning is model-free, so no augmented transition_model() is needed — only
reset/step/is_terminal/all_states/legal_actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.grid_world import GridWorld


@dataclass
class PatrolGridWorld(GridWorld):
    # the sequence of cells that defines the enemy's beat. In deterministic mode it is
    # walked as a full cycle (the ping-pong trip out *and* back); in random mode only its
    # distinct cells matter, as the region the enemy random-walks over.
    patrol_path: list = field(default_factory=list)
    enemy_reward: float = -100.0  # one-time penalty on collision (failure-terminal)
    random_enemy: bool = False    # see module docstring: random walk vs. fixed cycle

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.random_enemy:
            # distinct patrol cells, order preserved; phase indexes a *position* now
            self._enemy_positions = list(dict.fromkeys(self.patrol_path))
            # lazy random walk: from each position the enemy may stay or step to a
            # grid-adjacent patrol cell (precomputed as candidate next-phase indices)
            self._enemy_moves = {
                i: [i] + [j for j, c in enumerate(self._enemy_positions)
                          if abs(c[0] - p[0]) + abs(c[1] - p[1]) == 1]
                for i, p in enumerate(self._enemy_positions)
            }
        else:
            self._enemy_positions = list(self.patrol_path)
        self.period = len(self._enemy_positions)
        self.state = (self.start, 0)

    # --- state plumbing (states are (agent_cell, phase)) ------------------------
    def reset(self):
        self.state = (self.start, 0)
        return self.state

    def initial_state(self):
        return (self.start, 0)

    def enemy_cell(self, phase: int) -> tuple[int, int]:
        return self._enemy_positions[phase]

    def _next_phase(self, phase: int) -> int:
        """Where the enemy's phase index goes next: a random adjacent/stay step in
        random mode, or the next cell in the fixed cycle otherwise."""
        if self.random_enemy:
            return self._rng.choice(self._enemy_moves[phase])
        return (phase + 1) % self.period

    def is_goal(self, state) -> bool:
        return state[0] == self.goal

    def is_terminal(self, state) -> bool:
        cell, phase = state
        return cell == self.goal or cell == self._enemy_positions[phase]

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
        enemy_before = self._enemy_positions[phase]
        next_cell = self._sample_next_cell(agent_cell, action)  # slip + move + shortcut
        next_phase = self._next_phase(phase)
        enemy_after = self._enemy_positions[next_phase]

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
