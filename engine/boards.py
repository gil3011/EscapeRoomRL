"""Hand-designed static board layouts, as an alternative to grid_world.random_layout()
for rooms that want a fixed, reproducible environment instead of a procedurally
generated one.

Room 1's wall clusters were authored by seeding a random walk from 5 spread-out
starting cells and growing each into a 2-4 cell cluster, then validated (goal
reachable from start, no isolated pockets — see tests/test_boards.py) and frozen
here as plain constants.
"""
from engine.grid_world import GridWorld

ROOM1_SIZE = 10
ROOM1_START = (0, 0)
ROOM1_GOAL = (9, 9)

ROOM1_WALLS = frozenset({
    (0, 2), (0, 3), (1, 2),
    (2, 0), (3, 0),
    (5, 7), (6, 7),
    (6, 2), (7, 2),
    (8, 3), (8, 4), (9, 3), (9, 4),
})
ROOM1_TRAPS = frozenset({(4, 6), (7, 3), (2, 8)})
ROOM1_SLIPPERY = frozenset({
    (1, 1), (1, 7), (3, 2), (3, 8), (5, 1),
    (5, 8), (6, 4), (7, 6), (8, 1), (8, 7),
})


def make_room1_grid(slip_prob: float = 0.2, goal_reward: float = 20.0,
                     trap_reward: float = -20.0) -> GridWorld:
    """No step_reward parameter on purpose — Room 1's Bellman equation has no separate
    step cost; V(s) is shaped entirely by discounting the terminal rewards (see
    SPRINTS.md Sprint 2, 'Reward model')."""
    return GridWorld(
        size=ROOM1_SIZE, start=ROOM1_START, goal=ROOM1_GOAL,
        walls=ROOM1_WALLS, traps=ROOM1_TRAPS, slippery=ROOM1_SLIPPERY,
        slip_prob=slip_prob, step_reward=0.0,
        goal_reward=goal_reward, trap_reward=trap_reward,
    )
