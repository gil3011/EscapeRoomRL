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
    (2, 0), (3, 0), (3, 7),
    (5, 7), (6, 7),
    (6, 2), (7, 2), (8, 7),
    (8, 3), (8, 4), (9, 3), (9, 4),
})
ROOM1_TRAPS = frozenset({(4, 6), (7, 3), (2, 8)})
ROOM1_SLIPPERY = frozenset({
    (1, 7), (3, 2), (3, 8), (5, 1),
    (5, 8), (6, 4), (7, 6), (8, 1), (9, 8),
})

# Fixed and not user-adjustable: kept high on purpose so V(s) stays visible across
# the board after discounting (see SPRINTS.md Sprint 2, "Reward model").
ROOM1_GOAL_REWARD = 100.0


def make_room1_grid(slip_prob: float = 0.2, trap_reward: float = -20.0) -> GridWorld:
    """No step_reward parameter on purpose — Room 1's Bellman equation has no separate
    step cost; V(s) is shaped entirely by discounting the terminal reward. No
    goal_reward parameter either: it's fixed at ROOM1_GOAL_REWARD, not tunable."""
    return GridWorld(
        size=ROOM1_SIZE, start=ROOM1_START, goal=ROOM1_GOAL,
        walls=ROOM1_WALLS, traps=ROOM1_TRAPS, slippery=ROOM1_SLIPPERY,
        slip_prob=slip_prob, step_reward=0.0,
        goal_reward=ROOM1_GOAL_REWARD, trap_reward=trap_reward,
    )


# --- Room 3 (SARSA) -------------------------------------------------------------
# Same fixed-board convention as Room 1, a different layout, plus one new mechanic:
# a shortcut tile. The wall clusters bow the natural route around the middle of the
# board; stepping onto the shortcut source (5,1) teleports the agent to (5,8), on the
# far side near the goal — cutting the shortest solve from 18 steps to 11 (validated:
# no overlaps, goal reachable, no isolated pockets, destination is a plain cell). Same
# reward convention as Room 1: no step cost, fixed high goal reward, tunable trap cost.
ROOM3_SIZE = 10
ROOM3_START = (0, 0)
ROOM3_GOAL = (9, 9)

ROOM3_WALLS = frozenset({
    (1, 3), (2, 3), (2, 4),
    (4, 1), (4, 2),
    (3, 6), (4, 6), (4, 7),
    (6, 4), (7, 4), (7, 5),
    (8, 8), (9, 7),
})
ROOM3_TRAPS = frozenset({(2, 6), (6, 7), (8, 2)})
ROOM3_SLIPPERY = frozenset({
    (1, 1), (1, 6), (3, 3), (3, 8), (5, 2),
    (5, 5), (6, 1), (7, 8), (8, 5), (9, 3),
})
# one (source -> destination) teleport pair; source is a plain cell the agent never
# rests on (it's relocated on arrival), destination is a plain open cell near the goal.
ROOM3_SHORTCUT_SRC = (5, 1)
ROOM3_SHORTCUT_DST = (5, 8)
ROOM3_SHORTCUTS = {ROOM3_SHORTCUT_SRC: ROOM3_SHORTCUT_DST}

# Fixed, matching Room 1, so V(start) stays on a comparable scale across the lobby.
ROOM3_GOAL_REWARD = 100.0


def make_room3_grid(slip_prob: float = 0.25, trap_reward: float = -20.0) -> GridWorld:
    """Room 3's fixed board. Same reward convention as Room 1 (no step cost, fixed
    goal reward), plus the shortcut-tile teleport. Default slip_prob is nudged up to
    0.25 — Room 3 is the difficulty step right after Room 1."""
    return GridWorld(
        size=ROOM3_SIZE, start=ROOM3_START, goal=ROOM3_GOAL,
        walls=ROOM3_WALLS, traps=ROOM3_TRAPS, slippery=ROOM3_SLIPPERY,
        shortcuts=dict(ROOM3_SHORTCUTS),
        slip_prob=slip_prob, step_reward=0.0,
        goal_reward=ROOM3_GOAL_REWARD, trap_reward=trap_reward,
    )
