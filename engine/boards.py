"""Hand-designed static board layouts, as an alternative to grid_world.random_layout()
for rooms that want a fixed, reproducible environment instead of a procedurally
generated one.

Room 1's wall clusters were authored by seeding a random walk from 5 spread-out
starting cells and growing each into a 2-4 cell cluster, then validated (goal
reachable from start, no isolated pockets — see tests/test_boards.py) and frozen
here as plain constants.
"""
from engine.grid_world import GridWorld
from engine.patrol_world import PatrolGridWorld

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


# --- Shared board for Rooms 3 (SARSA) and 4 (Q-learning) ------------------------
# The two rooms run on the SAME terrain, so the only things that differ between them are
# the algorithm (on-policy SARSA vs off-policy Q-learning) and each room's own mechanic —
# Room 3's shortcut tile, Room 4's patrol enemy. The layout is authored around Room 4's
# needs: a wall barrier down column 5 with a rows 2-6 gap that the patrol guards. Same
# reward convention as Room 1: no step cost, fixed high goal reward, tunable trap cost.
GRID34_SIZE = 10
GRID34_START = (0, 0)
GRID34_GOAL = (9, 9)
GRID34_WALLS = frozenset({
    (0, 5), (1, 5), (7, 5), (8, 5), (9, 5),  # the barrier, with a rows 2-6 gap
    (2, 2), (2, 3),
    (6, 2), (7, 2),
    (4, 8), (5, 8),
})
GRID34_TRAPS = frozenset({(3, 7), (6, 3), (1, 8)})
GRID34_SLIPPERY = frozenset({
    (1, 1), (8, 1), (1, 6), (8, 7),
    (3, 1), (9, 2), (2, 9), (7, 8),
})


# --- Room 3 (SARSA) -------------------------------------------------------------
# The shared board above plus one mechanic of its own: a shortcut tile (and no patrol).
# Stepping onto the shortcut source (5,1) teleports the agent to (7,7), on the far side
# of the barrier near the goal — cutting the shortest solve from 18 steps to 10
# (validated: no overlaps, destination is a plain cell, teleport shortens without
# trivializing). Same reward convention as Room 1.
ROOM3_SIZE = GRID34_SIZE
ROOM3_START = GRID34_START
ROOM3_GOAL = GRID34_GOAL
ROOM3_WALLS = GRID34_WALLS
ROOM3_TRAPS = GRID34_TRAPS
ROOM3_SLIPPERY = GRID34_SLIPPERY
# one (source -> destination) teleport pair; source is a plain cell the agent never
# rests on (it's relocated on arrival), destination is a plain open cell near the goal.
ROOM3_SHORTCUT_SRC = (5, 1)
ROOM3_SHORTCUT_DST = (7, 7)
ROOM3_SHORTCUTS = {ROOM3_SHORTCUT_SRC: ROOM3_SHORTCUT_DST}

# Fixed, matching Room 1, so V(start) stays on a comparable scale across the lobby.
ROOM3_GOAL_REWARD = 100.0


def make_room3_grid(slip_prob: float = 0.25, trap_reward: float = -20.0,
                    deadly_traps: bool = False) -> GridWorld:
    """Room 3's board — the same terrain as Room 4 (minus the patrol and trap door),
    plus the shortcut-tile teleport. Same reward convention as Room 1 (no step cost,
    fixed goal reward); default slip_prob is nudged up to 0.25. With `deadly_traps` on,
    the trap tiles end the episode in failure instead of merely costing reward."""
    return GridWorld(
        size=ROOM3_SIZE, start=ROOM3_START, goal=ROOM3_GOAL,
        walls=ROOM3_WALLS, traps=ROOM3_TRAPS, slippery=ROOM3_SLIPPERY,
        shortcuts=dict(ROOM3_SHORTCUTS),
        slip_prob=slip_prob, step_reward=0.0,
        goal_reward=ROOM3_GOAL_REWARD, trap_reward=trap_reward, deadly_traps=deadly_traps,
    )


# --- Room 4 (Q-learning) --------------------------------------------------------
# The hardest grid room: the shared board above plus one new mechanic — a moving patrol
# enemy. The vertical wall barrier at column 5 leaves a 5-row gap (rows 2-6) that every
# crossing must pass through; the enemy ping-pongs deterministically up and down that gap,
# so the crossing has to be *timed*. Validated (no overlaps; enemy never on start/goal;
# goal reachable; shortest timed solve over (cell,phase) space is 18 steps).
ROOM4_SIZE = GRID34_SIZE
ROOM4_START = GRID34_START
ROOM4_GOAL = GRID34_GOAL
ROOM4_WALLS = GRID34_WALLS
ROOM4_TRAPS = GRID34_TRAPS
ROOM4_SLIPPERY = GRID34_SLIPPERY

# ping-pong patrol up and down the five gap cells in column 5. Stored as the full cycle
# (out and back), so period = len(path) and enemy_cell(phase) = path[phase].
_ROOM4_PATROL_SEG = [(2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]
ROOM4_PATROL_PATH = _ROOM4_PATROL_SEG[:-1] + _ROOM4_PATROL_SEG[:0:-1]

ROOM4_GOAL_REWARD = 100.0    # fixed, matching the family, so V(start) stays comparable
ROOM4_ENEMY_REWARD = -100.0  # default terminal collision penalty (tunable in the sidebar)


def make_room4_grid(slip_prob: float = 0.25, trap_reward: float = -20.0,
                    enemy_reward: float = ROOM4_ENEMY_REWARD) -> PatrolGridWorld:
    """Room 4's fixed board with the deterministic ping-pong patrol enemy. Same reward
    convention as Rooms 1/3 (no step cost, fixed goal reward), plus a terminal enemy_reward
    on collision."""
    return PatrolGridWorld(
        size=ROOM4_SIZE, start=ROOM4_START, goal=ROOM4_GOAL,
        walls=ROOM4_WALLS, traps=ROOM4_TRAPS, slippery=ROOM4_SLIPPERY,
        patrol_path=list(ROOM4_PATROL_PATH), enemy_reward=enemy_reward,
        slip_prob=slip_prob, step_reward=0.0,
        goal_reward=ROOM4_GOAL_REWARD, trap_reward=trap_reward,
    )
