from collections import deque

from engine.boards import (
    ROOM1_GOAL, ROOM1_SIZE, ROOM1_SLIPPERY, ROOM1_START, ROOM1_TRAPS, ROOM1_WALLS,
    ROOM3_GOAL, ROOM3_SHORTCUT_DST, ROOM3_SHORTCUT_SRC, ROOM3_SIZE, ROOM3_SLIPPERY,
    ROOM3_START, ROOM3_TRAPS, ROOM3_WALLS,
    ROOM4_GOAL, ROOM4_PATROL_PATH, ROOM4_SIZE, ROOM4_SLIPPERY, ROOM4_START,
    ROOM4_TRAPS, ROOM4_WALLS,
    make_room1_grid, make_room3_grid, make_room4_grid,
)
from engine.grid_world import ACTION_DELTAS


def _neighbors(cell, size):
    i, j = cell
    for di, dj in ACTION_DELTAS.values():
        ni, nj = i + di, j + dj
        if 0 <= ni < size and 0 <= nj < size:
            yield (ni, nj)


def _reachable_from(start, walls, size):
    visited = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for n in _neighbors(cur, size):
            if n not in walls and n not in visited:
                visited.add(n)
                queue.append(n)
    return visited


def test_no_overlap_between_special_cells():
    groups = [ROOM1_WALLS, ROOM1_TRAPS, ROOM1_SLIPPERY, {ROOM1_START}, {ROOM1_GOAL}]
    seen = set()
    for group in groups:
        overlap = seen & group
        assert not overlap, f"overlapping cells: {overlap}"
        seen |= group


def test_goal_reachable_from_start():
    reachable = _reachable_from(ROOM1_START, ROOM1_WALLS, ROOM1_SIZE)
    assert ROOM1_GOAL in reachable


def test_no_isolated_pockets():
    all_open_cells = {
        (i, j) for i in range(ROOM1_SIZE) for j in range(ROOM1_SIZE)
    } - ROOM1_WALLS
    reachable = _reachable_from(ROOM1_START, ROOM1_WALLS, ROOM1_SIZE)
    assert reachable == all_open_cells


def test_every_open_cell_has_a_legal_action():
    # the wall clusters are scattered fairly densely -- make sure none of them
    # accidentally box in a cell on all 4 sides, which would leave it with no
    # legal action once wall-bumping moves are excluded from the model.
    grid = make_room1_grid()
    model = grid.transition_model()
    actions_by_state: dict = {}
    for (s, a) in model:
        actions_by_state.setdefault(s, []).append(a)
    for s in grid.all_states():
        if not grid.is_terminal(s):
            assert actions_by_state.get(s), f"{s} has no legal action"


# --- Room 3 ---------------------------------------------------------------------

def _shortest_path(start, goal, walls, size, shortcuts):
    """BFS length start->goal; stepping onto a shortcut source lands on its dest."""
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        cur, d = queue.popleft()
        if cur == goal:
            return d
        for n in _neighbors(cur, size):
            if n in walls:
                continue
            landing = shortcuts.get(n, n)
            if landing not in seen:
                seen.add(landing)
                queue.append((landing, d + 1))
    return None


def test_room3_no_overlap_between_special_cells():
    groups = [ROOM3_WALLS, ROOM3_TRAPS, ROOM3_SLIPPERY,
              {ROOM3_SHORTCUT_SRC}, {ROOM3_SHORTCUT_DST}, {ROOM3_START}, {ROOM3_GOAL}]
    seen = set()
    for group in groups:
        overlap = seen & group
        assert not overlap, f"overlapping cells: {overlap}"
        seen |= group


def test_room3_goal_reachable_and_no_isolated_pockets():
    all_open = {(i, j) for i in range(ROOM3_SIZE) for j in range(ROOM3_SIZE)} - ROOM3_WALLS
    reachable = _reachable_from(ROOM3_START, ROOM3_WALLS, ROOM3_SIZE)
    assert ROOM3_GOAL in reachable
    assert reachable == all_open


def test_room3_shortcut_destination_is_a_plain_cell():
    # a teleport should just relocate the agent -- landing on a wall/trap/slippery
    # cell or the start/goal would make its effect compound in confusing ways.
    dst = ROOM3_SHORTCUT_DST
    assert dst not in ROOM3_WALLS
    assert dst not in ROOM3_TRAPS
    assert dst not in ROOM3_SLIPPERY
    assert dst not in (ROOM3_START, ROOM3_GOAL, ROOM3_SHORTCUT_SRC)


def test_room3_shortcut_saves_steps_without_trivializing():
    with_sc = _shortest_path(ROOM3_START, ROOM3_GOAL, ROOM3_WALLS, ROOM3_SIZE,
                             {ROOM3_SHORTCUT_SRC: ROOM3_SHORTCUT_DST})
    without_sc = _shortest_path(ROOM3_START, ROOM3_GOAL, ROOM3_WALLS, ROOM3_SIZE, {})
    assert with_sc is not None and without_sc is not None
    assert without_sc - with_sc >= 3, "shortcut should meaningfully shorten the route"
    assert with_sc >= 8, "shortcut shouldn't trivialize the room"


def test_room3_every_open_cell_has_a_legal_action():
    grid = make_room3_grid()
    model = grid.transition_model()
    actions_by_state: dict = {}
    for (s, a) in model:
        actions_by_state.setdefault(s, []).append(a)
    for s in grid.all_states():
        if not grid.is_terminal(s):
            assert actions_by_state.get(s), f"{s} has no legal action"


# --- Room 4 (patrol enemy + trap door) ------------------------------------------

def _timed_solve(start, goal, walls, size, shortcuts, patrol_path):
    """Shortest slip-free path over (cell, phase) space, honouring the trap-door
    teleport and enemy collisions (including swaps) — mirrors PatrolGridWorld."""
    period = len(patrol_path)
    start_state = (start, 0)
    seen = {start_state}
    queue = deque([(start_state, 0)])
    while queue:
        (cell, phase), d = queue.popleft()
        if cell == goal:
            return d
        nphase = (phase + 1) % period
        e_before, e_after = patrol_path[phase], patrol_path[nphase]
        for n in _neighbors(cell, size):
            if n in walls:
                continue
            ncell = shortcuts.get(n, n)
            if ncell == e_after or (ncell == e_before and cell == e_after):
                continue  # caught
            ns = (ncell, nphase)
            if ns not in seen:
                seen.add(ns)
                queue.append((ns, d + 1))
    return None


def test_room4_no_overlap_between_special_cells():
    groups = [ROOM4_WALLS, ROOM4_TRAPS, ROOM4_SLIPPERY, set(ROOM4_PATROL_PATH),
              {ROOM4_START}, {ROOM4_GOAL}]
    seen = set()
    for group in groups:
        overlap = seen & set(group)
        assert not overlap, f"overlapping cells: {overlap}"
        seen |= set(group)


def test_room4_enemy_never_occupies_start_or_goal():
    assert ROOM4_START not in ROOM4_PATROL_PATH
    assert ROOM4_GOAL not in ROOM4_PATROL_PATH


def test_room4_goal_reachable_and_no_isolated_pockets():
    all_open = {(i, j) for i in range(ROOM4_SIZE) for j in range(ROOM4_SIZE)} - ROOM4_WALLS
    reachable = _reachable_from(ROOM4_START, ROOM4_WALLS, ROOM4_SIZE)
    assert ROOM4_GOAL in reachable
    assert reachable == all_open


def test_room4_is_solvable_with_correct_timing():
    d = _timed_solve(ROOM4_START, ROOM4_GOAL, ROOM4_WALLS, ROOM4_SIZE,
                     {}, list(ROOM4_PATROL_PATH))
    assert d is not None, "no timed path threads the patrol — board is unsolvable"
    assert d >= 18


def test_room4_factory_builds_a_consistent_patrol_grid():
    grid = make_room4_grid()
    assert grid.reset() == ((0, 0), 0)
    # deterministic ping-pong: the phase indexes the full out-and-back cycle
    assert grid.period == len(ROOM4_PATROL_PATH)
    # the barrier gap: crossing column 5 is only possible at rows 2-6
    open_col5 = [i for i in range(ROOM4_SIZE) if (i, 5) not in ROOM4_WALLS]
    assert open_col5 == [2, 3, 4, 5, 6]
