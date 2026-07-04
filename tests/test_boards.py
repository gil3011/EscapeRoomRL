from collections import deque

from engine.boards import (
    ROOM1_GOAL, ROOM1_SIZE, ROOM1_SLIPPERY, ROOM1_START, ROOM1_TRAPS, ROOM1_WALLS,
    ROOM3_GOAL, ROOM3_SHORTCUT_DST, ROOM3_SHORTCUT_SRC, ROOM3_SIZE, ROOM3_SLIPPERY,
    ROOM3_START, ROOM3_TRAPS, ROOM3_WALLS,
    make_room1_grid, make_room3_grid,
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
