from collections import deque

from engine.boards import (
    ROOM1_GOAL, ROOM1_SIZE, ROOM1_SLIPPERY, ROOM1_START, ROOM1_TRAPS, ROOM1_WALLS,
    make_room1_grid,
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
