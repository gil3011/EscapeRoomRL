from engine.grid_world import DOWN, RIGHT, UP
from engine.patrol_world import PatrolGridWorld


def test_reset_returns_augmented_start_state():
    g = PatrolGridWorld(size=3, start=(0, 0), goal=(2, 2), patrol_path=[(0, 2), (1, 2)])
    assert g.reset() == ((0, 0), 0)
    assert g.initial_state() == ((0, 0), 0)


def test_enemy_advances_deterministically_each_step():
    g = PatrolGridWorld(size=4, start=(0, 0), goal=(3, 3),
                        patrol_path=[(0, 3), (1, 3), (2, 3), (1, 3)])  # period 4 ping-pong
    g.reset()
    assert g.enemy_cell(0) == (0, 3)
    state, _r, done, _i = g.step(DOWN)  # agent (0,0)->(1,0); phase 0->1
    assert state == ((1, 0), 1)
    assert not done
    assert g.enemy_cell(1) == (1, 3)


def test_walking_into_the_enemy_is_terminal_failure():
    # phase 1 enemy sits at (1,3); agent at (1,2) stepping RIGHT lands on it.
    g = PatrolGridWorld(size=4, start=(1, 2), goal=(3, 3),
                        patrol_path=[(0, 3), (1, 3)], enemy_reward=-50.0)
    g.reset()
    state, reward, done, info = g.step(RIGHT)
    assert done and reward == -50.0
    assert info["outcome"] == "caught"
    assert state == ((1, 3), 1)


def test_swapping_past_the_enemy_counts_as_a_collision():
    # agent (0,3), enemy_before (1,3) at phase 0; enemy moves to (0,3) at phase 1 while
    # the agent moves down to (1,3) -- they cross, which must be caught, not a free pass.
    g = PatrolGridWorld(size=4, start=(0, 3), goal=(3, 0), patrol_path=[(1, 3), (0, 3)])
    g.reset()
    _state, _reward, done, info = g.step(DOWN)
    assert done and info["outcome"] == "caught"


def test_reaching_the_goal_is_terminal_success():
    g = PatrolGridWorld(size=2, start=(0, 0), goal=(0, 1),
                        patrol_path=[(1, 0), (1, 1)], goal_reward=7.0)
    g.reset()
    state, reward, done, info = g.step(RIGHT)
    assert done and reward == 7.0
    assert info["outcome"] == "goal"
    assert g.is_goal(state)


def test_augmented_state_space_and_phase_dependent_terminals():
    g = PatrolGridWorld(size=3, start=(0, 0), goal=(2, 2), patrol_path=[(0, 2), (1, 2)])
    states = g.all_states()
    assert len(states) == 9 * 2  # 9 cells x 2 phases
    # movement ignores phase
    assert set(g.legal_actions(((0, 0), 0))) == set(g.legal_actions(((0, 0), 1)))
    # goal is terminal at any phase; an enemy cell is terminal only at its phase
    assert g.is_terminal(((2, 2), 0)) and g.is_terminal(((2, 2), 1))
    assert g.is_terminal(((0, 2), 0))       # enemy is here at phase 0
    assert not g.is_terminal(((0, 2), 1))   # ...but not at phase 1


def test_shortcut_teleport_still_works_under_the_subclass():
    g = PatrolGridWorld(size=3, start=(0, 0), goal=(2, 2), patrol_path=[(0, 2), (1, 2)],
                        shortcuts={(0, 1): (2, 0)}, slip_prob=0.0)
    g.reset()
    state, _r, _d, _i = g.step(RIGHT)  # (0,0)->(0,1)->teleport (2,0); phase advances
    assert state == ((2, 0), 1)


def test_slip_still_fires_and_phase_advances_regardless():
    g = PatrolGridWorld(size=3, start=(1, 1), goal=(2, 2), patrol_path=[(0, 0), (0, 1)],
                        slippery=frozenset({(1, 1)}), slip_prob=1.0, seed=0)
    g.reset()
    state, _r, _d, _i = g.step(UP)  # intended UP->(0,1), but slip forces another legal dir
    assert state[0] != (0, 1)       # slipped somewhere else
    assert state[1] == 1            # enemy phase advances no matter what the agent did


def test_random_enemy_phase_indexes_distinct_patrol_cells():
    # ping-pong path with duplicates -> random mode collapses to the 4 distinct cells
    g = PatrolGridWorld(size=8, start=(0, 0), goal=(7, 7), random_enemy=True,
                        patrol_path=[(3, 5), (4, 5), (5, 5), (6, 5), (5, 5), (4, 5)])
    assert g.period == 4
    assert [g.enemy_cell(p) for p in range(4)] == [(3, 5), (4, 5), (5, 5), (6, 5)]


def test_random_enemy_only_stays_or_steps_to_an_adjacent_patrol_cell():
    g = PatrolGridWorld(size=8, start=(0, 0), goal=(7, 7), random_enemy=True, seed=1,
                        patrol_path=[(3, 5), (4, 5), (5, 5), (6, 5)])
    g.reset()
    seen_positions = set()
    prev = g.enemy_cell(g.state[1])
    for _ in range(200):
        if g.is_terminal(g.state):
            g.reset()
            prev = g.enemy_cell(g.state[1])
            continue
        _s, _r, _d, _i = g.step(DOWN)
        cur = g.enemy_cell(g.state[1])
        seen_positions.add(cur)
        # each enemy step is a stay or a one-cell move along the patrol line
        assert abs(cur[0] - prev[0]) + abs(cur[1] - prev[1]) <= 1
        prev = cur
    # over many steps the random walk visits more than one cell (it isn't stuck)
    assert len(seen_positions) > 1


def test_random_enemy_is_reproducible_under_a_seed():
    def run(seed):
        g = PatrolGridWorld(size=8, start=(0, 0), goal=(7, 7), random_enemy=True, seed=seed,
                            patrol_path=[(3, 5), (4, 5), (5, 5), (6, 5)])
        g.reset()
        phases = []
        for _ in range(30):
            if g.is_terminal(g.state):
                break
            g.step(DOWN)
            phases.append(g.state[1])
        return phases
    assert run(7) == run(7)
