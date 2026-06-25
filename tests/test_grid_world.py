from engine.grid_world import DOWN, RIGHT, UP, GridWorld, random_layout


def test_reset_returns_start():
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    assert g.reset() == (0, 0)


def test_wall_blocks_movement():
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2), walls=frozenset({(0, 1)}))
    g.reset()
    state, _, _, _ = g.step(RIGHT)
    assert state == (0, 0)


def test_boundary_blocks_movement():
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    g.reset()
    state, _, _, _ = g.step(UP)
    assert state == (0, 0)


def test_reaching_goal_terminates_with_goal_reward():
    g = GridWorld(size=2, start=(0, 0), goal=(0, 1), goal_reward=1.0)
    g.reset()
    state, reward, done, _ = g.step(RIGHT)
    assert state == (0, 1)
    assert done is True
    assert reward == 1.0


def test_trap_terminates_with_trap_reward():
    g = GridWorld(size=2, start=(0, 0), goal=(1, 1), traps=frozenset({(0, 1)}), trap_reward=-1.0)
    g.reset()
    state, reward, done, _ = g.step(RIGHT)
    assert state == (0, 1)
    assert done is True
    assert reward == -1.0


def test_slip_probability_one_never_takes_intended_action():
    g = GridWorld(size=5, start=(2, 2), goal=(4, 4), slippery=frozenset({(2, 2)}),
                  slip_prob=1.0, seed=0)
    outcomes = set()
    for _ in range(50):
        g.reset()
        state, _, _, _ = g.step(RIGHT)
        outcomes.add(state)
    assert (2, 3) not in outcomes


def test_transition_model_probabilities_sum_to_one():
    g = GridWorld(size=4, start=(0, 0), goal=(3, 3), slippery=frozenset({(1, 1)}), slip_prob=0.2)
    model = g.transition_model()
    assert len(model) > 0
    for outcomes in model.values():
        total = sum(p for p, *_ in outcomes)
        assert abs(total - 1.0) < 1e-9


def test_transition_model_excludes_terminal_states():
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2))
    model = g.transition_model()
    assert all(s != g.goal for (s, _a) in model)


def test_random_layout_avoids_start_and_goal():
    walls, slippery, traps = random_layout(size=10, start=(0, 0), goal=(9, 9),
                                            n_slippery=8, n_traps=2, n_walls=3, seed=1)
    placed = walls | slippery | traps
    assert (0, 0) not in placed
    assert (9, 9) not in placed
    assert len(walls) == 3
    assert len(slippery) == 8
    assert len(traps) == 2


def test_step_after_terminal_raises():
    g = GridWorld(size=2, start=(0, 0), goal=(0, 1))
    g.reset()
    g.step(RIGHT)
    try:
        g.step(DOWN)
        assert False, "expected ValueError"
    except ValueError:
        pass
