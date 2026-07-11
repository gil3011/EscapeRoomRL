from engine.grid_world import DOWN, LEFT, RIGHT, UP, GridWorld, random_layout, run_episode


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


def test_trap_gives_negative_reward_but_does_not_terminate():
    g = GridWorld(size=2, start=(0, 0), goal=(1, 1), traps=frozenset({(0, 1)}), trap_reward=-1.0)
    g.reset()
    state, reward, done, _ = g.step(RIGHT)
    assert state == (0, 1)
    assert done is False
    assert reward == -1.0
    # the episode keeps going -- stepping again from the trap cell works normally
    state, _reward, _done, _ = g.step(DOWN)
    assert state == (1, 1)


def test_deadly_trap_terminates_the_episode_as_a_failure():
    g = GridWorld(size=2, start=(0, 0), goal=(1, 1), traps=frozenset({(0, 1)}),
                  trap_reward=-1.0, deadly_traps=True)
    g.reset()
    state, reward, done, _ = g.step(RIGHT)
    assert state == (0, 1)
    assert reward == -1.0
    assert done is True            # deadly: the episode ends here
    assert g.is_terminal(state)    # ...and the trap cell is terminal
    assert g.is_goal(state) is False  # ...but it is a failure, not a win


def test_slip_probability_one_never_takes_intended_action():
    g = GridWorld(size=5, start=(2, 2), goal=(4, 4), slippery=frozenset({(2, 2)}),
                  slip_prob=1.0, seed=0)
    outcomes = set()
    for _ in range(50):
        g.reset()
        state, _, _, _ = g.step(RIGHT)
        outcomes.add(state)
    assert (2, 3) not in outcomes


def test_slip_never_bumps_a_wall_when_a_legal_alternative_exists():
    # from (1,0): UP->(0,0), DOWN->(2,0), LEFT bumps the boundary, RIGHT bumps the
    # wall at (1,1). With slip_prob=1.0 and intended=UP, the only legal alternative
    # is DOWN -- slip must never "choose" the wall and leave the agent stuck at (1,0).
    g = GridWorld(size=3, start=(1, 0), goal=(2, 2), walls=frozenset({(1, 1)}),
                  slippery=frozenset({(1, 0)}), slip_prob=1.0, seed=0)
    outcomes = set()
    for _ in range(30):
        g.reset()
        state, _, _, _ = g.step(UP)
        outcomes.add(state)
    assert outcomes == {(2, 0)}


def test_slip_with_no_legal_alternative_keeps_the_intended_action():
    # a 1x2 corridor: from (0,0) the only legal action at all is RIGHT. Slipping has
    # nowhere else to go, so the intended action must still proceed.
    g = GridWorld(size=2, start=(0, 0), goal=(1, 1), walls=frozenset({(1, 0)}),
                  slippery=frozenset({(0, 0)}), slip_prob=1.0, seed=0)
    g.reset()
    state, _, _, _ = g.step(RIGHT)
    assert state == (0, 1)


def test_transition_model_slip_outcomes_exclude_wall_bumps():
    g = GridWorld(size=3, start=(1, 0), goal=(2, 2), walls=frozenset({(1, 1)}),
                  slippery=frozenset({(1, 0)}), slip_prob=0.3)
    model = g.transition_model()
    outcomes = model[((1, 0), UP)]
    next_states = {s2 for _p, s2, _r, _d in outcomes}
    # only the intended UP->(0,0) and the one legal alternative DOWN->(2,0) -- never
    # a probability mass parked on (1,0) itself from a wall-bumping "alternative"
    assert next_states == {(0, 0), (2, 0)}
    assert abs(sum(p for p, *_ in outcomes) - 1.0) < 1e-9


def test_step_onto_shortcut_relocates_to_destination():
    # from (0,0), RIGHT lands on (0,1), a shortcut whose destination is (2,0) --
    # the agent should end up at the destination, never resting on the source.
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2), shortcuts={(0, 1): (2, 0)},
                  slip_prob=0.0)
    g.reset()
    state, _, _, _ = g.step(RIGHT)
    assert state == (2, 0)


def test_slip_onto_shortcut_relocates_to_destination():
    # at (1,1): UP is intended; DOWN and LEFT bump walls, so RIGHT->(1,2) is the only
    # legal slip alternative. (1,2) is a shortcut to (0,0), so every slip teleports there.
    g = GridWorld(size=3, start=(1, 1), goal=(2, 2),
                  walls=frozenset({(2, 1), (1, 0)}),
                  slippery=frozenset({(1, 1)}),
                  shortcuts={(1, 2): (0, 0)},
                  slip_prob=1.0, seed=0)
    outcomes = set()
    for _ in range(20):
        g.reset()
        state, _, _, _ = g.step(UP)
        outcomes.add(state)
    assert outcomes == {(0, 0)}


def test_transition_model_reflects_shortcut_teleport():
    # from (0,0), RIGHT would land on (0,1), a shortcut to (1,1). The model's outcome
    # for ((0,0), RIGHT) must show the destination, not the source cell.
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2), shortcuts={(0, 1): (1, 1)},
                  slip_prob=0.0)
    model = g.transition_model()
    next_states = {s2 for _p, s2, _r, _d in model[((0, 0), RIGHT)]}
    assert next_states == {(1, 1)}


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


def test_transition_model_excludes_actions_that_bump_a_wall_or_boundary():
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2), walls=frozenset({(0, 1)}))
    model = g.transition_model()
    # from (0,0): RIGHT bumps the wall, UP and LEFT bump the boundary -- none legal
    assert ((0, 0), RIGHT) not in model
    assert ((0, 0), UP) not in model
    assert ((0, 0), LEFT) not in model
    # DOWN actually moves the agent, so it stays available
    assert ((0, 0), DOWN) in model


def test_every_non_terminal_state_has_at_least_one_legal_action():
    g = GridWorld(size=4, walls=frozenset({(0, 1), (1, 1)}))
    model = g.transition_model()
    actions_by_state: dict = {}
    for (s, a) in model:
        actions_by_state.setdefault(s, []).append(a)
    for s in g.all_states():
        if not g.is_terminal(s):
            assert actions_by_state.get(s), f"{s} has no legal action"


def test_random_layout_avoids_start_and_goal():
    walls, slippery, traps = random_layout(size=10, start=(0, 0), goal=(9, 9),
                                            n_slippery=8, n_traps=2, n_walls=3, seed=1)
    placed = walls | slippery | traps
    assert (0, 0) not in placed
    assert (9, 9) not in placed
    assert len(walls) == 3
    assert len(slippery) == 8
    assert len(traps) == 2


def test_run_episode_returns_rewards_alongside_the_path():
    g = GridWorld(size=2, start=(0, 0), goal=(0, 1), goal_reward=5.0, step_reward=0.0,
                  slip_prob=0.0)
    policy = {(0, 0): RIGHT}
    path, rewards, steps, success = run_episode(g, policy, max_steps=10)
    assert path == [(0, 0), (0, 1)]
    assert rewards == [5.0]
    assert steps == 1
    assert success is True


def test_run_episode_times_out_without_reaching_goal():
    g = GridWorld(size=3, start=(0, 0), goal=(2, 2), slip_prob=0.0)
    policy = {s: UP for s in g.all_states()}  # always bounces against the top edge
    path, rewards, steps, success = run_episode(g, policy, max_steps=5)
    assert steps == 5
    assert success is False
    assert len(rewards) == 5


def test_step_after_terminal_raises():
    g = GridWorld(size=2, start=(0, 0), goal=(0, 1))
    g.reset()
    g.step(RIGHT)
    try:
        g.step(DOWN)
        assert False, "expected ValueError"
    except ValueError:
        pass
