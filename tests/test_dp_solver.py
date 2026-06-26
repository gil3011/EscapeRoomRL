from engine.dp_solver import (
    expected_steps_to_absorption, policy_iteration, uniform_policy_dist, value_iteration,
)
from engine.grid_world import DOWN, RIGHT, GridWorld


def test_value_iteration_matches_hand_computation():
    # 2x2 grid, start=(0,0), goal=(0,1), no slip, no step cost.
    # By hand: V(0,0)=1.0 (one step right to goal), V(1,1)=1.0 (one step up to goal),
    # V(1,0)=0.9 (one step from either of the above, discounted once).
    grid = GridWorld(size=2, start=(0, 0), goal=(0, 1), step_reward=0.0,
                      goal_reward=1.0, trap_reward=-1.0, slip_prob=0.0)
    model = grid.transition_model()
    states = grid.all_states()

    history = value_iteration(model, states, gamma=0.9, theta=1e-8, max_iterations=500)
    V = history[-1]["V"]

    assert abs(V[(0, 0)] - 1.0) < 1e-4
    assert abs(V[(1, 1)] - 1.0) < 1e-4
    assert abs(V[(1, 0)] - 0.9) < 1e-4
    assert history[-1]["policy"][(0, 0)] == RIGHT


def test_policy_iteration_matches_value_iteration():
    grid = GridWorld(size=2, start=(0, 0), goal=(0, 1), step_reward=0.0,
                      goal_reward=1.0, trap_reward=-1.0, slip_prob=0.0)
    model = grid.transition_model()
    states = grid.all_states()

    vi_V = value_iteration(model, states, gamma=0.9, theta=1e-8, max_iterations=500)[-1]["V"]
    pi_V = policy_iteration(model, states, gamma=0.9, theta=1e-8, max_iterations=500)[-1]["V"]

    for s in states:
        assert abs(vi_V[s] - pi_V[s]) < 1e-3


def test_v_decays_with_distance_from_goal_when_no_step_reward():
    # 5x5 open grid, goal at the far end of row 0 — with step_reward=0, V should
    # strictly increase as cells get closer to the goal, purely from discounting.
    grid = GridWorld(size=5, start=(0, 0), goal=(0, 4), step_reward=0.0,
                      goal_reward=1.0, trap_reward=-1.0, slip_prob=0.0)
    model = grid.transition_model()
    states = grid.all_states()

    V = value_iteration(model, states, gamma=0.9, theta=1e-8, max_iterations=500)[-1]["V"]

    row0 = [(0, j) for j in range(4)]  # (0,4) is the goal itself, excluded
    values = [V[s] for s in row0]
    assert values == sorted(values), "V should increase monotonically toward the goal"


def test_expected_steps_one_step_to_goal():
    grid = GridWorld(size=2, start=(0, 0), goal=(0, 1), step_reward=0.0, slip_prob=0.0)
    model = grid.transition_model()
    states = grid.all_states()

    always_right = lambda _s, _actions: {RIGHT: 1.0}
    T = expected_steps_to_absorption(model, states, always_right, theta=1e-8)
    assert abs(T[(0, 0)] - 1.0) < 1e-4


def test_expected_steps_unaffected_by_a_non_terminal_trap():
    # a policy that walks RIGHT then DOWN passes directly through (0,1) on the way
    # to the goal at (1,1). Marking (0,1) a trap shouldn't change the step count --
    # traps no longer end the episode, only the reward is affected.
    plain = GridWorld(size=2, start=(0, 0), goal=(1, 1), step_reward=0.0, slip_prob=0.0)
    trapped = GridWorld(size=2, start=(0, 0), goal=(1, 1), traps=frozenset({(0, 1)}),
                         step_reward=0.0, slip_prob=0.0)

    right_then_down = lambda s, _actions: {RIGHT: 1.0} if s == (0, 0) else {DOWN: 1.0}
    for g in (plain, trapped):
        model = g.transition_model()
        states = g.all_states()
        T = expected_steps_to_absorption(model, states, right_then_down, theta=1e-8)
        assert abs(T[(0, 0)] - 2.0) < 1e-4


def test_expected_steps_random_policy_is_larger_than_direct_policy():
    grid = GridWorld(size=5, start=(0, 0), goal=(0, 4), step_reward=0.0, slip_prob=0.0)
    model = grid.transition_model()
    states = grid.all_states()

    always_right = lambda _s, _actions: {RIGHT: 1.0}
    direct = expected_steps_to_absorption(model, states, always_right, theta=1e-6)
    random_walk = expected_steps_to_absorption(model, states, uniform_policy_dist, theta=1e-6)

    assert direct[(0, 0)] < random_walk[(0, 0)]
