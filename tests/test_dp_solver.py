from engine.dp_solver import policy_iteration, value_iteration
from engine.grid_world import RIGHT, GridWorld


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
