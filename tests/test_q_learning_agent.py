from engine.boards import make_room4_grid
from engine.grid_world import GridWorld, run_episode
from engine.q_learning_agent import train_q_learning
from engine.sarsa_agent import train_sarsa


def test_q_learning_solves_small_deterministic_grid():
    g = GridWorld(size=4, start=(0, 0), goal=(3, 3), slip_prob=0.0)
    result = train_q_learning(g, episodes=500, gamma=0.9, alpha=0.5, epsilon_start=1.0,
                              epsilon_end=0.01, epsilon_decay_fraction=0.7, max_steps=100,
                              snapshot_interval=100, seed=0)
    _path, _r, steps, success = run_episode(g, result["policy"], max_steps=100)
    assert success
    assert steps == 6  # Manhattan-optimal on a clear 4x4
    assert result["v_start"] > 0


def test_q_learning_and_sarsa_agree_without_stochasticity():
    # with no slip and no exploration risk, on-policy and off-policy TD(0) converge to
    # the same optimal policy — the update-rule difference only bites when acting is risky.
    g = GridWorld(size=5, start=(0, 0), goal=(4, 4), slip_prob=0.0)
    ql = train_q_learning(g, episodes=800, alpha=0.5, epsilon_end=0.01, seed=1, max_steps=100)
    sa = train_sarsa(g, episodes=800, alpha=0.5, epsilon_end=0.01, seed=1, max_steps=100)
    _p, _r, ql_steps, ql_ok = run_episode(g, ql["policy"], 100)
    _p, _r, sa_steps, sa_ok = run_episode(g, sa["policy"], 100)
    assert ql_ok and sa_ok
    assert ql_steps == sa_steps == 8


def test_outcome_is_recorded_and_the_enemy_can_catch():
    grid = make_room4_grid(slip_prob=0.25)
    result = train_q_learning(grid, episodes=300, max_steps=120, snapshot_interval=100, seed=0)
    outcomes = result["history"]["outcome"]
    assert len(outcomes) == 300
    assert set(outcomes) <= {"goal", "caught", "timeout"}
    assert "caught" in outcomes  # the moving enemy is a real, lethal hazard


def test_q_learning_threads_the_patrol_once_trained():
    # slip-free so the greedy rollout is deterministic; must time its crossing of the
    # column-5 gap so the patrol never catches it.
    grid = make_room4_grid(slip_prob=0.0)
    result = train_q_learning(grid, episodes=4000, gamma=0.9, alpha=0.3, epsilon_start=1.0,
                              epsilon_end=0.02, epsilon_decay_fraction=0.8, max_steps=200,
                              snapshot_interval=1000, seed=0)
    path, _r, steps, success = run_episode(grid, result["policy"], max_steps=200)
    assert success, "trained policy should reach the goal without being caught"
    assert steps <= 22  # near the 18-step optimal timed solve
    # the enemy cell is never the agent's cell at the same phase along the escape
    for cell, phase in path:
        assert cell != grid.enemy_cell(phase)


def test_stop_flag_halts_q_learning_immediately():
    grid = make_room4_grid()
    result = train_q_learning(grid, episodes=1000, max_steps=100, stop_flag_ref=[True], seed=0)
    assert result["episodes_run"] == 0
