from engine.boards import ROOM3_SHORTCUT_DST, ROOM3_SHORTCUT_SRC, make_room3_grid
from engine.grid_world import GridWorld, run_episode
from engine.sarsa_agent import train_sarsa


def test_sarsa_learns_to_reach_goal_on_a_small_deterministic_grid():
    g = GridWorld(size=4, start=(0, 0), goal=(3, 3), slip_prob=0.0)
    result = train_sarsa(g, episodes=500, gamma=0.9, alpha=0.5, epsilon_start=1.0,
                         epsilon_end=0.01, epsilon_decay_fraction=0.7, max_steps=100,
                         snapshot_interval=100, seed=0)
    path, _rewards, _steps, success = run_episode(g, result["policy"], max_steps=100)
    assert success
    assert result["v_start"] > 0


def test_sarsa_solves_room3_and_uses_the_shortcut():
    # slip-free so the greedy rollout is deterministic; enough episodes to converge on
    # 100 states. The optimal route runs through the shortcut, so a solved policy that
    # reaches the goal quickly must be routing the agent onto the teleport.
    grid = make_room3_grid(slip_prob=0.0)
    result = train_sarsa(grid, episodes=2500, gamma=0.9, alpha=0.3, epsilon_start=1.0,
                         epsilon_end=0.02, epsilon_decay_fraction=0.8, max_steps=200,
                         snapshot_interval=200, seed=0)
    path, _rewards, steps, success = run_episode(grid, result["policy"], max_steps=200)
    assert success
    # the shortcut destination should appear in the trajectory (agent took the teleport),
    # and the source never does (it's relocated away from on arrival).
    assert ROOM3_SHORTCUT_DST in path
    assert ROOM3_SHORTCUT_SRC not in path
    assert steps < 18  # the no-shortcut shortest path is 18; using the teleport beats it


def test_epsilon_and_alpha_decay_over_training():
    g = GridWorld(size=4, start=(0, 0), goal=(3, 3), slip_prob=0.0)
    result = train_sarsa(g, episodes=200, alpha=0.4, decay_alpha=True, epsilon_start=1.0,
                         epsilon_end=0.05, epsilon_decay_fraction=0.5, max_steps=50, seed=0)
    eps = result["history"]["epsilon"]
    alpha = result["history"]["alpha"]
    assert eps[0] == 1.0 and eps[-1] == 0.05
    assert alpha[0] == 0.4 and alpha[-1] < 0.4  # decayed toward alpha/10


def test_history_and_checkpoints_have_matching_lengths():
    g = GridWorld(size=4, start=(0, 0), goal=(3, 3), slip_prob=0.0)
    result = train_sarsa(g, episodes=150, snapshot_interval=50, max_steps=50, seed=0)
    assert result["episodes_run"] == 150
    for key in ("reward", "steps", "td_error", "epsilon", "alpha"):
        assert len(result["history"][key]) == 150
    # snapshots at 50, 100, 150 -- and 150 is both a multiple and the final episode,
    # so it must not be double-counted.
    episodes = [c["episode"] for c in result["checkpoints"]]
    assert episodes == [50, 100, 150]


def test_stop_flag_halts_training_immediately():
    g = GridWorld(size=4, start=(0, 0), goal=(3, 3), slip_prob=0.0)
    result = train_sarsa(g, episodes=1000, max_steps=50, stop_flag_ref=[True], seed=0)
    assert result["episodes_run"] == 0
