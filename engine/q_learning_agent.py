"""Q-learning — off-policy TD(0) control for the grid rooms, model-free.

Same shape as SARSA (Room 3) and reusing its helpers — the *only* substantive
difference is the TD target:

    SARSA (on-policy):   Q(s,a) ← Q(s,a) + α[r + γ·Q(s', a') − Q(s,a)]   a' = sampled
    Q-learning (off):    Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s', a') − Q(s,a)]

SARSA bootstraps off the next action its ε-greedy policy *actually takes*, so it learns
the value of the policy it runs (exploration risk included). Q-learning bootstraps off
the *best* next action regardless of what it does next, so it learns the optimal policy's
values directly. Against Room 4's patrol that difference becomes visible: Q-learning
learns the optimal, tightly-timed route past the enemy, where an on-policy learner would
keep a safety margin because ε-exploration near the patrol risks a fatal step.

Adapted from `code examples/q_learning.py`, retargeted onto GridWorld's API and reused
unchanged for PatrolGridWorld's augmented `(cell, phase)` states (states are opaque here).
"""
from __future__ import annotations

import random

from engine.grid_world import ACTIONS, run_episode
from engine.sarsa_agent import _epsilon_greedy, _linear_decay, greedy_policy, q_values


def train_q_learning(grid, *, episodes: int, gamma: float = 0.9, alpha: float = 0.2,
                     decay_alpha: bool = False, epsilon_start: float = 1.0,
                     epsilon_end: float = 0.05, epsilon_decay_fraction: float = 0.8,
                     max_steps: int = 200, snapshot_interval: int = 50, seed: int | None = None,
                     emit=None, stop_flag_ref=None) -> dict:
    """Train Q-learning on `grid` and return a result dict:
        {history, checkpoints, Q, policy, values, v_start, episodes_run}
    history holds per-episode {reward, steps, td_error, epsilon, alpha, outcome} lists,
    where outcome is "goal" / "caught" / "timeout" (read from step()'s info) so the Train
    tab can chart the escape/caught/timeout breakdown. Same emit/stop_flag_ref contract as
    train_sarsa.
    """
    rng = random.Random(seed)
    states = grid.all_states()
    Q = {s: {a: 0.0 for a in ACTIONS} for s in states}
    legal_actions = {s: grid.legal_actions(s) for s in states if not grid.is_terminal(s)}

    history: dict[str, list] = {"reward": [], "steps": [], "td_error": [],
                                "epsilon": [], "alpha": [], "outcome": []}
    checkpoints: list[dict] = []
    alpha_end = alpha * 0.1

    def snapshot(episode_num: int) -> None:
        policy = greedy_policy(Q, legal_actions)
        values = q_values(Q, legal_actions)
        path, _rewards, _steps, success = run_episode(grid, policy, max_steps)
        checkpoints.append({"episode": episode_num, "policy": policy, "values": values,
                            "trajectory": path, "success": success})

    for ep in range(episodes):
        if stop_flag_ref is not None and stop_flag_ref[0]:
            break
        eps = _linear_decay(epsilon_start, epsilon_end, epsilon_decay_fraction, ep, episodes)
        lr = (_linear_decay(alpha, alpha_end, epsilon_decay_fraction, ep, episodes)
              if decay_alpha else alpha)

        s = grid.reset()
        ep_reward = 0.0
        td_sum = 0.0
        steps = 0
        outcome = "timeout"
        for _ in range(max_steps):
            a = _epsilon_greedy(Q[s], legal_actions[s], eps, rng)  # behaviour policy
            s2, r, done, info = grid.step(a)
            steps += 1
            ep_reward += r
            if done:
                td = r - Q[s][a]  # terminal: no bootstrap
                Q[s][a] += lr * td
                td_sum += abs(td)
                outcome = info.get("outcome", "goal")  # plain grids only end on the goal
                break
            best_next = max(Q[s2][a2] for a2 in legal_actions[s2])  # off-policy: greedy max
            td = r + gamma * best_next - Q[s][a]
            Q[s][a] += lr * td
            td_sum += abs(td)
            s = s2

        history["reward"].append(ep_reward)
        history["steps"].append(steps)
        history["td_error"].append(td_sum / steps if steps else 0.0)
        history["epsilon"].append(eps)
        history["alpha"].append(lr)
        history["outcome"].append(outcome)

        if snapshot_interval and (ep + 1) % snapshot_interval == 0:
            snapshot(ep + 1)
            if emit is not None:
                emit("metrics", {"history": history, "checkpoints": checkpoints})

    episodes_run = len(history["reward"])
    if episodes_run and (not checkpoints or checkpoints[-1]["episode"] != episodes_run):
        snapshot(episodes_run)

    policy = greedy_policy(Q, legal_actions)
    values = q_values(Q, legal_actions)
    result = {
        "history": history,
        "checkpoints": checkpoints,
        "Q": Q,
        "policy": policy,
        "values": values,
        "v_start": values[grid.initial_state()],
        "episodes_run": episodes_run,
    }
    if emit is not None:
        emit("result", result)
    return result
