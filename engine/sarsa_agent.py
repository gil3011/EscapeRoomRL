"""SARSA — on-policy TD(0) control for the grid rooms, model-free.

Unlike Room 1's dynamic programming, this agent never touches GridWorld's
transition_model(): it learns Q(s,a) purely from the (s, a, r, s') transitions it
experiences by calling step(). "On-policy" means the TD target bootstraps off the
*next action the ε-greedy behaviour policy actually samples* (a'), not the greedy
max — so it learns the value of the policy it is really running, exploration risk
included. That single choice of a' is the one line that will differ from Room 4's
off-policy Q-learning (which bootstraps off max_a Q(s',a)).

Adapted from `code examples/sarsa.py`, retargeted onto GridWorld's tuple-state,
integer-action API. One deliberate difference from that reference: Room 3 has no step
cost (its reward model matches Room 1), which would make bumping a wall a free no-op,
so the behaviour policy chooses only among *legal* moves — the actions that actually
change the agent's cell (grid.legal_actions). The agent still learns every reward and
the slip dynamics entirely from experience; only dead-end moves are excluded, which
keeps the greedy policy from ever displaying a wall-bump arrow (the same property
Room 1 has). See docs/room3.md.
"""
from __future__ import annotations

import random

from engine.grid_world import ACTIONS, run_episode


def _greedy_action(q_s: dict, legal: list[int], rng: random.Random) -> int:
    best = max(q_s[a] for a in legal)
    return rng.choice([a for a in legal if q_s[a] == best])


def _epsilon_greedy(q_s: dict, legal: list[int], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.choice(legal)
    return _greedy_action(q_s, legal, rng)


def greedy_policy(Q: dict, legal_actions: dict) -> dict:
    """Deterministic greedy policy (state -> action) over legal moves. Ties break to
    the lowest action index so the rendered arrows are stable between reruns."""
    policy = {}
    for s, legal in legal_actions.items():
        best = max(Q[s][a] for a in legal)
        policy[s] = min(a for a in legal if Q[s][a] == best)
    return policy


def q_values(Q: dict, legal_actions: dict) -> dict:
    """State-value heatmap V(s) = max_a Q(s,a) over legal actions — Rooms 2-4's
    analogue of Room 1's exact V, derived from the learned Q-table."""
    return {s: max(Q[s][a] for a in legal) for s, legal in legal_actions.items()}


def _linear_decay(start: float, end: float, fraction: float, ep: int, total: int) -> float:
    """Linear ramp from `start` to `end` over the first `fraction` of episodes, then
    flat at `end` for the rest (used for both ε and, optionally, α)."""
    decay_span = max(1, int(fraction * total))
    if ep >= decay_span:
        return end
    return start + (end - start) * (ep / decay_span)


def train_sarsa(grid, *, episodes: int, gamma: float = 0.9, alpha: float = 0.2,
                decay_alpha: bool = False, epsilon_start: float = 1.0,
                epsilon_end: float = 0.05, epsilon_decay_fraction: float = 0.8,
                max_steps: int = 200, snapshot_interval: int = 50, seed: int | None = None,
                emit=None, stop_flag_ref=None) -> dict:
    """Train SARSA on `grid` and return a result dict:
        {history, checkpoints, Q, policy, values, v_start, episodes_run}
    where history holds per-episode {reward, steps, td_error, epsilon, alpha} lists and
    checkpoints is a list of {episode, policy, values, trajectory, success} taken every
    `snapshot_interval` episodes for the Board tab's replay slider.

    If `emit` is given (the TrainingRunner queue callback) it is called with
    ("metrics", {history, checkpoints}) at each snapshot and ("result", result) at the
    end, so the page can update live. If `stop_flag_ref[0]` goes True the loop exits
    early and still returns/represents whatever was learned so far.
    """
    rng = random.Random(seed)
    states = grid.all_states()
    Q = {s: {a: 0.0 for a in ACTIONS} for s in states}
    legal_actions = {s: grid.legal_actions(s) for s in states if not grid.is_terminal(s)}

    history: dict[str, list] = {"reward": [], "steps": [], "td_error": [],
                                "epsilon": [], "alpha": []}
    checkpoints: list[dict] = []
    alpha_end = alpha * 0.1  # when decay_alpha is on, α ramps down to a tenth

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
        a = _epsilon_greedy(Q[s], legal_actions[s], eps, rng)
        ep_reward = 0.0
        td_sum = 0.0
        steps = 0
        for _ in range(max_steps):
            s2, r, done, _info = grid.step(a)
            steps += 1
            ep_reward += r
            if done:
                td = r - Q[s][a]  # terminal: no bootstrap, Q(terminal, .) = 0
                Q[s][a] += lr * td
                td_sum += abs(td)
                break
            a2 = _epsilon_greedy(Q[s2], legal_actions[s2], eps, rng)
            td = r + gamma * Q[s2][a2] - Q[s][a]
            Q[s][a] += lr * td
            td_sum += abs(td)
            s, a = s2, a2

        history["reward"].append(ep_reward)
        history["steps"].append(steps)
        history["td_error"].append(td_sum / steps if steps else 0.0)
        history["epsilon"].append(eps)
        history["alpha"].append(lr)

        if snapshot_interval and (ep + 1) % snapshot_interval == 0:
            snapshot(ep + 1)
            if emit is not None:
                emit("metrics", {"history": history, "checkpoints": checkpoints})

    # always represent the final learned policy as a checkpoint, even if training
    # stopped between snapshot intervals (early stop, or episodes not a multiple).
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
        "v_start": values[grid.start],
        "episodes_run": episodes_run,
    }
    if emit is not None:
        emit("result", result)
    return result
