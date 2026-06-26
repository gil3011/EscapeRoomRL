"""Dynamic Programming over a known GridWorld transition model — Room 1.

Adapted from code examples/value_iteration.py and policy_iteration_deterministic.py,
retargeted from their flat (s,a,s')-keyed dicts onto GridWorld.transition_model()'s
{(s,a): [(prob, s2, r, done), ...]} shape. Both value_iteration and policy_iteration
are the Bellman equation, applied in-place exactly like the examples (each state's
update can see other states already updated earlier in the same sweep).
"""
from __future__ import annotations


def _actions_by_state(model: dict) -> dict:
    actions_by_state: dict = {}
    for (s, a) in model:
        actions_by_state.setdefault(s, []).append(a)
    return actions_by_state


def _action_value(model: dict, V: dict, s, a: int, gamma: float) -> float:
    return sum(p * (r + gamma * V[s2]) for p, s2, r, _done in model[(s, a)])


def value_iteration(model: dict, states: list, gamma: float, theta: float = 1e-3,
                     max_iterations: int = 200) -> list[dict]:
    """Bellman optimality equation: V(s) <- max_a sum_s' P(s'|s,a)[R + gamma*V(s')].

    Returns the full per-iteration history (cheap on a 10x10 grid):
    [{"V": {...}, "policy": {...}, "delta": float, "policy_changes": int}, ...]
    """
    actions_by_state = _actions_by_state(model)
    V = {s: 0.0 for s in states}
    policy: dict = {}
    history = []

    for _ in range(max_iterations):
        delta = 0.0
        prev_policy = dict(policy)
        for s, actions in actions_by_state.items():
            old_v = V[s]
            best_a, best_v = None, float("-inf")
            for a in actions:
                v = _action_value(model, V, s, a, gamma)
                if v > best_v:
                    best_v, best_a = v, a
            V[s] = best_v
            policy[s] = best_a
            delta = max(delta, abs(old_v - best_v))

        policy_changes = sum(1 for s in policy if prev_policy.get(s) != policy[s])
        history.append({"V": dict(V), "policy": dict(policy), "delta": delta,
                         "policy_changes": policy_changes})
        if delta < theta:
            break

    return history


def policy_iteration(model: dict, states: list, gamma: float, theta: float = 1e-3,
                      max_iterations: int = 200) -> list[dict]:
    """Alternates the Bellman expectation equation (policy evaluation, swept to
    convergence under the current fixed policy) with greedy policy improvement,
    until no state's action changes. One history entry per outer (improvement) step.
    """
    actions_by_state = _actions_by_state(model)
    non_terminal = list(actions_by_state.keys())
    V = {s: 0.0 for s in states}
    policy = {s: actions_by_state[s][0] for s in non_terminal}
    history = []

    for _ in range(max_iterations):
        eval_delta = theta
        for _ in range(max_iterations):
            eval_delta = 0.0
            for s in non_terminal:
                old_v = V[s]
                v = _action_value(model, V, s, policy[s], gamma)
                V[s] = v
                eval_delta = max(eval_delta, abs(old_v - v))
            if eval_delta < theta:
                break

        new_policy = {}
        for s in non_terminal:
            best_a, best_v = None, float("-inf")
            for a in actions_by_state[s]:
                v = _action_value(model, V, s, a, gamma)
                if v > best_v:
                    best_v, best_a = v, a
            new_policy[s] = best_a

        policy_changes = sum(1 for s in non_terminal if policy[s] != new_policy[s])
        policy = new_policy
        history.append({"V": dict(V), "policy": dict(policy), "delta": eval_delta,
                         "policy_changes": policy_changes})
        if policy_changes == 0:
            break

    return history
