"""G and V — the two numbers used to judge how well a room's agent is doing.

G is the realized, discounted return of one specific episode: "how much reward did
THIS run actually get?" V is the value function learned during training (or its
model-free analogue, max_a Q(s,a)): "how much reward did training predict this state
is worth?" Comparing a sampled G against the trained V is the standard way to sanity
-check a learned value function — no separate display-only score needed. Because
reward is discounted by gamma each step, a faster episode keeps more of a delayed
goal reward, so a faster solve already shows up as a higher G with no extra
bookkeeping (see plan.md §2.2).
"""


def discounted_return(rewards: list[float], gamma: float) -> float:
    """G = sum_t gamma^t * r_t, for one episode's sequence of rewards."""
    return sum((gamma ** t) * r for t, r in enumerate(rewards))
