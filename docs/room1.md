## The Frozen Vault — Dynamic Programming

A 10x10 grid where the **model is known**: every cell's neighbors, slip probability, and
reward are fully specified up front. That means there's nothing to explore — the optimal
policy can be computed directly by solving the Bellman equation, with no trial and error.
Rooms 2-4 will face the same kind of grid without being allowed to see the model; this room
is the baseline "what's possible when you already know everything" case.

### The board

A fixed, hand-designed layout (not regenerated between runs):

- **Walls** (gray): 13 cells scattered across 5 small clusters. Moving into one, or off the
  edge of the board, is illegal — you just don't move. The optimal policy is never even
  *offered* these moves as an option (see "No walking into walls" below), so it can't end up
  recommending one by accident.
- **Slippery cells** (blue): with probability `slip_prob`, the action that's actually taken
  is a random one of the other three instead of the one you chose — including, unluckily,
  one that bumps into a wall. That's a property of the ice, not a choice the policy makes.
- **Traps** (red): stepping here costs `trap_reward` (a large penalty) but does **not** end
  the episode — it's an expensive cell to pass through, not an instant failure. Step on one
  twice and you pay twice.
- **Goal** (green): stepping here ends the episode and pays out the goal reward.

### The Bellman equation

Every value in this room comes from one equation, used two ways:

**Bellman optimality equation** (Value Iteration) — repeatedly applying this to every cell
converges to the value of playing *optimally* from there:

```
V(s) <- max_a  sum_s'  P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]
```

**Bellman expectation equation** (Policy Iteration's evaluation step) — the same equation
with the `max` replaced by a *fixed* policy's choice, used to score a specific policy before
improving it:

```
V(s) <- sum_s'  P(s'|s, pi(s)) * [ R(s,a,s') + gamma * V(s') ]
```

Policy Iteration alternates: evaluate the current policy fully (repeat the second equation
until it stops changing), then improve it (switch each cell to whichever action the first
equation says is best), and repeat until the policy itself stops changing. It usually needs
far fewer *outer* steps than Value Iteration needs sweeps — but each outer step does a full
inner evaluation, so the iteration-replay slider on the Board tab will have noticeably fewer
points for Policy Iteration than for Value Iteration. That's expected, not a bug.

### Reward model — no step reward, and the goal reward is fixed

`R(s,a,s')` is exactly **0** for every move that doesn't end the episode or land on a trap.
There's no per-step penalty bolted on to make the agent hurry — it doesn't need one. Because
future reward is discounted by `gamma` for every step it's delayed, a higher `gamma` means
the goal's pull reaches farther across the board, and a lower `gamma` makes that pull fade to
almost nothing a few steps away. Watch the V-heatmap change shape as you move the `gamma`
slider — that's the entire mechanism, with nothing else mixed in.

The goal reward itself is **fixed at a high value and not adjustable** — that's deliberate.
Since it's the only thing (along with `gamma`) shaping V(s) across the whole board, keeping it
high keeps the heatmap's value spread visible instead of collapsing toward zero everywhere.

### No walking into walls

The set of actions Value/Policy Iteration even consider at each cell excludes any action that
would just bump into a wall or off the edge of the board — those moves are never offered, so
`max_a` can never select one. The learned policy will never *deliberately* point into a wall.
(Slip can still accidentally land you on a bump as a random side effect of a legal move near a
slippery cell — that's the ice being unpredictable, not the policy making a bad choice.)

### Parameters

| Parameter | What it controls |
|---|---|
| Slip probability | Chance a slippery cell substitutes a random action for the one chosen |
| Trap reward | Penalty for stepping on a trap (doesn't end the episode) |
| Method | Value Iteration or Policy Iteration |
| Gamma (γ) | Discount factor — how much a delayed reward is worth today |
| Theta (θ) | Convergence threshold — stop when the biggest change in V drops below this |
| Max iterations | Safety cap; real convergence happens well before this on a 10x10 grid |
| Max attempt steps | Safety cap on a single escape-attempt rollout |

Goal reward is intentionally **not** in this list — it's fixed (shown read-only in the sidebar).

### Scoring

The **Escape Score** shown after an escape attempt compares the steps actually taken against
`par_steps` — the expected number of steps a *uniformly random* policy would take to reach the
goal on this exact board. Since traps don't end the episode, the only thing a random walk can
end on is the goal itself — which, undirected, typically takes hundreds of steps on this board,
making a trained policy's score climb close to 1000. Only timing out without reaching the goal
scores 0; wandering through a trap along the way still counts as success if the goal is reached
within the step limit, just with a reward penalty already paid. The Train tab also shows the
*optimal* policy's own expected steps next to the random baseline, so you can see directly how
much better-than-random the solved policy is.
