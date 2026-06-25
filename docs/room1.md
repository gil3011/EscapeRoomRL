## The Frozen Vault — Dynamic Programming

A 10x10 grid where the **model is known**: every cell's neighbors, slip probability, and
reward are fully specified up front. That means there's nothing to explore — the optimal
policy can be computed directly by solving the Bellman equation, with no trial and error.
Rooms 2-4 will face the same kind of grid without being allowed to see the model; this room
is the baseline "what's possible when you already know everything" case.

### The board

A fixed, hand-designed layout (not regenerated between runs):

- **Walls** (gray): 13 cells scattered across 5 small clusters. Moving into one, or off the
  edge of the board, is illegal — you just don't move.
- **Slippery cells** (blue): with probability `slip_prob`, the action that's actually taken
  is a random one of the other three instead of the one you chose.
- **Traps** (red): stepping here ends the episode with `trap_reward` (a large penalty).
- **Goal** (green): stepping here ends the episode with `goal_reward` (a large bonus).

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

### Reward model — no step reward

`R(s,a,s')` is exactly **0** for every move that doesn't end the episode. The only nonzero
rewards are `goal_reward` and `trap_reward`, paid once, on the step that ends things. There's
no extra per-step penalty bolted on to make the agent hurry — it doesn't need one. Because
future reward is discounted by `gamma` for every step it's delayed, a higher `gamma` means
the goal's pull reaches farther across the board, and a lower `gamma` makes that pull fade to
almost nothing a few steps away. Watch the V-heatmap change shape as you move the `gamma`
slider — that's the entire mechanism, with nothing else mixed in.

### Parameters

| Parameter | What it controls |
|---|---|
| Slip probability | Chance a slippery cell substitutes a random action for the one chosen |
| Goal reward | Payout for reaching the goal — kept high so it stays visible after discounting |
| Trap reward | Penalty for stepping on a trap |
| Method | Value Iteration or Policy Iteration |
| Gamma (γ) | Discount factor — how much a delayed reward is worth today |
| Theta (θ) | Convergence threshold — stop when the biggest change in V drops below this |
| Max iterations | Safety cap; real convergence happens well before this on a 10x10 grid |
| Max attempt steps | Safety cap on a single escape-attempt rollout |

### Scoring

The **Escape Score** shown after an escape attempt compares the steps actually taken against
`par_steps` — the expected number of steps a *uniformly random* policy would take to reach
*some* terminal cell (goal or trap) on this exact board. Comparing against a random walker
rather than against the optimal policy's own time is what makes a well-trained run's score
look good (close to 1000) instead of comparing the agent to itself. A trap or a timeout scores
0 regardless of how many steps were taken — only reaching the goal counts as success. The Train
tab also shows the *optimal* policy's own expected steps next to the random baseline, so you
can see directly how much better-than-random the solved policy is.
