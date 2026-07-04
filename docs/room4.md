## The Off-Policy Cellar — Q-Learning

The hardest grid room, and the twin of Room 3. Same model-free setup — the agent only
calls `step()` and learns `Q(s, a)` from experience — but two things are new: the
**algorithm** (off-policy Q-learning instead of on-policy SARSA) and the **board** (a
moving patrol enemy and a trap door).

### SARSA vs. Q-learning — one line of difference

Both update `Q(s, a)` toward a one-step estimate `r + γ·(value of the next state)`. The
only difference is *which* next-state value they bootstrap off:

```
SARSA (on-policy):    Q(s,a) ← Q(s,a) + α·[ r + γ·Q(s', a') − Q(s,a) ]     a' = the action ε-greedy actually takes
Q-learning (off):     Q(s,a) ← Q(s,a) + α·[ r + γ·max_a' Q(s', a') − Q(s,a) ]   the best action, taken or not
```

- **SARSA** bootstraps off the action it *actually* takes next — including the random
  ε-greedy exploration steps. So it learns the value of the policy it is really running,
  exploration risk and all, and ends up **cautious** near danger.
- **Q-learning** bootstraps off the *best* next action regardless of what it does next. So
  it learns the **optimal** policy's values directly, even while behaving ε-greedily — and
  ends up cutting things close, because the value it's learning assumes it will act
  optimally from here on.

When acting carries no risk (Room 3's board without the enemy, no slip) the two converge
to the *same* policy — the difference only bites when exploration can hurt. Which is
exactly what this room's board provides.

### The moving patrol enemy 👾 — and why the state has to grow

A 👾 patrols up and down the gap in the wall barrier (column 5), moving one cell every
step. **Touching it ends the episode** with the `enemy_reward` penalty — the first room
you can actually *lose*, not just time out.

A moving hazard breaks something subtle: if the agent only knew its own cell, the same
cell would be safe on one visit and lethal on another, depending on where the patrol
happens to be. That's not a Markov state — the future isn't predictable from it. The fix
is to **augment the state to `(agent_cell, enemy_phase)`**, where `enemy_phase` is the
patrol's position in its fixed cycle. The enemy moves deterministically, so its phase is
all the extra information the agent needs. That's the room's second lesson: *a dynamic
hazard forces you to grow the state.* It's why the Board tab has an **enemy-phase slider** —
the value of a cell genuinely depends on where the patrol is.

Because the optimal route crosses the patrol's gap, escaping means **timing** the crossing
for a moment the 👾 is elsewhere. Q-learning learns to thread it tightly; a SARSA-trained
agent would leave a bigger safety margin, because its ε-greedy exploration near the patrol
occasionally walks straight in.

### The trap door 🕳️ — a bad teleport

There's a tempting-looking cell just past the chokepoint that is actually a **trap door**:
step on it (on purpose or via a slip) and you're flung backward toward the start (↩). It
looks like it's on a fast route; the agent has to learn it's a mistake — the mirror image
of Room 3's *helpful* shortcut.

### Board features

- **Walls** (dark): the column-5 barrier leaves a 4-cell gap (rows 3-6) — the only crossing.
- **Slippery cells** (blue): random legal move with probability `slip_prob` (kept away from
  the gap, so the crossing risk is about *timing*, not luck).
- **Traps** (red): a non-terminal cost, as before.
- **Trap door** (🕳️ → ↩, red): teleports the agent backward.
- **Patrol enemy** (👾, red trail): moving, terminal-on-contact.
- **Goal** (green 🚪): the only success-terminal cell.

Reward model matches Rooms 1/3 — no step cost, fixed high goal reward, discounting shapes
V — plus the terminal `enemy_reward` on collision.

### Scoring — V and G

- **V(start)** = `max_a Q((start, phase 0), a)`. (Q-learning's `max` update gives it a mild
  optimism/overestimate — *maximization bias* — which is itself a known Q-learning trait.)
- **G** = the discounted return of one greedy escape rollout, shown next to V on the Board
  tab. A rollout that gets caught earns the big negative `enemy_reward`, so a low G is the
  signal of a mistimed run.

### Parameters

| Parameter | What it controls |
|---|---|
| Slip probability | Chance a slippery cell substitutes a random legal action |
| Trap reward | Non-terminal trap cost |
| Enemy reward | Terminal penalty for colliding with the patrol |
| Learning rate α | How far each step nudges Q toward the new TD estimate |
| Decay α | If on, α ramps down to a tenth over training |
| Gamma (γ) | Discount factor |
| Episodes | How many episodes to train (larger than Room 3 — the state space is bigger) |
| Max steps / episode | Safety cap per episode |
| ε start / end / decay fraction | Exploration schedule |
| Snapshot interval | How often to checkpoint for Board-tab replay |
| Max attempt steps | Safety cap on one escape-attempt rollout |

Goal reward, the board, and the patrol path are fixed.
