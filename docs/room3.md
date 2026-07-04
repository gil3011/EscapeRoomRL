## The On-Policy Corridor — SARSA

Same 10x10 grid feel as Room 1, but the agent is now working **blind to the model**.
In Room 1 the full transition model was handed over and the optimal policy was computed
directly from the Bellman equation — no moving required. Here the agent only ever calls
`step(action)` and sees what happens: the next cell, the reward, whether the episode
ended. It never gets to peek at the slip probabilities or the reward map. It has to learn
the value of each move, `Q(s, a)`, purely from experience — one step at a time.

"Unknown model" doesn't mean the environment is hidden from *you*; it means the
*algorithm* never reads it. Everything the agent knows, it learned by trying.

### SARSA — learning from the action you actually take

SARSA updates `Q(s, a)` after every single step, using the transition it just lived
through — `(s, a, r, s', a')`, the five things the name is built from:

```
Q(s, a) ← Q(s, a) + α · [ r + γ·Q(s', a') − Q(s, a) ]
```

The term in brackets is the **TD-error**: the gap between what the agent *thought* the
move was worth, `Q(s, a)`, and a fresh one-step estimate, `r + γ·Q(s', a')`. Nudge the
old value a fraction `α` of the way toward the new estimate, and repeat forever.

The key detail is `a'`: it's the **next action the ε-greedy policy actually samples** at
`s'` — not the best possible action. That's what makes SARSA **on-policy**: it learns the
value of the policy it is really running, exploration mistakes and all. If exploring near
a trap sometimes costs it, SARSA folds that risk into the values and learns a slightly
more cautious route. (Room 4's Q-learning will instead bootstrap off `max_a Q(s', a)` —
the best action, whether or not it takes it — which is the one line that separates
off-policy from on-policy. The Info tab there lays them side by side.)

`ε-greedy` is the exploration knob: with probability ε take a random move, otherwise the
current best. ε starts high (explore the unknown grid) and decays toward a small value
(exploit what's been learned). Only *legal* moves — the ones that actually change the
agent's cell — are ever chosen; with no step cost, bumping a wall would be a free no-op,
so it's simply left out, the same way Room 1's policy never walks into a wall.

### The shortcut tile 🌀 — something to discover

New to this room: one **shortcut tile**. Step onto it — on purpose, or because a slip
pushed you there — and you're instantly relocated across the board to its paired
destination (◎). Nothing tells the agent it exists; it's just another thing to learn from
experience. On this board the shortcut cuts the fastest route from 18 steps down to 11,
so a well-trained policy learns to steer *onto* the teleport, exactly the way it learns to
route around the trap. Watch the checkpoint slider on the Board tab: early on the arrows
ignore it, and later they funnel toward it.

### Board features

- **Walls** (dark): blocked cells; moving into one or off the edge is a no-op.
- **Slippery cells** (blue): with probability `slip_prob`, a random *legal* direction is
  taken instead of the chosen one.
- **Traps** (red): stepping here costs `trap_reward`, but doesn't end the episode.
- **Shortcut** (teal 🌀 → ◎): stepping on the source teleports the agent to the destination.
- **Goal** (green 🚪): the only terminal cell; stepping here ends the episode and pays out.

The reward model matches Room 1 exactly: **no step cost**, a fixed high goal reward. With
γ < 1, discounting alone makes closer-to-the-goal cells worth more (`V(s) ≈ γ^d · goal`),
so a shorter escape is automatically a higher-value one — no artificial per-step penalty
needed. That also keeps V(start) on the same scale as Room 1 for the lobby scoreboard.

### Scoring — V and G

- **V(start)** = `max_a Q(start, a)` from the learned Q-table: what training predicts the
  start cell is worth. Stable for a given config, so it's what the lobby scoreboard shows.
- **G** = the actual discounted return `Σ γ^t r_t` of one escape-attempt rollout. One noisy
  sample of what V predicts on average; shown next to V on the Board tab for comparison.
  A few attempts' G values clustering around V is a sign the learned values are accurate.

### Parameters

| Parameter | What it controls |
|---|---|
| Slip probability | Chance a slippery cell substitutes a random legal action |
| Trap reward | Penalty for stepping on a trap (doesn't end the episode) |
| Learning rate α | How far each step nudges Q toward the new TD estimate |
| Decay α | If on, α ramps down to a tenth over training (like ε) |
| Gamma (γ) | Discount factor — how much a delayed reward is worth now |
| Episodes | How many full episodes to train for |
| Max steps / episode | Safety cap so an early random policy can't wander forever |
| ε start / end | Exploration rate at the beginning and end of training |
| ε decay fraction | Fraction of episodes over which ε linearly decays to ε end |
| Snapshot interval | How often to checkpoint the policy for Board-tab replay |
| Max attempt steps | Safety cap on a single escape-attempt rollout |

Goal reward and the board layout are fixed and not adjustable.
