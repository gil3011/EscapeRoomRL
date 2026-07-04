## The Frozen Vault — Dynamic Programming

A 10x10 grid where the **model is known**: every cell's neighbors, slip probability, and
reward are fully specified up front. Get from the start to the goal, choosing the best move
at every cell. Because the model is known, the optimal policy can be computed directly by
solving the Bellman equation — no trial and error needed.

- **Walls** (dark): blocked cells. Moving into one, or off the edge of the board, is illegal.
- **Slippery cells** (blue): with probability `slip_prob`, a random *legal* direction is taken
  instead of the one chosen.
- **Traps** (red): stepping here costs `trap_reward`, but doesn't end the episode.
- **Goal** (green): stepping here ends the episode and pays out the goal reward.

### Parameters

| Parameter | What it controls |
|---|---|
| Slip probability | Chance a slippery cell substitutes a random action for the one chosen |
| Trap reward | Penalty for stepping on a trap (doesn't end the episode) |
| Method | Value Iteration or Policy Iteration |
| Gamma (γ) | Discount factor — how much a delayed reward is worth today |
| Max attempt steps | Safety cap on a single escape-attempt rollout |

Goal reward, theta (convergence threshold), and max iterations are fixed and not adjustable.
