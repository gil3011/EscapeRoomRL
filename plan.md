# EscapeRoomRL — Project Plan

A Streamlit app where a reinforcement-learning agent escapes a sequence of six rooms,
each teaching/demonstrating a different RL algorithm. The player (developer) tunes every
algorithm and environment parameter from a sidebar, watches training progress live, and
replays individual episodes afterward to see what the agent learned and when.

Reference material already in the repo (`code examples/`):
- Classic tabular RL scripts (`grid_world.py`, `value_iteration.py`, `policy_iteration_*.py`,
  `monte_carlo*.py`, `sarsa.py`, `q_learning.py`, `epsilon_greedy.py`) — algorithm logic reference.
- `code examples/dql/` — a full Streamlit + PyTorch DQN app (LunarLander). This is the UI/UX
  and architecture template: sidebar with grouped `st.expander`s, a background-thread training
  loop draining into a `queue.Queue` and `st.session_state`, Plotly charts, and an animated
  episode-replay slider. We reuse these patterns across all six rooms instead of reinventing them.

## 1. Locked design decisions

These were ambiguous in the spec and have been decided with you before writing this plan:

| Decision | Choice |
|---|---|
| Room count / Monte Carlo placement | **6 rooms.** Monte Carlo gets its own room (Room 2), inserted between DP and SARSA. One algorithm per room, as required. |
| Function approximation (Rooms 5–6) | **PyTorch DQN**, adapted from `code examples/dql/dqn.py` (`ReplayBuffer`/`PrioritizedReplayBuffer`, `QNetwork`, `DQNAgent`), retargeted to small `obs_dim`/`n_actions`. |
| UI language | **English.** |
| Cross-room progression | **Free navigation.** Every room is independently trainable any time; a lobby page shows a scoreboard of best saved results per room. No locking. |

Two smaller judgment calls, stated here so they're easy to correct later if wrong:

- **Room 6 "lookahead" sensing** is modeled as an **omnidirectional radius** of `L` meters
  around the agent (not a forward-facing cone), since the agent has no heading — only a
  velocity. "Ahead" = "within sensing range," reported as the `K` nearest obstacles.
- **Room 6 obstacles are static within an episode**; "dynamic" in the spec is read as
  *procedurally re-randomized between episodes* (count + position), which is also what's
  needed for the end-of-training "random room" generalization test. Moving-during-episode
  obstacles are listed as a stretch toggle (§6), not baseline behavior.

## 2. Cross-room concepts

### 2.1 Single exit per room
Every room has exactly one **success-terminal state** (the exit) that advances the narrative.
Rooms may *also* have failure-terminal states (a collision, a timeout) — those end the episode
but are not "the exit." This satisfies the spec's "single terminal state that allows leaving to
the next room" without forbidding failure states. Note: a "trap" isn't automatically one of
these — in Room 1, traps are a costly *non-terminal* hazard, not a failure state (see §4).

### 2.2 Scoring: G (one episode's return) vs. V (the trained value estimate)
No invented "Escape Score" — every room reports the same two standard RL quantities instead:
- **G** — the realized, discounted return of *one specific* rollout:
  `G = sum_t gamma^t * r_t`. Computed post-hoc from whatever rewards a single escape-attempt
  episode actually received (`engine/scoring.discounted_return(rewards, gamma)`). Because reward
  is discounted every step, reaching the goal sooner keeps more of a delayed goal reward, so a
  faster solve already shows up as a higher G — "faster = more reward" falls out naturally,
  no special-casing needed. Shown on the Board tab after an escape attempt; not persisted as a
  "best" anywhere, since it's one noisy sample, not a stable measure of the room.
- **V** — what training predicts the start state is worth: the learned value function's
  estimate, or its model-free analogue `max_a Q(s,a)` for rooms that only learn Q. Saved in
  `runs/<room_id>/history.json` (field `v_start`) every solve, and that's what the **lobby
  scoreboard** reads — V is stable and reproducible for a given config, unlike G. Shown on the
  Train tab as the headline "score of the training" — it doesn't depend on any one rollout, only
  on having trained the room. For Room 1 (DP) this is exact, straight from the Bellman equation;
  Rooms 2-4 (MC/SARSA/Q-learning) derive it from the learned Q-table; Rooms 5-6 (DQN) derive it
  from the network's own output at the start state.
- **Why both**: V is an average over all the randomness a stochastic environment can produce; G
  is one sample of it. A few escape attempts' G values clustering around V is itself a sanity
  check that the learned value function is accurate — comparing a sample to its own prediction is
  more informative (and more standard) than comparing a trained run to an arbitrary baseline.
  Neither number is fed back into training; both are purely for display.

### 2.3 Training run lifecycle
Every room's Train tab follows the same convention (copied from the LunarLander example):
**▶ Train / ⏹ Stop / 🔄 Reset** buttons, training runs on a background thread, a `queue.Queue`
drains into `st.session_state` on each rerun, and a metrics snapshot is taken every N
episodes/iterations (configurable) so charts don't redraw on every single step.

### 2.4 Checkpoints & replay
At every snapshot interval, the current policy/Q-table/network weights are saved as a
**checkpoint**, paired with a short recorded trajectory from a greedy rollout at that point in
training. The Board tab's replay control is a dropdown/slider over checkpoints — "what had the
agent learned by episode 50? By episode 500? By the end?" — using the same Plotly
frames+slider animation already built for the LunarLander replay.

### 2.5 Persistence
Training data is saved to disk (`runs/<room_id>/`), not just kept in `st.session_state`, so
closing and reopening the app doesn't lose progress — this is an explicit requirement
("the app must save and display learning-progress graphs"), not just a nice-to-have.

### 2.6 Lobby & free navigation
`app.py` is a lobby page built with `st.navigation`/`st.Page`: a short explanation of the game,
a table of all 6 rooms with their algorithm and trained V(start), and links into each room.
Rooms are always unlocked.

## 3. Architecture

### 3.1 Tech stack
- **Streamlit** — UI framework (multipage via `st.navigation`).
- **NumPy** — all tabular environments/agents (Rooms 1–4) and continuous physics (Rooms 5–6).
- **PyTorch (CPU)** — DQN for Rooms 5–6 only, installed the same lightweight way as the example
  (`--extra-index-url https://download.pytorch.org/whl/cpu`).
- **Plotly** — all charts and the grid/continuous renderers (heatmaps, shapes, animated frames).
- **Pandas** — small tables (recent-episodes view), optional.
- No `gymnasium`/`stable-baselines3` — environments are small, custom, and don't need a gym
  wrapper; keeping the dependency list short matches the simplicity requirement.

### 3.2 Repo layout
```
EscapeRoomRL/
├── app.py                          # lobby page + st.navigation registration
├── pages/
│   ├── 1_room1_dynamic_programming.py
│   ├── 2_room2_monte_carlo.py
│   ├── 3_room3_sarsa.py
│   ├── 4_room4_q_learning.py
│   ├── 5_room5_continuous_dqn.py
│   └── 6_room6_dynamic_obstacles.py
├── engine/
│   ├── grid_world.py               # GridWorld MDP — Rooms 1-4
│   ├── continuous_world.py         # ContinuousWorld physics — Rooms 5-6
│   ├── obstacles.py                # obstacle generation + collision — Room 6
│   ├── dp_solver.py                # value iteration / policy iteration
│   ├── monte_carlo_agent.py
│   ├── sarsa_agent.py
│   ├── q_learning_agent.py
│   ├── dqn_agent.py                 # adapted from code examples/dql/dqn.py
│   ├── training_runner.py          # shared thread + queue + snapshot runner
│   ├── scoring.py                   # G (discounted_return) — the V half lives in each room's solver
│   └── storage.py                   # save/load runs, checkpoints, best G per room
├── ui/
│   ├── sidebar_helpers.py
│   ├── charts.py
│   ├── grid_render.py               # Plotly heatmap + policy arrows
│   └── continuous_render.py         # Plotly shapes + trajectory animation
├── docs/
│   ├── room1.md … room6.md          # theory + parameter glossary — rendered verbatim by each Info tab
├── runs/                             # gitignored — regenerated by training, not source
├── tests/
│   ├── test_grid_world.py
│   ├── test_dp_solver.py
│   ├── test_continuous_world.py
│   └── test_obstacles.py
├── .streamlit/config.toml
├── requirements.txt
├── .gitignore
├── README.md
├── PROGRESS.md
└── plan.md
```

Each room's `docs/roomN.md` is the **single source of truth** for that room's explanation — the
Info tab just renders it with `st.markdown`. This avoids writing the same explanation twice (once
for the user, once in code) and directly satisfies the "progress documented in md files"
requirement as something that grows with the project rather than being written only at the end.

### 3.3 Shared engine modules (used by multiple rooms)
- **`grid_world.py` — `GridWorld`**: one configurable class for all four grid rooms (size,
  start, goal, walls, slippery cells + slip probability, traps, optional shortcut-tile
  teleports, step/goal/trap rewards).
  Gym-like API: `reset()`, `step(action) -> (state, reward, done, info)`. Also exposes
  `transition_model()` — `P(s'|s,a)` and `R(s,a,s')` — used **only** by Room 1's DP solver; Rooms
  2–4 only ever call `step()`, which is what makes the model "unknown" to them even though it's
  the same underlying class. One well-tested engine instead of four near-duplicates.
- **`patrol_world.py` — `PatrolGridWorld(GridWorld)`**: Room 4 only. Augments the state to
  `(agent_cell, patrol_phase)` and adds a deterministic moving enemy (failure-terminal on
  contact), leaving the base `GridWorld` — and Rooms 1-3 — unchanged. The phase indexes the
  enemy's position in its fixed patrol cycle, which is what restores the Markov property once a
  hazard moves.
- **`continuous_world.py` — `ContinuousWorld`**: 10×10 m room, position `(x, y)` continuous,
  velocity `(vx, vy) ∈ {-1, 0, 1}²` set directly by the chosen action (not acceleration), `dt =
  0.02 s`, boundary-clipped (no out-of-bounds failure). Room 6 subclasses it to add a static
  (per-episode) obstacle list and the partial-observation window.
- **`training_runner.py`**: generalizes the LunarLander `_train_loop`/`queue.Queue` pattern so
  each room's page only supplies a per-step callback; the thread lifecycle, stop flag, and
  snapshot cadence are shared code.
- **`scoring.py`**, **`storage.py`**: implement §2.2 and §2.5 once, used identically by all rooms.

## 4. Room specifications

Common page layout for every room (per the spec): **sidebar** (all algorithm + environment +
training-run parameters, Train/Stop/Reset) and three tabs — **Info** (renders `docs/roomN.md`),
**Train** (live metrics + Plotly learning curves + recent-episode table + V, the score of the
training), **Board** (rendered room, checkpoint replay slider, "▶ Run escape attempt" reporting
G next to V for comparison).

| # | Name | Algorithm | Model | State space | Action space |
|---|---|---|---|---|---|
| 1 | The Frozen Vault | Dynamic Programming (Value/Policy Iteration) | **Known** | 10×10 grid cell | 4 (U/D/L/R) |
| 2 | The Sampling Chamber | Monte Carlo control | Unknown | 10×10 grid cell | 4 (U/D/L/R) |
| 3 | The On-Policy Corridor | SARSA | Unknown | 10×10 grid cell | 4 (U/D/L/R) |
| 4 | The Off-Policy Cellar | Q-Learning | Unknown | 10×10 grid cell | 4 (U/D/L/R) |
| 5 | The Open Floor | DQN (function approximation) | Unknown | continuous `(x,y,vx,vy)` | 9 (vx,vy combos) |
| 6 | The Obstacle Gauntlet | DQN (function approximation) | Unknown | continuous + nearby obstacles | 9 (vx,vy combos) |

### Room 1 — The Frozen Vault (Dynamic Programming)
- **Task**: known-model 10×10 `GridWorld` on a **static, hand-designed board** (not procedurally
  regenerated): 13 wall cells scattered across 5 small clusters (not one corridor — generated via
  a seeded random-cluster scatter, then validated for reachability/connectivity and frozen),
  three traps, ten slippery cells (slip probability `p_slip`: intended action succeeds, otherwise
  a random action is substituted), one start cell, one high-reward goal cell. Moving into a wall
  or off the edge of the board is illegal — the agent simply doesn't move, and that action is
  never even offered to the policy (see Algorithm below). The board's *structure* is fixed; its
  *parameters* (slip probability, trap reward, γ) stay tunable from the sidebar — see SPRINTS.md
  Sprint 2 for the exact layout and reasoning.
- **Traps are not terminal.** Stepping on one costs `trap_reward` but the episode continues —
  a costly cell to pass through, not an instant failure. Only the goal ends the episode.
- **Reward model**: **no step reward.** `R(s,a,s')=0` for every transition except entering the
  goal (always) or a trap (a one-time cost each time it's entered). The goal reward is **fixed
  at a high value, not exposed in the sidebar** — since it's the only thing (with γ) shaping
  V(s) across the board, keeping it high keeps that shape visible instead of collapsing toward
  zero. With γ<1, V(s) still naturally encodes "closer is better" purely from discounting
  (`V(s) ≈ γ^d·goal_reward`), so there's no need for an artificial per-step cost to make the
  agent hurry — γ alone controls how fast V decays away from the goal, which is the point of
  the room.
- **Algorithm**: sidebar choice of **Value Iteration** or **Policy Iteration**, both directly
  implementing the Bellman equation over `transition_model()` — no environment interaction
  needed, "training" = Bellman sweeps to convergence (`max|ΔV| < θ`). Value Iteration applies the
  Bellman *optimality* equation (`V(s) ← max_a Σ P(s'|s,a)[R + γV(s')]`); Policy Iteration
  alternates the Bellman *expectation* equation (evaluate the current policy) with greedy
  improvement. `transition_model()` only offers actions that actually move the agent — an action
  that would bump a wall or the board edge is left out of the model entirely, so `max_a` can
  never select one; the learned policy can't deliberately walk into a wall. Slip is restricted
  the same way: it only ever substitutes another *legal* direction, never a wall bump.
- **Sidebar**: γ (0.5-1.0), θ, max iterations, DP method, slip probability, trap reward (no step
  reward, no goal reward, no grid-regeneration controls — the board and the goal payout are both
  fixed by design).
- **Train tab**: Δ V per iteration (log-scale), iterations-to-converge, V-heatmap snapshot per
  iteration (cheap to store all of them — DP converges in well under 100 sweeps).
- **Board tab**: grid heatmap (V) + policy arrows; slider over *iteration number* (not episodes)
  to watch the policy sharpen sweep by sweep; "Run escape attempt" does one stochastic rollout
  of the final greedy policy (slip still applies) from the start cell.

### Room 2 — The Sampling Chamber (Monte Carlo control)
- **Task**: same `GridWorld` engine, unknown-model framing (agent only calls `step()`).
  Slippery cells present, no traps by default.
- **Algorithm**: first-visit MC control, ε-greedy, with an **Exploring Starts** toggle (on =
  random start state each episode, matching `monte_carlo_es.py`; off = fixed start + ε-greedy
  exploration only, matching `monte_carlo_no_es.py`).
- **Sidebar**: episode count, γ, ε start/end + decay, exploring-starts toggle, max steps/episode
  (safety cap — MC needs full episodes, and a bad early policy can wander a long time on a
  slippery grid), grid regeneration.
- **Train tab**: reward & steps per episode, ε decay curve, per-state first-visit update-count
  heatmap (same idea as the `update_counts` table in `epsilon_greedy.py`).
- **Board tab**: same heatmap/arrows + checkpoint replay + escape attempt as Room 1.

### Room 3 — The On-Policy Corridor (SARSA)
- **Task**: same `GridWorld` engine, unknown-model framing (the agent only ever calls
  `step()`). Runs on its own **fixed, hand-designed 10×10 board** (`engine/boards.py`, same
  convention as Room 1 — see SPRINTS.md Sprint 3), not a procedurally regenerated one: walls,
  traps, and slippery cells laid out the same way Room 1's were, plus one new mechanic — a
  **shortcut tile** that instantly relocates the agent to a different cell when stepped on
  (whether by choice or by slip). This pulls the "shortcut tiles" idea out of the §6 stretch
  list into this room's core scope, since it's a cheap way to give an unknown-model room
  something genuinely new to discover through experience, not just a re-skinned Room 1. Reward
  model matches Room 1 exactly: no step cost, fixed high goal reward — discounting alone shapes
  V(s), same as every grid room going forward.
- **Algorithm**: on-policy TD(0) (SARSA), ε-greedy behavior policy, α learning rate. On-policy
  means the TD target bootstraps off the *actual next action the ε-greedy policy samples*
  (`Q(s,a) ← Q(s,a) + α[r + γQ(s',a') − Q(s,a)]`), not the best possible one — contrasted
  explicitly with Room 4's off-policy `max` update once that room's docs are written.
- **Sidebar**: slip probability, trap reward (board structure and goal reward are fixed, same
  as Room 1 — no grid-regeneration control), α (+ optional decay), γ, ε start/end + decay,
  episode count, max steps/episode.
- **Train tab**: reward & steps per episode, mean |TD-error| per episode, ε/α curves.
- **Board tab**: Q-derived policy/value heatmap + arrows (the shortcut tile drawn as its own
  marker with a line to its destination cell), checkpoint replay, escape attempt.

### Room 4 — The Off-Policy Cellar (Q-Learning)
- **Task**: same `GridWorld` family, unknown-model, on its own **fixed board** — but the hardest
  grid room, with two mechanics Rooms 1-3 didn't have: a **moving patrol enemy** and a **trap
  door**. Walls, non-terminal traps, and slippery cells carry over from Room 3 (slip stays 0.25).
- **Moving patrol enemy** (headline difficulty): a hazard that **moves one cell along a fixed
  ping-pong patrol every step**. Colliding with it is a **failure-terminal state** (`enemy_reward`,
  a large penalty) — the first room where you can *lose*, not just time out. Because the enemy
  moves, the agent's cell alone is no longer a Markov state (the same cell is safe or deadly
  depending on where the patroller is), so Room 4's state is augmented to
  **`(agent_cell, patrol_phase)`**, where `patrol_phase` indexes the enemy's deterministic
  position in its cycle. That expansion is itself the room's second lesson — *a dynamic hazard
  forces you to grow the state* — and it takes the tabular problem from ~85 states to ~85·P
  (still small; P = patrol period). Implemented as `PatrolGridWorld(GridWorld)` in
  `engine/patrol_world.py`, leaving Rooms 1-3 untouched.
- **Trap door** (from §6): a **bad teleport** — a cell that, when stepped on (by choice or via
  slip), relocates the agent *backward* toward the start. Reuses Room 3's `shortcuts` engine, but
  the destination sets you back instead of ahead; it looks like it sits on a fast route and has to
  be learned-around. Flagged as a trap door (via a `trap_door_sources` set on the grid) so the
  shared renderer draws it in a warning color, distinct from Room 3's teal helpful shortcut.
- **Reward model**: same as Rooms 1/3 — no step cost, fixed high goal reward, discounting shapes
  V(s) — plus the terminal `enemy_reward` on collision. Non-terminal traps still cost `trap_reward`.
- **Algorithm**: off-policy TD(0) Q-learning: `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') −
  Q(s,a)]`. The `max` over next actions (vs. Room 3's *sampled* `a'`) is the entire on/off-policy
  difference — and here it finally has a visible payoff: against the patrol, Q-learning learns the
  optimal, tightly-timed route that cuts close to the enemy, whereas an on-policy learner (Room 3's
  SARSA) would leave a safety buffer because its ε-greedy exploration risks stepping into the
  patrol. The Info tab lays the two update rules side by side and points at this behavior.
- **Sidebar**: Room 3's controls (slip, trap reward, α + decay, γ, ε schedule, episodes, max
  steps/episode, snapshot interval, max attempt steps) plus `enemy_reward` (terminal collision
  penalty). Board structure, goal reward, and the patrol path are fixed.
- **Train tab**: Room 3's metrics (reward/steps/mean |TD-error|/ε) plus an **outcome breakdown**
  per episode (escaped / caught / timed-out rate over training) — a cheap signal specific to this
  room's new failure mode.
- **Board tab**: Room 3's checkpoint-replay + escape-attempt, extended for the moving enemy — a
  **patrol-phase slider** picks which enemy position to view the value heatmap/policy for (V now
  depends on phase), and the escape replay **animates the enemy alongside the agent**, so you can
  watch it time its run past the patrol. V(start) = `max_a Q((start, phase 0), a)`.

### Room 5 — The Open Floor (continuous state, DQN)
- **Task**: `ContinuousWorld`, 10×10 m, start corner, single 1×1 m goal zone in the far corner.
  Position is continuous; velocity is one of 9 discrete combinations of `vx, vy ∈ {-1,0,1}`
  m/s, chosen directly as the action (action index → velocity mapping below). Step:
  `x += vx·dt; y += vy·dt` with `dt = 0.02 s`, position clipped to stay inside the room.
  Episode ends on reaching the goal zone (success) or exceeding `max_steps` (timeout).

  | action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|---|---|
  | (vx,vy) | (-1,-1) | (-1,0) | (-1,1) | (0,-1) | (0,0) | (0,1) | (1,-1) | (1,0) | (1,1) |

- **Reward**: small negative step cost + terminal goal bonus; an optional **distance-shaping**
  toggle (potential-based, same idea as the LunarLander example's shaping term) defaults ON,
  since a purely sparse reward over a ≤1000-step episode is a hard exploration problem for
  vanilla DQN — flagged in §10 as something to validate early and turn off once it's confirmed
  unnecessary.
- **Observation**: `[x/10, y/10, vx, vy]` (normalized position, raw velocity) → `obs_dim = 4`.
- **Algorithm**: `DQNAgent` adapted from `code examples/dql/dqn.py` almost unchanged
  (`obs_dim=4`, `n_actions=9`).
- **Sidebar**: DQN hyperparameters (lr, γ, ε schedule, hidden units, batch size, buffer size,
  target-update frequency, train-every-N-steps, optional PER) — directly mirrors the
  LunarLander sidebar's "DQN hyperparameters" expander; environment basics (max episode steps,
  goal zone position/size, shaping weight) under an "Advanced" expander since the room
  size/dt/velocity set are fixed by spec.
- **Train tab**: reward (raw + smoothed) / loss / ε, same 2×2 subplot layout as the LunarLander
  example.
- **Board tab**: animated `(x,y)` trajectory with a velocity arrow, start marker, goal zone,
  room boundary; checkpoint replay slider; "Run escape attempt" reports time-to-exit in seconds
  (`steps × 0.02`), G for that rollout, and V(start) (≈ `max_a Q(start, a)`) for comparison.

### Room 6 — The Obstacle Gauntlet (dynamic obstacles)
- **Task**: Room 5's `ContinuousWorld` + a list of circular obstacles (diameter 0.5 m, i.e.
  radius 0.25 m), count and positions **re-randomized every episode** within sidebar-configured
  ranges (min/max count, spawn region excluding a safe buffer around start and goal, minimum
  inter-obstacle spacing so layouts stay solvable). Collision with an obstacle is a new
  failure-terminal state (distinct from the timeout failure already in Room 5).
- **Observation/partial visibility** (the explicit "control over the agent's observation"
  requirement): base `[x/10, y/10, vx, vy]` plus the **K nearest obstacles within L meters**
  (both sidebar-configurable), each as a relative `(Δx, Δy)` from agent-center to
  obstacle-center, sorted by distance, padded with a fixed sentinel when fewer than K are in
  range → `obs_dim = 4 + 2K`. Sensing is modeled as an **omnidirectional radius** (see §1) since
  the agent has no heading.
- **Reward**: Room 5's step cost + goal bonus, plus a collision penalty on hitting an obstacle.
- **Algorithm**: same `DQNAgent` class as Room 5, just a larger `obs_dim`.
- **Sidebar**: everything from Room 5, plus obstacle controls (count range, lookahead `L`,
  `K`, obstacle width override, minimum spacing, layout random seed).
- **Train tab**: Room 5's metrics plus an outcome breakdown per episode (success / collision /
  timeout rate over time) — a cheap, useful signal specific to this room's added failure mode.
- **Board tab**: Room 5's renderer plus obstacles drawn as circles and a faint ring showing the
  current sensing radius `L`, so the partial-observability window is literally visible. Adds
  the spec's explicit end-of-training feature: a **"🎲 Generate random room & test"** button
  that procedurally builds a fresh, never-trained-on layout and runs the frozen learned policy
  on it with no further learning, reporting success/collision/timeout and G — the generalization
  test called out in the spec.

## 5. Info tab content (docs/roomN.md)

Each file covers, briefly: what the room's task is; which algorithm runs and the one-paragraph
intuition for it; what's known vs. unknown about the environment; the full parameter glossary
for that room's sidebar; and how that room derives V from what it learns. Written once, reused
by the in-app Info tab and as project documentation — no duplication.

## 6. Stretch / Phase-2 features (after all 6 rooms work end-to-end)

Optional, off-by-default toggles, available across the grid rooms (1–4) and noted as the spec's
"can add dynamic components" line — not required for the core deliverable. Note: Room 1 already
ships with traps and wall barriers as core, static-board features (see §4) — they're listed here
only in case Rooms 2-4 want *additional* or *randomized* versions of the same idea:
- **Bonus tiles**: a one-time pickup reward at a fixed or random cell.
- ~~Shortcut tiles~~ — pulled forward into Room 3's core scope instead of staying a stretch idea
  (see §4, Room 3): a tile that teleports the agent elsewhere, cheap to build and a good fit for
  a fixed board.
- ~~Simple patrol enemy~~ — pulled forward into Room 4's core scope (see §4, Room 4): a cell that
  ping-pongs on a fixed path, terminal-on-contact, with the state augmented to include its phase.
- ~~Bad-teleport "trap door"~~ — also pulled into Room 4 core (a shortcut whose destination sets
  the agent back toward the start).
- **Moving obstacles in Room 6**: upgrade the static-per-episode obstacles to drift during the
  episode, once the static version is trained and validated.

## 7. Documentation & git workflow

- **`README.md`**: project overview, how to run (`streamlit run app.py`), room list.
- **`docs/roomN.md`**: per-room theory + parameter glossary (§5), rendered by each Info tab.
- **`PROGRESS.md`**: updated at the end of every phase below — what shipped, what was decided,
  what's next. This is the running md-file progress record the spec asks for, kept current
  throughout rather than written once at the end.
- **Git**: this directory isn't a repo yet. Plan is to `git init` at the start of Phase 0, then
  one commit per phase (not one giant commit at the end), so the history itself documents
  progress room by room. Will confirm with you before running `git init`.

## 8. Testing

Small and targeted, matching the "simple, readable" code requirement — not exhaustive coverage:
- `test_grid_world.py`: transition correctness (walls block movement, slip probability fires at
  the expected rate, terminal states end episodes).
- `test_dp_solver.py`: value iteration converges to a known hand-computed answer on a tiny fixed
  grid (mirrors the existing `value_iteration.py` example, used as a regression check).
- `test_continuous_world.py`: position integrates correctly, boundary clipping works.
- `test_obstacles.py`: generated layouts respect count range, spacing, and the start/goal safe
  zone; observation padding is correct when fewer than K obstacles are in range.

## 9. Phased roadmap

| Phase | Scope | Git commit |
|---|---|---|
| 0 | Repo init, folder scaffold, shared engine bases (`GridWorld`, `ContinuousWorld` skeletons), `training_runner`, `storage`, `scoring`, UI helper skeletons, lobby `app.py`, `requirements.txt`, `.gitignore`, `README.md` | "Scaffold project structure and shared engine utilities" |
| 1 | Room 1 — DP | "Add Room 1: Dynamic Programming" |
| 2 | Room 2 — Monte Carlo | "Add Room 2: Monte Carlo control" |
| 3 | Room 3 — SARSA | "Add Room 3: SARSA" |
| 4 | Room 4 — Q-Learning | "Add Room 4: Q-Learning" |
| 5 | Room 5 — Continuous DQN | "Add Room 5: continuous state, DQN" |
| 6 | Room 6 — Dynamic obstacles | "Add Room 6: dynamic obstacles, partial observability" |
| 7 | Lobby scoreboard polish, docs pass, stretch toggles (§6) if wanted, final review | "Polish lobby, docs, stretch features" |

Each phase ends with a `PROGRESS.md` update before the commit.

This table is the high-level reference; the executable, checklist-level breakdown — with Phases
5 and 6 each split into two sprints (physics/observation logic validated before any training is
wired up) — lives in [SPRINTS.md](SPRINTS.md). Work from that file day to day; come back here
when a design decision needs revisiting.

## 10. Risks & things to validate early

- **Sparse reward in Rooms 5–6**: a ≤1000-step episode with only a terminal reward is a hard
  exploration problem for vanilla DQN. Mitigation already in the plan (distance shaping,
  default ON) — validate during Phase 5 that it actually converges in a reasonable wall-clock
  time for a live demo; drop shaping only once confirmed unnecessary.
- **Background-thread training inside Streamlit**: proven pattern already in
  `code examples/dql/streamlit_app.py` (daemon thread + `queue.Queue` + stop flag) — low risk,
  just reuse it via `training_runner.py` rather than re-deriving it per room.
- **Tabular rooms are fast**: 100 states × 4 actions converges quickly for Rooms 1–4, which is
  good (fast live demo) but means most of the "this takes a while" feel of increasing difficulty
  will come from Rooms 5–6, not 1–4 — intentional, matches the spec's progression.
- **Room 6 observation interpretation** (§1): omnidirectional sensing radius is an assumption,
  not explicit in the spec — flagged for you to correct if you meant a directional cone.

## 11. Requirements traceability

| Spec requirement | Where addressed |
|---|---|
| ≥4 rooms, increasing difficulty | §4 — 6 rooms, DP → MC → SARSA → Q-Learning → continuous DQN → dynamic obstacles |
| Rooms 1–3 on 10×10 grid | Rooms 1, 2, 3 (and 4) all use `GridWorld` at 10×10 |
| Room 1: known model → DP, slippery cells | Room 1 spec |
| Room 2 (spec's "room 2"): SARSA, unknown model, slippery cells | Room 3 here (MC inserted before it per §1) |
| Room 3 (spec's "room 3"): Q-Learning | Room 4 here |
| Room 4 (spec): continuous (x,y,vx,vy), 10×10 m, dt=0.02s, discrete velocity {-1,0,1} | Room 5 here |
| Room 5 (spec): dynamic obstacles, 0.5 m width, observation lookahead control, random-room generalization test | Room 6 here |
| Monte Carlo somewhere simple | Room 2 here |
| Control all algorithm/training parameters per room | Every room's sidebar (§4) |
| Save & display learning-progress graphs | §2.5 persistence + every room's Train tab |
| Replay individual episodes post-training | §2.4 checkpoint replay |
| Faster escape ⇒ higher reward | §2.2 |
| Simple, readable code | Shared engine modules (§3.3) instead of per-room duplication |
| Git documentation | §7 |
| Progress documented in md files | `PROGRESS.md` + `docs/roomN.md` (§7) |
