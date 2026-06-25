# EscapeRoomRL — Sprint Plan

Deliverable-based sprints, no calendar dates — move to the next sprint when the current one's
Definition of Done is met, whatever that takes. Each sprint is one mergeable increment and one
git commit (per [plan.md](plan.md) §7). Update `PROGRESS.md` at the end of every sprint.

Mapping back to [plan.md](plan.md) §9's phases (Room 5 and Room 6 are each split in two here,
since their physics/observation logic is worth validating before any training is wired up):

| Sprint | plan.md phase | Scope |
|---|---|---|
| 1 | Phase 0 | Scaffolding |
| 2 | Phase 1 | Room 1 — Dynamic Programming |
| 3 | Phase 2 | Room 2 — Monte Carlo |
| 4 | Phase 3 | Room 3 — SARSA |
| 5 | Phase 4 | Room 4 — Q-Learning |
| 6 | Phase 5a | Room 5 — continuous physics engine |
| 7 | Phase 5b | Room 5 — DQN integration & UI |
| 8 | Phase 6a | Room 6 — obstacles & partial observation |
| 9 | Phase 6b | Room 6 — training, generalization test & polish |
| 10 | Phase 7 | Final polish |

---

## Sprint 1 — Scaffolding
**Goal**: stand up the shared engine + UI skeleton so every later sprint is "add a room," not
"rebuild infrastructure."
**Depends on**: nothing.

Backlog:
- [ ] `git init`, `.gitignore`, `requirements.txt` (streamlit, numpy, plotly, pandas — torch added in Sprint 6)
- [ ] Folder structure per plan.md §3.2
- [ ] `engine/grid_world.py` — `GridWorld` mechanics (movement, walls, slip, terminal states, `transition_model()`); rooms 1-4 all need this immediately
- [ ] `engine/training_runner.py` — shared thread + queue + snapshot runner
- [ ] `engine/storage.py` — save/load runs, checkpoints, best score to `runs/<room_id>/`
- [ ] `engine/scoring.py` — Escape Score formula
- [ ] `ui/sidebar_helpers.py`, `ui/charts.py`, `ui/grid_render.py` skeletons
- [ ] `app.py` — lobby page, `st.navigation` registering all 6 rooms (placeholders for now)
- [ ] `README.md` skeleton, `PROGRESS.md` started
- [ ] `tests/test_grid_world.py`

**Definition of Done**: `streamlit run app.py` launches and shows the lobby listing all 6 rooms
(placeholders OK); `GridWorld` unit tests pass; first git commit made.
**Out of scope**: any room's UI/algorithm; `ContinuousWorld` (Sprint 6).

---

## Sprint 2 — Room 1: Dynamic Programming
**Goal**: first fully working room end-to-end — validates the sidebar/Info/Train/Board pattern
every later room will reuse.
**Depends on**: Sprint 1.

Backlog:
- [ ] `engine/dp_solver.py` — value iteration + policy iteration over `GridWorld.transition_model()`
- [ ] `pages/1_room1_dynamic_programming.py` — sidebar (γ, θ, max iterations, DP method, slip probability, rewards, grid regeneration) + Info/Train/Board tabs
- [ ] `docs/room1.md`
- [ ] Train tab: ΔV per iteration chart, per-iteration V snapshots
- [ ] Board tab: V-heatmap + policy arrows, iteration slider, "Run escape attempt"
- [ ] `runs/room1/` persistence wired through `storage.py`

**Definition of Done**: every Room 1 parameter is tunable from the sidebar; running to
convergence shows the ΔV curve; any iteration's policy can be replayed on the board; an escape
attempt reports an Escape Score; reopening the app reloads Room 1's last run from disk;
`test_dp_solver.py` passes against a hand-checked tiny grid.
**Out of scope**: Rooms 2+; any GridWorld feature only needed later (traps/bonus toggles → §6 stretch in plan.md).

---

## Sprint 3 — Room 2: Monte Carlo
**Goal**: first model-free room; reuses Sprint 2's page pattern almost unchanged.
**Depends on**: Sprint 2.

Backlog:
- [ ] `engine/monte_carlo_agent.py` — first-visit MC control, ε-greedy, exploring-starts toggle
- [ ] `pages/2_room2_monte_carlo.py`
- [ ] `docs/room2.md`
- [ ] Train tab: reward/steps per episode, ε decay, visit-count heatmap
- [ ] Board tab: policy/value heatmap, episode-checkpoint replay slider, escape attempt

**Definition of Done**: same shape as Sprint 2's DoD, for Room 2; toggling exploring-starts
visibly changes early-episode behavior.
**Out of scope**: Rooms 3+.

---

## Sprint 4 — Room 3: SARSA
**Goal**: first TD-learning room.
**Depends on**: Sprint 3.

Backlog:
- [ ] `engine/sarsa_agent.py`
- [ ] `pages/3_room3_sarsa.py`
- [ ] `docs/room3.md`
- [ ] Train tab: reward/steps per episode, mean |TD-error|, ε/α curves
- [ ] Board tab: same pattern as Rooms 1-2

**Definition of Done**: trains to a working exit policy on the default slippery grid; checkpoint
replay visibly improves from early to late episodes.
**Out of scope**: Room 4+.

---

## Sprint 5 — Room 4: Q-Learning
**Goal**: complete the grid-room arc; Info tab explicitly contrasts on-policy vs. off-policy
against Room 3.
**Depends on**: Sprint 4.

Backlog:
- [ ] `engine/q_learning_agent.py`
- [ ] `pages/4_room4_q_learning.py`
- [ ] `docs/room4.md` (incl. SARSA-vs-Q-learning comparison)
- [ ] Train/Board tabs: same pattern

**Definition of Done**: trains successfully; Info tab clearly explains the update-rule
difference from Room 3.
**Out of scope**: any plan.md §6 stretch toggle (enemy/trap/bonus/shortcut) — deferred to Sprint 10.

---

## Sprint 6 — Room 5a: continuous physics engine
**Goal**: get `ContinuousWorld` physics right and visually verified *before* touching DQN, so a
training failure later is unambiguously an algorithm problem, not a physics bug.
**Depends on**: Sprint 1 (independent of Rooms 1-4).

Backlog:
- [ ] `engine/continuous_world.py` — position/velocity integration, 9-action mapping, boundary clipping, goal-zone detection
- [ ] `ui/continuous_render.py` — room boundary, start, goal zone, agent position + velocity arrow
- [ ] `tests/test_continuous_world.py`
- [ ] Scratch harness driving a few fixed action sequences to confirm trajectories render correctly
- [ ] Add CPU-only `torch` to `requirements.txt` ahead of Sprint 7

**Definition of Done**: a fixed/manual action sequence produces a visually correct,
boundary-respecting trajectory in the Plotly renderer; unit tests pass.
**Out of scope**: DQN, reward shaping, the actual Room 5 page (Sprint 7).

---

## Sprint 7 — Room 5b: DQN integration & UI
**Goal**: wire the validated physics engine to a trained policy and the full room page.
**Depends on**: Sprint 6.

Backlog:
- [ ] `engine/dqn_agent.py` adapted from `code examples/dql/dqn.py` (`obs_dim=4`, `n_actions=9`)
- [ ] Reward function incl. distance-shaping toggle
- [ ] `pages/5_room5_continuous_dqn.py` — sidebar (DQN hyperparams + advanced env params), Train tab (reward/loss/ε subplot), Board tab (animated trajectory, checkpoint replay, escape attempt in seconds)
- [ ] `docs/room5.md`
- [ ] `runs/room5/` persistence (network weights via `torch.save`)

**Definition of Done**: training visibly converges to a policy that reaches the goal zone faster
than a random policy, in a reasonable wall-clock time for a live demo; checkpoint replay shows
early-vs-late trajectories; escape attempt reports time-to-exit.
**Out of scope**: obstacles (Room 6).

---

## Sprint 8 — Room 6a: obstacles & partial observation
**Goal**: get obstacle generation, collision, and the K-nearest observation window right and
visually verified before training.
**Depends on**: Sprint 7 (reuses `ContinuousWorld` + `DQNAgent`).

Backlog:
- [ ] `engine/obstacles.py` — procedural layout generation (count range, spacing, start/goal safe zones), circle-collision check, K-nearest-within-`L` + sentinel padding
- [ ] `ContinuousWorld` extension for Room 6 (`obs_dim = 4 + 2K`)
- [ ] `ui/continuous_render.py` — draw obstacles + sensing-radius ring
- [ ] `tests/test_obstacles.py`
- [ ] Scratch harness to generate a few layouts and manually verify rendering/collision/observation by eye

**Definition of Done**: generated layouts always respect count/spacing/safe-zone constraints;
observation vector has the right shape and correct padding when fewer than K obstacles are in
range; unit tests pass.
**Out of scope**: training/Room 6 page wiring, the random-room generalization button (Sprint 9).

---

## Sprint 9 — Room 6b: training, generalization test & polish
**Goal**: finish Room 6 end-to-end, including the spec's explicit post-training generalization test.
**Depends on**: Sprint 8.

Backlog:
- [ ] `pages/6_room6_dynamic_obstacles.py` — full sidebar/Info/Train/Board
- [ ] `docs/room6.md`
- [ ] Train tab: Room 5's metrics + success/collision/timeout outcome breakdown
- [ ] Board tab: "🎲 Generate random room & test" button (frozen policy, no learning, new layout)
- [ ] `runs/room6/` persistence

**Definition of Done**: trains to a policy that reaches the goal while avoiding obstacles
meaningfully more often than a random policy; the random-room button runs the frozen policy on
a fresh layout and reports outcome + Escape Score.
**Out of scope**: stretch features (moving obstacles, plan.md §6).

---

## Sprint 10 — Final polish
**Goal**: tie the six rooms together into the finished product.
**Depends on**: Sprints 2-9 all done.

Backlog:
- [ ] Lobby page: real scoreboard reading every room's `best.json`, replacing Sprint 1's placeholders
- [ ] Full docs pass (`README.md`, all `docs/roomN.md`, `PROGRESS.md`)
- [ ] Optional: any plan.md §6 stretch toggles still wanted (traps/bonus/shortcut/patrol-enemy on rooms 1-4, moving obstacles on room 6)
- [ ] `/code-review` pass before calling it done

**Definition of Done**: lobby shows accurate best-score-per-room for every trained room; every
Info tab reviewed for accuracy; no open TODOs in code.
**Out of scope**: anything not already in plan.md — new feature requests start a new sprint.
