# Progress

## Sprint 1 — Scaffolding ✅
- Repo initialized; folder structure matches plan.md §3.2.
- Shared engine: `GridWorld` (+ `random_layout` generator), `TrainingRunner`
  (background-thread + queue pattern generalized from the `code examples/dql` reference
  app), `storage` (history/checkpoints/best-score persistence under `runs/<room_id>/`),
  `scoring` (Escape Score formula).
- Shared UI helpers: `grid_render.render_grid`, `charts.line_chart`, `sidebar_helpers.train_stop_reset`.
- `app.py` lobby lists all 6 rooms with a scoreboard (reads `best.json` per room — empty
  until rooms are actually trained). Each room page is a stub until its sprint lands.
- `tests/test_grid_world.py` passing.

## Sprint 2 — Room 1: Dynamic Programming ✅
- `engine/boards.py` — Room 1's static board: 13 walls in 5 scattered clusters, 3 traps,
  10 slippery cells, validated (goal reachable, no isolated pockets) and frozen as constants.
- `engine/dp_solver.py` — `value_iteration` and `policy_iteration` (Bellman optimality and
  expectation equations, in-place sweeps adapted from `code examples/value_iteration.py` and
  `policy_iteration_deterministic.py`), plus `expected_steps_to_absorption` for the scoring
  baseline. No `step_reward` — V(s) is shaped entirely by discounting the terminal rewards.
- `engine/grid_world.py` — added `run_episode()`, a shared deterministic-policy rollout used
  by escape attempts (will be reused by Rooms 2-4).
- `pages/1_room1_dynamic_programming.py` — full sidebar (environment + DP + escape-attempt
  params), Info/Train/Board tabs. DP converges in milliseconds so this room runs synchronously
  on the ▶ Solve click instead of using `TrainingRunner`'s background thread.
- `docs/room1.md` — theory, the Bellman equation, parameter glossary, scoring explanation.
- Verified end-to-end via `pytest` (19/19 passing, including a hand-checked Bellman-equation
  result) and Streamlit's `AppTest` (solve → view toggle → escape attempt → reset, zero
  exceptions). Sanity numbers on the default config: converges in 23 iterations; optimal policy
  averages ~18.2 steps to the goal across 200 stochastic trials (slip_prob=0.2) with a 100%
  success rate, against a ~151-step random-walk baseline — Escape Score ≈ 880.
- **Fix**: `st.button`/`st.plotly_chart`/`st.dataframe` calls used the newer `width=` parameter,
  which doesn't exist on Streamlit 1.43 — only on this project's pinned 1.58 (`.venv`). A second
  Streamlit install on `PATH` (a different Python on this machine) hit that gap and crashed on
  Room 1's first button. Reverted to `use_container_width=True`, which works on both; verified
  against both Python installs. README now calls out always using `.venv\Scripts\streamlit`.
- **Redesign (post-review feedback)**:
  - Traps are now **non-terminal** — stepping on one costs `trap_reward` but the episode
    continues. `GridWorld.is_terminal()` only checks the goal now; `_reward_for` returns
    `done=False` for traps. Knock-on effect: `transition_model()` now includes entries for trap
    cells too (they're walkable), and `ui/grid_render.py` now draws policy arrows on them.
  - `transition_model()` now **excludes any action that bumps a wall or the board edge**
    (`_move(s,a) == s`) from the model entirely, so `value_iteration`/`policy_iteration`'s
    `max_a` can never select one — the learned policy can't deliberately walk into a wall.
    Slip can still accidentally land on a bump as a side effect of a legal action; only the
    *chosen* action is restricted. Required reworking `expected_steps_to_absorption`'s
    `policy_dist` callback to `(state, actions) -> dist` so the random baseline correctly
    normalizes over however many actions are actually legal at a given state, not always 4.
  - `goal_reward` is now a **fixed constant** (`ROOM1_GOAL_REWARD = 100.0` in
    `engine/boards.py`), removed from `make_room1_grid()`'s signature entirely and from the
    sidebar (shown read-only instead).
  - Net effect on the default config: par_steps (random baseline) jumped from ~151 to ~524
    steps — since a random walk can no longer be cut short by a trap, its only way to finish is
    to stumble onto the goal, which takes much longer undirected. Optimal steps is unchanged
    (~18); Escape Score for a good run is now closer to 950-960.
  - Added 5 tests covering the new behavior (non-terminal trap, wall-bump exclusion, every open
    cell has a legal action); full suite is 22/22 passing.
- **Scoring redesign**: replaced the invented "Escape Score" (par_steps vs. a random-walk
  baseline) with the two standard RL quantities instead, per feedback — `expected_steps_to_absorption`,
  `uniform_policy_dist`, and `greedy_policy_dist` are deleted (no remaining callers).
  - **V(start)** — `history[-1]["V"][grid.start]`, already sitting in the solve output. Shown on
    the Train tab as the headline "score of the training."
  - **G** — the realized discounted return of one escape-attempt rollout
    (`engine.scoring.discounted_return(rewards, gamma)`). Required widening
    `run_episode()`'s return to `(path, rewards, steps_taken, success)` so there's something to
    discount, using the *same* gamma the room was solved with (stashed at solve time, not
    re-read from a possibly-since-moved slider).
  - `runs/<room_id>/best.json`'s field is now `G` (was `score`); lobby shows "Best G".
  - Verified via `pytest` (25/25, including a new `tests/test_scoring.py`) and `AppTest`: on the
    default config, V(start) ≈ 16.04 and one escape attempt's G ≈ 16.68 — close, as expected,
    since G is one stochastic sample of what V predicts on average.
- **Board colors + lobby metric (post-review feedback)**:
  - `ui/grid_render.py`: the heatmap used `Blues`, the same hue as the slippery-cell overlay —
    swapped to `Oranges` so the continuous V signal never competes with a categorical marker.
    Also found and fixed a real bug while at it: the goal cell's stored V is always 0 (terminal
    states are never updated by the solver), so it was rendering as the *lowest*-value cell on
    the heatmap — it now gets its own solid green marker instead, like walls already had. Start
    is now purple (was green, clashing with the new goal marker) and the replayed agent position
    is rendered as a top-layer annotation rather than a separate trace, so it can never be
    visually hidden under a wall/goal/trap overlay mid-replay.
  - Lobby now shows **V(start)** per room instead of "Best G" — V is stable for a given config,
    G is one noisy rollout sample, so V is the one that belongs on a leaderboard. Since nothing
    else ever read `best.json`, `save_best_if_higher`/`load_best` are now dead code — removed
    (G is still shown on Room 1's own Board tab, just not tracked as a "best").
  - Verified via `pytest` (25/25, unaffected) and `AppTest` (lobby + full Room 1 flow, zero
    exceptions); visual color check left to you on the already-running app, since you already
    had it open.
- **Slip never bumps a wall (post-review feedback)**: factored out a new
  `GridWorld._legal_actions(state)` helper (shared by `step()`, `transition_model()`, and
  `_outcomes()`) so slipping on an icy cell only ever substitutes another *legal* direction —
  never one that would just bump a wall or the board edge. Previously the substituted direction
  was drawn from all 3 non-intended compass directions regardless of legality, occasionally
  "wasting" a slip on a wall it was standing right next to. Two of Room 1's ten slippery cells
  (`(1,1)`, `(5,8)`) are directly wall-adjacent, so this is a real, measurable effect on this
  board: V(start) shifted slightly (≈16.04 → ≈15.92 at default settings) once that probability
  mass got redistributed to the remaining legal directions. If a cell's only legal action is the
  intended one, slip simply has nowhere else to go that turn. Added 3 tests
  (`test_slip_never_bumps_a_wall_when_a_legal_alternative_exists`,
  `test_slip_with_no_legal_alternative_keeps_the_intended_action`,
  `test_transition_model_slip_outcomes_exclude_wall_bumps`); full suite is 28/28 passing.
- Sprints re-ordered: Room 2 (Monte Carlo) is the one non-mandatory room (plan.md §1) and has
  been moved to Sprint 9, after the mandatory arc (Rooms 1, 3, 4, 5, 6) — see SPRINTS.md's
  reordering note.

## Sprint 3 — Room 3: SARSA ✅
- **Design decisions (locked with the user)**: Room 3 gets a fixed, hand-designed board like
  Room 1 (not procedural); the same reward model as Room 1 (no step cost, fixed high goal
  reward); the same hazard mix (walls/traps/slippery) **plus a new shortcut tile** — pulled
  forward from plan.md §6's stretch list into core scope.
- `engine/grid_world.py` — added an optional `shortcuts: dict[cell, cell]` field. Landing on a
  shortcut source (by choice **or** via slip) teleports the agent to its destination; handled in
  `_move()`, so `step()`, `transition_model()`, `_outcomes()`, and slip's legal-alternative draw
  all pick it up for free. Default empty, so Room 1 is unaffected. Also exposed a public
  `legal_actions()` accessor (thin wrapper over `_legal_actions`) for the model-free rooms.
- `engine/boards.py` — Room 3's static board: 13 walls in scattered clusters (different seed
  from Room 1), 3 non-terminal traps, 10 slippery cells, and a `(5,1)→(5,8)` shortcut. Validated
  (no overlaps, goal reachable, no isolated pockets, destination is a plain cell); the shortcut
  cuts the fastest solve from 18 steps to 11. Default `slip_prob` nudged to 0.25 (difficulty step
  after Room 1). Goal reward fixed at 100, matching Room 1 so V(start) stays comparable.
- `engine/sarsa_agent.py` — on-policy TD(0): Q-table, ε-greedy behaviour policy that bootstraps
  off the *sampled* next action `a'` (not the greedy max — the line that will differ from Room 4).
  ε (and optional α) linear decay, per-episode metrics (reward/steps/mean |TD-error|/ε/α),
  checkpoints every N episodes (policy + Q-derived values + a greedy rollout). With no step cost,
  the behaviour policy is restricted to legal (position-changing) moves — documented as a modeling
  choice that keeps the greedy policy from ever showing a wall-bump arrow, same as Room 1.
- `pages/3_room3_sarsa.py` — unlike Room 1's instant DP solve, this room trains on the shared
  `TrainingRunner` background thread and streams metrics over its queue; the page drains the queue
  each rerun and self-reruns (~0.4 s) while training, then persists on the final result. Controls
  grouped env/SARSA/exploration; Info/Train/Board tabs mirror Room 1 (checkpoint-replay slider +
  escape attempt reporting G next to V). Reloads the last run from disk on a cold start.
- `ui/grid_render.py` — draws shortcut tiles in teal (🌀 source → ◎ destination) with a dashed
  connecting line; skips a policy arrow on the source cell (the agent never rests there).
- `docs/room3.md` — known-vs-unknown framing vs. Room 1, the SARSA update rule with "on-policy"
  explained, the shortcut mechanic, the V-vs-G scoring convention, full parameter glossary.
- **Verified**: `pytest` 41/41 (added shortcut, Room 3 board, and SARSA-agent tests); an AppTest
  smoke run trained → viewed Train/Board tabs → ran an escape attempt with zero exceptions and a
  clean `history.json` persistence round-trip. At the default config (3000 episodes, slip 0.25)
  the greedy policy escapes on 100% of 300 stochastic rollouts, averaging 11 steps (the
  shortcut-optimal path), using the shortcut 100% of the time, with V(start) ≈ 32 (≈ γ¹¹·100).
- Next: **Sprint 4** — Room 4 (Q-Learning); Info tab contrasts off-policy `max` update vs. Room 3's
  on-policy SARSA, reusing this room's page/agent structure.
