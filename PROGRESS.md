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
- Next: **Sprint 3** — Room 2 (Monte Carlo).
