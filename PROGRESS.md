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
- Next: **Sprint 3** — Room 2 (Monte Carlo).
