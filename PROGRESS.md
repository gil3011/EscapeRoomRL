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
- Next: **Sprint 2** — Room 1 (Dynamic Programming).
