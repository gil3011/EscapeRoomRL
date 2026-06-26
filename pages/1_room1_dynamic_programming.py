"""Room 1 — Dynamic Programming. Known model, solved directly via the Bellman equation."""
from pathlib import Path

import streamlit as st

from engine.boards import ROOM1_GOAL_REWARD, make_room1_grid
from engine.dp_solver import (
    expected_steps_to_absorption, greedy_policy_dist, policy_iteration, uniform_policy_dist,
    value_iteration,
)
from engine.grid_world import run_episode
from engine.scoring import escape_score
from engine.storage import load_checkpoint, load_history, save_best_if_higher, save_checkpoint, save_history
from ui.charts import line_chart
from ui.grid_render import render_grid

ROOM_ID = "room1"

st.title("🧮 Room 1 — The Frozen Vault")

defaults = {
    "history": None, "par_steps": None, "optimal_steps": None,
    "attempt_path": None, "attempt_steps": None, "attempt_success": None, "attempt_score": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(f"{ROOM_ID}_{key}", value)

# reload the last run from disk once per fresh session
if st.session_state[f"{ROOM_ID}_history"] is None:
    saved_history = load_checkpoint(ROOM_ID, "snapshots")
    saved_meta = load_history(ROOM_ID)
    if saved_history is not None and saved_meta is not None:
        st.session_state[f"{ROOM_ID}_history"] = saved_history
        st.session_state[f"{ROOM_ID}_par_steps"] = saved_meta["par_steps"]
        st.session_state[f"{ROOM_ID}_optimal_steps"] = saved_meta["optimal_steps"]

with st.sidebar:
    st.header("Room 1 controls")

    with st.expander("Environment", expanded=True):
        slip_prob = st.slider("Slip probability", 0.0, 0.9, 0.2, 0.05)
        st.caption(f"Goal reward: **{ROOM1_GOAL_REWARD:.0f}** (fixed — kept high so V(s) "
                   "stays visible after discounting; not adjustable).")
        trap_reward = st.slider("Trap reward", -100.0, -1.0, -20.0, 1.0,
                                 help="Stepping on a trap costs this much, but doesn't end the episode.")

    with st.expander("DP algorithm", expanded=True):
        method = st.radio("Method", ["Value Iteration", "Policy Iteration"])
        gamma = st.slider("Gamma (γ)", 0.5, 1.0, 0.9, 0.01)
        theta = st.select_slider("Theta (θ)", [1e-2, 1e-3, 1e-4, 1e-5], value=1e-3)
        max_iterations = st.slider("Max iterations", 10, 500, 200, 10)

    with st.expander("Escape attempt", expanded=False):
        max_attempt_steps = st.slider("Max attempt steps", 20, 300, 150, 10)

    col1, col2 = st.columns(2)
    solve_clicked = col1.button("▶ Solve", use_container_width=True)
    reset_clicked = col2.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    for key, value in defaults.items():
        st.session_state[f"{ROOM_ID}_{key}"] = value
    st.rerun()

if solve_clicked:
    grid = make_room1_grid(slip_prob=slip_prob, trap_reward=trap_reward)
    model = grid.transition_model()
    states = grid.all_states()

    solver = value_iteration if method == "Value Iteration" else policy_iteration
    history = solver(model, states, gamma, theta, max_iterations)
    final_policy = history[-1]["policy"]

    par_T = expected_steps_to_absorption(model, states, uniform_policy_dist)
    opt_T = expected_steps_to_absorption(model, states, greedy_policy_dist(final_policy))
    par_steps = par_T[grid.start]
    optimal_steps = opt_T[grid.start]

    st.session_state[f"{ROOM_ID}_history"] = history
    st.session_state[f"{ROOM_ID}_par_steps"] = par_steps
    st.session_state[f"{ROOM_ID}_optimal_steps"] = optimal_steps
    st.session_state[f"{ROOM_ID}_attempt_path"] = None

    save_checkpoint(ROOM_ID, "snapshots", history)
    save_history(ROOM_ID, {
        "deltas": [h["delta"] for h in history],
        "policy_changes": [h["policy_changes"] for h in history],
        "par_steps": par_steps,
        "optimal_steps": optimal_steps,
        "method": method,
        "gamma": gamma,
    })

tab_info, tab_train, tab_board = st.tabs(["ℹ️ Info", "📈 Train", "🗺️ Board"])

with tab_info:
    st.markdown(Path("docs/room1.md").read_text(encoding="utf-8"))

with tab_train:
    history = st.session_state[f"{ROOM_ID}_history"]
    if history is None:
        st.info("Configure parameters in the sidebar, then press ▶ Solve.")
    else:
        deltas = [h["delta"] for h in history]
        policy_changes = [h["policy_changes"] for h in history]
        par_steps = st.session_state[f"{ROOM_ID}_par_steps"]
        optimal_steps = st.session_state[f"{ROOM_ID}_optimal_steps"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Iterations", len(history))
        c2.metric("Final max|ΔV|", f"{deltas[-1]:.2e}")
        c3.metric("Par (random) steps", f"{par_steps:.1f}")
        c4.metric("Optimal steps", f"{optimal_steps:.1f}")
        speedup = f"{par_steps / optimal_steps:.1f}x" if optimal_steps > 0 else "—"
        c5.metric("Speedup", speedup)

        st.plotly_chart(line_chart({"max|ΔV|": deltas}, title="Convergence", log_y=True),
                         use_container_width=True)
        st.plotly_chart(line_chart({"policy changes": policy_changes}, title="Policy stability"),
                         use_container_width=True)

        st.subheader("Recent iterations")
        n = min(20, len(history))
        st.dataframe({
            "Iteration": list(range(len(history) - n + 1, len(history) + 1)),
            "max|ΔV|": [f"{d:.2e}" for d in deltas[-n:]],
            "Policy changes": policy_changes[-n:],
        }, use_container_width=True, hide_index=True)

with tab_board:
    history = st.session_state[f"{ROOM_ID}_history"]
    if history is None:
        st.info("Solve the room first to see the board.")
    else:
        grid = make_room1_grid(slip_prob=slip_prob, trap_reward=trap_reward)
        final_policy = history[-1]["policy"]

        view = st.radio("View", ["📊 Iteration snapshot", "🏁 Escape attempt"], horizontal=True)

        if view == "📊 Iteration snapshot":
            idx = st.slider("Iteration", 0, len(history) - 1, len(history) - 1)
            snap = history[idx]
            fig = render_grid(grid, values=snap["V"], policy=snap["policy"],
                               title=f"Iteration {idx + 1}/{len(history)}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            if st.button("🏁 Run escape attempt"):
                path, steps, success = run_episode(grid, final_policy, max_attempt_steps)
                par_steps = st.session_state[f"{ROOM_ID}_par_steps"]
                score = escape_score(steps, par_steps, success)
                st.session_state[f"{ROOM_ID}_attempt_path"] = path
                st.session_state[f"{ROOM_ID}_attempt_steps"] = steps
                st.session_state[f"{ROOM_ID}_attempt_success"] = success
                st.session_state[f"{ROOM_ID}_attempt_score"] = score
                save_best_if_higher(ROOM_ID, score, steps, success)

            path = st.session_state[f"{ROOM_ID}_attempt_path"]
            if path is None:
                st.caption("Click 'Run escape attempt' to roll out the solved policy once, with slip applied.")
            else:
                step_idx = st.slider("Step", 0, len(path) - 1, len(path) - 1)
                fig = render_grid(grid, values=history[-1]["V"], policy=final_policy,
                                   agent_pos=path[step_idx], title="Escape attempt")
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Steps taken", st.session_state[f"{ROOM_ID}_attempt_steps"])
                result = "✅ Escaped" if st.session_state[f"{ROOM_ID}_attempt_success"] else "❌ Failed"
                c2.metric("Result", result)
                c3.metric("Escape Score", st.session_state[f"{ROOM_ID}_attempt_score"])
