"""Room 1 — Dynamic Programming. Known model, solved directly via the Bellman equation."""
from pathlib import Path

import streamlit as st

from engine.boards import ROOM1_GOAL_REWARD, make_room1_grid
from engine.dp_solver import policy_iteration, value_iteration
from engine.grid_world import run_episode
from engine.scoring import discounted_return
from engine.storage import save_checkpoint, save_history
from ui.charts import line_chart
from ui.grid_render import render_grid

ROOM_ID = "room1"

st.title("🧮 Room 1 — The Frozen Vault")

defaults = {
    "history": None, "v_start": None, "gamma": None,
    "attempt_path": None, "attempt_steps": None, "attempt_success": None, "attempt_g": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(f"{ROOM_ID}_{key}", value)

st.header("Room 1 controls")

theta = 1e-3  # fixed — not user-adjustable
max_iterations = 150  # fixed — not user-adjustable

env_col, dp_col, attempt_col = st.columns(3)
with env_col:
    slip_prob = st.slider("Slip probability", 0.0, 0.9, 0.2, 0.05)
    trap_reward = st.slider("Trap reward", -100.0, -1.0, -20.0, 1.0)

with dp_col:
    method = st.radio("Method", ["Value Iteration", "Policy Iteration"])
    gamma = st.slider("Gamma (γ)", 0.5, 1.0, 0.9, 0.01)

with attempt_col:
    max_attempt_steps = st.slider("Max attempt steps", 20, 300, 150, 10)

solve_clicked = st.button("▶ Solve", use_container_width=True)

st.divider()

if solve_clicked:
    grid = make_room1_grid(slip_prob=slip_prob, trap_reward=trap_reward)
    model = grid.transition_model()
    states = grid.all_states()

    solver = value_iteration if method == "Value Iteration" else policy_iteration
    history = solver(model, states, gamma, theta, max_iterations)
    v_start = history[-1]["V"][grid.start]

    st.session_state[f"{ROOM_ID}_history"] = history
    st.session_state[f"{ROOM_ID}_v_start"] = v_start
    st.session_state[f"{ROOM_ID}_gamma"] = gamma
    st.session_state[f"{ROOM_ID}_attempt_path"] = None

    save_checkpoint(ROOM_ID, "snapshots", history)
    save_history(ROOM_ID, {
        "deltas": [h["delta"] for h in history],
        "policy_changes": [h["policy_changes"] for h in history],
        "v_start": v_start,
        "gamma": gamma,
        "method": method,
    })

tab_info, tab_train, tab_board = st.tabs(["ℹ️ Info", "📈 Train Result", "🏁 Run episode"])

with tab_info:
    preview_grid = make_room1_grid(slip_prob=slip_prob, trap_reward=trap_reward)
    st.plotly_chart(render_grid(preview_grid, title="Room 1 layout"), use_container_width=True)

    st.markdown(Path("docs/room1.md").read_text(encoding="utf-8"))

with tab_train:
    history = st.session_state[f"{ROOM_ID}_history"]
    if history is None:
        st.info("Configure parameters above, then press ▶ Solve.")
    else:
        grid = make_room1_grid(slip_prob=slip_prob, trap_reward=trap_reward)
        deltas = [h["delta"] for h in history]
        policy_changes = [h["policy_changes"] for h in history]
        v_start = st.session_state[f"{ROOM_ID}_v_start"]

        st.subheader("Iteration snapshot")
        idx = st.slider("Iteration", 0, len(history) - 1, len(history) - 1)
        snap = history[idx]
        fig = render_grid(grid, values=snap["V"], policy=snap["policy"],
                           title=f"Iteration {idx + 1}/{len(history)}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The value function and greedy policy as they stood after this many "
                   "sweeps — drag the slider to watch both settle toward their final shape.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Iterations", len(history))
        c2.metric("Final max|ΔV|", f"{deltas[-1]:.2e}")
        c3.metric("V(start)", f"{v_start:.2f}",
                  help="The trained value function's estimate at the start cell — the "
                       "expected discounted return under the optimal policy. This is "
                       "the score of the training itself.")

        st.plotly_chart(line_chart({"max|ΔV|": deltas}, title="Convergence", log_y=True),
                         use_container_width=True)
        st.caption("The biggest change to any cell's value on each sweep — trending down "
                   "toward zero means V(s) has converged and stopped updating.")

        st.plotly_chart(line_chart({"policy changes": policy_changes}, title="Policy stability"),
                         use_container_width=True)
        st.caption("How many states switched to a different preferred action on each sweep "
                   "— reaching zero means the policy itself has stopped changing.")

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
        st.info("Solve the room first to run an episode.")
    else:
        grid = make_room1_grid(slip_prob=slip_prob, trap_reward=trap_reward)
        final_policy = history[-1]["policy"]

        if st.button("🏁 Run escape attempt"):
            path, rewards, steps, success = run_episode(grid, final_policy, max_attempt_steps)
            solved_gamma = st.session_state[f"{ROOM_ID}_gamma"]
            G = discounted_return(rewards, solved_gamma)
            st.session_state[f"{ROOM_ID}_attempt_path"] = path
            st.session_state[f"{ROOM_ID}_attempt_steps"] = steps
            st.session_state[f"{ROOM_ID}_attempt_success"] = success
            st.session_state[f"{ROOM_ID}_attempt_g"] = G

        path = st.session_state[f"{ROOM_ID}_attempt_path"]
        if path is None:
            st.caption("Click 'Run escape attempt' to roll out the solved policy once, with slip applied.")
        else:
            step_idx = st.slider("Step", 0, len(path) - 1, 0)
            fig = render_grid(grid, values=history[-1]["V"], policy=final_policy,
                               agent_pos=path[step_idx], path=path, title="Escape attempt")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Steps taken", st.session_state[f"{ROOM_ID}_attempt_steps"])
            result = "✅ Escaped" if st.session_state[f"{ROOM_ID}_attempt_success"] else "❌ Failed"
            c2.metric("Result", result)
            c3.metric("G (this episode)", f"{st.session_state[f'{ROOM_ID}_attempt_g']:.2f}",
                      help="The actual discounted return collected during this one rollout.")
            c4.metric("V(start)", f"{st.session_state[f'{ROOM_ID}_v_start']:.2f}",
                      help="What training predicted this state was worth — compare against G.")
