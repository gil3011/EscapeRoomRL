"""Room 3 — SARSA. Model-free, on-policy TD(0): the agent learns Q(s,a) from the
transitions it experiences, never touching the transition model Room 1 solved directly.

Structure mirrors Room 1 (sidebar-style controls + Info/Train/Board tabs), but training
is genuinely episode-by-episode, so unlike Room 1's instant DP solve this room runs on
the shared TrainingRunner background thread and streams metrics back over a queue.
"""
from collections import deque
from pathlib import Path

import streamlit as st

from engine.boards import ROOM3_GOAL_REWARD, make_room3_grid
from engine.grid_world import run_episode
from engine.sarsa_agent import train_sarsa
from engine.scoring import discounted_return
from engine.storage import load_checkpoint, load_history, save_checkpoint, save_history
from engine.training_runner import TrainingRunner
from ui.charts import line_chart
from ui.grid_render import render_grid
from ui.sidebar_helpers import train_stop_reset

ROOM_ID = "room3"

st.title("🔁 Room 3 — The On-Policy Corridor")

defaults = {
    "history": None, "checkpoints": None, "v_start": None, "gamma": None,
    "policy": None, "values": None, "env": None, "training": False,
    "attempt_path": None, "attempt_steps": None, "attempt_success": None, "attempt_g": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(f"{ROOM_ID}_{key}", value)

runner: TrainingRunner = st.session_state.setdefault(f"{ROOM_ID}_runner", TrainingRunner())


def _train_fn(cfg, emit, stop_flag_ref):
    """Runs on the TrainingRunner thread. Builds the room's grid from the tunable
    env params, then hands off to the generic SARSA loop, which emits metrics/result."""
    grid = make_room3_grid(cfg["slip_prob"], cfg["trap_reward"])
    train_sarsa(grid, emit=emit, stop_flag_ref=stop_flag_ref, **cfg["sarsa"])


def _moving_avg(xs, window=50):
    out, running = [], 0.0
    dq: deque = deque()
    for x in xs:
        dq.append(x)
        running += x
        if len(dq) > window:
            running -= dq.popleft()
        out.append(running / len(dq))
    return out


# reload the last run from disk on a cold page load, so results survive an app restart
if st.session_state[f"{ROOM_ID}_history"] is None and not st.session_state[f"{ROOM_ID}_training"]:
    saved = load_history(ROOM_ID)
    saved_cps = load_checkpoint(ROOM_ID, "snapshots")
    if saved and saved_cps:
        st.session_state[f"{ROOM_ID}_history"] = {
            k: saved[k] for k in ("reward", "steps", "td_error", "epsilon", "alpha")
        }
        st.session_state[f"{ROOM_ID}_checkpoints"] = saved_cps
        st.session_state[f"{ROOM_ID}_v_start"] = saved.get("v_start")
        st.session_state[f"{ROOM_ID}_gamma"] = saved.get("gamma")
        st.session_state[f"{ROOM_ID}_policy"] = saved_cps[-1]["policy"]
        st.session_state[f"{ROOM_ID}_values"] = saved_cps[-1]["values"]
        st.session_state[f"{ROOM_ID}_env"] = {
            "slip_prob": saved.get("slip_prob", 0.25),
            "trap_reward": saved.get("trap_reward", -20.0),
        }

st.header("Room 3 controls")
training = st.session_state[f"{ROOM_ID}_training"]

env_col, algo_col, explore_col = st.columns(3)
with env_col:
    st.markdown("**Environment**")
    slip_prob = st.slider("Slip probability", 0.0, 0.9, 0.25, 0.05, disabled=training)
    trap_reward = st.slider("Trap reward", -100.0, -1.0, -20.0, 1.0, disabled=training,
                            help="Stepping on a trap costs this much, but doesn't end the episode.")
    st.caption(f"Goal reward fixed at {ROOM3_GOAL_REWARD:.0f} (like Room 1). Board layout is fixed.")

with algo_col:
    st.markdown("**SARSA**")
    alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01, disabled=training)
    decay_alpha = st.toggle("Decay α over training", value=False, disabled=training,
                            help="Ramp α down to a tenth on the same schedule as ε.")
    gamma = st.slider("Gamma (γ)", 0.5, 1.0, 0.9, 0.01, disabled=training)
    episodes = st.slider("Episodes", 100, 10000, 3000, 100, disabled=training)
    max_steps = st.slider("Max steps / episode", 20, 500, 200, 10, disabled=training)

with explore_col:
    st.markdown("**Exploration & replay**")
    eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05, disabled=training)
    eps_end = st.slider("ε end", 0.0, 0.5, 0.05, 0.01, disabled=training)
    eps_decay_fraction = st.slider("ε decay fraction", 0.1, 1.0, 0.8, 0.05, disabled=training,
                                   help="Fraction of episodes over which ε linearly decays to ε end.")
    snapshot_interval = st.slider("Snapshot interval", 10, 500, 50, 10, disabled=training,
                                  help="Save a checkpoint (Q-derived policy + a greedy rollout) every N episodes.")
    max_attempt_steps = st.slider("Max attempt steps", 20, 300, 150, 10)

start, stop, reset = train_stop_reset(training)

st.divider()

if reset and not training:
    for key, value in defaults.items():
        st.session_state[f"{ROOM_ID}_{key}"] = value
    st.rerun()

if start and not training:
    for key in ("history", "checkpoints", "policy", "values", "v_start",
                "attempt_path", "attempt_steps", "attempt_success", "attempt_g"):
        st.session_state[f"{ROOM_ID}_{key}"] = defaults[key]
    st.session_state[f"{ROOM_ID}_training"] = True
    st.session_state[f"{ROOM_ID}_gamma"] = gamma
    st.session_state[f"{ROOM_ID}_env"] = {"slip_prob": slip_prob, "trap_reward": trap_reward}
    cfg = {
        "slip_prob": slip_prob,
        "trap_reward": trap_reward,
        "sarsa": {
            "episodes": episodes, "gamma": gamma, "alpha": alpha, "decay_alpha": decay_alpha,
            "epsilon_start": eps_start, "epsilon_end": eps_end,
            "epsilon_decay_fraction": eps_decay_fraction, "max_steps": max_steps,
            "snapshot_interval": snapshot_interval,
        },
    }
    runner.start(_train_fn, cfg)
    st.rerun()

if stop and training:
    runner.stop()

# drain whatever the training thread has emitted since the last rerun
for msg_type, payload in runner.drain():
    if msg_type in ("metrics", "result"):
        st.session_state[f"{ROOM_ID}_history"] = payload["history"]
        st.session_state[f"{ROOM_ID}_checkpoints"] = payload["checkpoints"]
    if msg_type == "result":
        st.session_state[f"{ROOM_ID}_policy"] = payload["policy"]
        st.session_state[f"{ROOM_ID}_values"] = payload["values"]
        st.session_state[f"{ROOM_ID}_v_start"] = payload["v_start"]
        env = st.session_state[f"{ROOM_ID}_env"]
        save_checkpoint(ROOM_ID, "snapshots", payload["checkpoints"])
        save_history(ROOM_ID, {
            **payload["history"],
            "v_start": payload["v_start"],
            "gamma": st.session_state[f"{ROOM_ID}_gamma"],
            "episodes": payload["episodes_run"],
            "slip_prob": env["slip_prob"],
            "trap_reward": env["trap_reward"],
        })
    if msg_type == "done":
        st.session_state[f"{ROOM_ID}_training"] = False

tab_info, tab_train, tab_board = st.tabs(["ℹ️ Info", "📈 Train Result", "🏁 Run episode"])

with tab_info:
    preview_grid = make_room3_grid(slip_prob=slip_prob, trap_reward=trap_reward)
    st.plotly_chart(render_grid(preview_grid, title="Room 3 layout"), use_container_width=True)
    st.markdown(Path("docs/room3.md").read_text(encoding="utf-8"))

with tab_train:
    history = st.session_state[f"{ROOM_ID}_history"]
    if st.session_state[f"{ROOM_ID}_training"]:
        st.info("🏃 Training… metrics update live below.")
    if history is None or not history["reward"]:
        st.caption("Configure parameters above, then press ▶ Train.")
    else:
        rewards = history["reward"]
        steps = history["steps"]
        td = history["td_error"]
        eps = history["epsilon"]
        alpha_hist = history["alpha"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes", len(rewards))
        c2.metric("Final ε", f"{eps[-1]:.3f}")
        v_start = st.session_state[f"{ROOM_ID}_v_start"]
        c3.metric("V(start)", "—" if v_start is None else f"{v_start:.2f}",
                  help="max_a Q(start, a) from the learned Q-table — the score of the training.")

        st.plotly_chart(line_chart({"reward": rewards, "smoothed": _moving_avg(rewards)},
                                   title="Reward per episode"), use_container_width=True)
        st.plotly_chart(line_chart({"steps": steps, "smoothed": _moving_avg(steps)},
                                   title="Steps per episode"), use_container_width=True)
        st.plotly_chart(line_chart({"mean |TD-error|": td}, title="Mean |TD-error| per episode",
                                   log_y=True), use_container_width=True)
        decayed = {"ε": eps}
        if any(a != alpha_hist[0] for a in alpha_hist):
            decayed["α"] = alpha_hist
        st.plotly_chart(line_chart(decayed, title="Exploration / learning-rate schedule"),
                        use_container_width=True)

        st.subheader("Recent episodes")
        n = min(20, len(rewards))
        st.dataframe({
            "Episode": list(range(len(rewards) - n + 1, len(rewards) + 1)),
            "Reward": [f"{r:.2f}" for r in rewards[-n:]],
            "Steps": steps[-n:],
            "Mean |TD|": [f"{t:.3f}" for t in td[-n:]],
        }, use_container_width=True, hide_index=True)

with tab_board:
    checkpoints = st.session_state[f"{ROOM_ID}_checkpoints"]
    env = st.session_state[f"{ROOM_ID}_env"]
    if not checkpoints or env is None:
        st.info("Train the room first to inspect and replay the learned policy.")
    else:
        grid = make_room3_grid(env["slip_prob"], env["trap_reward"])

        mode = st.radio("View", ["📊 Checkpoint snapshot", "🏁 Escape attempt"], horizontal=True)

        if mode == "📊 Checkpoint snapshot":
            idx = st.slider("Checkpoint", 0, len(checkpoints) - 1, len(checkpoints) - 1)
            cp = checkpoints[idx]
            fig = render_grid(grid, values=cp["values"], policy=cp["policy"],
                              title=f"After {cp['episode']} episodes")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The Q-derived value heatmap and greedy policy as of this checkpoint — "
                       "drag to watch the policy sharpen as SARSA learns.")
        else:
            policy = st.session_state[f"{ROOM_ID}_policy"] or checkpoints[-1]["policy"]
            if st.button("🏁 Run escape attempt"):
                path, rewards, steps, success = run_episode(grid, policy, max_attempt_steps)
                solved_gamma = st.session_state[f"{ROOM_ID}_gamma"] or gamma
                st.session_state[f"{ROOM_ID}_attempt_path"] = path
                st.session_state[f"{ROOM_ID}_attempt_steps"] = steps
                st.session_state[f"{ROOM_ID}_attempt_success"] = success
                st.session_state[f"{ROOM_ID}_attempt_g"] = discounted_return(rewards, solved_gamma)

            path = st.session_state[f"{ROOM_ID}_attempt_path"]
            if path is None:
                st.caption("Click 'Run escape attempt' to roll out the greedy policy once, with slip applied.")
            else:
                step_idx = st.slider("Step", 0, len(path) - 1, 0)
                values = st.session_state[f"{ROOM_ID}_values"] or checkpoints[-1]["values"]
                fig = render_grid(grid, values=values, policy=policy,
                                  agent_pos=path[step_idx], path=path, title="Escape attempt")
                st.plotly_chart(fig, use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Steps taken", st.session_state[f"{ROOM_ID}_attempt_steps"])
                result = "✅ Escaped" if st.session_state[f"{ROOM_ID}_attempt_success"] else "❌ Failed"
                m2.metric("Result", result)
                m3.metric("G (this episode)", f"{st.session_state[f'{ROOM_ID}_attempt_g']:.2f}",
                          help="The actual discounted return collected during this one rollout.")
                v_start = st.session_state[f"{ROOM_ID}_v_start"]
                m4.metric("V(start)", "—" if v_start is None else f"{v_start:.2f}",
                          help="What training predicted the start was worth — compare against G.")

# keep polling while the background thread is training
if st.session_state[f"{ROOM_ID}_training"]:
    import time
    time.sleep(0.4)
    st.rerun()
