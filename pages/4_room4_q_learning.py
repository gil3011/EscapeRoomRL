"""Room 4 — Q-Learning. Off-policy TD(0) on the hardest grid board: a moving patrol
enemy (state augmented to (agent_cell, phase)).

Structurally the twin of Room 3 (SARSA) — same board, same TrainingRunner wiring and tab
layout — so the two rooms read as a matched pair. The only algorithmic difference is the
update rule (max over next actions, not the sampled one); the visible difference is the
moving enemy and the phase slider needed to view a value function that now depends on it.
"""
from collections import deque
from pathlib import Path

import streamlit as st

from engine.boards import ROOM4_ENEMY_REWARD, ROOM4_GOAL_REWARD, make_room4_grid
from engine.grid_world import run_episode
from engine.q_learning_agent import train_q_learning
from engine.scoring import discounted_return
from engine.storage import load_checkpoint, load_history, save_checkpoint, save_history
from engine.training_runner import TrainingRunner
from ui.charts import line_chart
from ui.grid_render import render_grid
from ui.sidebar_helpers import train_stop

ROOM_ID = "room4"

st.title("🎯 Room 4 — The Off-Policy Cellar")

defaults = {
    "history": None, "checkpoints": None, "v_start": None, "gamma": None,
    "policy": None, "values": None, "env": None, "training": False,
    "attempt_path": None, "attempt_steps": None, "attempt_success": None, "attempt_g": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(f"{ROOM_ID}_{key}", value)

runner: TrainingRunner = st.session_state.setdefault(f"{ROOM_ID}_runner", TrainingRunner())


def _train_fn(cfg, emit, stop_flag_ref):
    grid = make_room4_grid(cfg["slip_prob"], cfg["trap_reward"], cfg["enemy_reward"])
    train_q_learning(grid, emit=emit, stop_flag_ref=stop_flag_ref, **cfg["ql"])


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


def _rolling_rates(outcomes, window=100):
    """Escaped / caught / timeout percentages over a moving window of episodes."""
    dq: deque = deque()
    counts = {"goal": 0, "caught": 0, "timeout": 0}
    series = {"escaped %": [], "caught %": [], "timeout %": []}
    label = {"goal": "escaped %", "caught": "caught %", "timeout": "timeout %"}
    for o in outcomes:
        dq.append(o)
        counts[o] += 1
        if len(dq) > window:
            counts[dq.popleft()] -= 1
        n = len(dq)
        for key, lbl in label.items():
            series[lbl].append(100 * counts[key] / n)
    return series


def _project(d, phase):
    """Augmented-state dict {(cell, phase): x} -> {cell: x} for one enemy phase."""
    return {cell: x for (cell, ph), x in d.items() if ph == phase}


# reload the last run from disk on a cold page load
if st.session_state[f"{ROOM_ID}_history"] is None and not st.session_state[f"{ROOM_ID}_training"]:
    saved = load_history(ROOM_ID)
    saved_cps = load_checkpoint(ROOM_ID, "snapshots")
    if saved and saved_cps:
        st.session_state[f"{ROOM_ID}_history"] = {
            k: saved[k] for k in ("reward", "steps", "td_error", "epsilon", "alpha", "outcome")
        }
        st.session_state[f"{ROOM_ID}_checkpoints"] = saved_cps
        st.session_state[f"{ROOM_ID}_v_start"] = saved.get("v_start")
        st.session_state[f"{ROOM_ID}_gamma"] = saved.get("gamma")
        st.session_state[f"{ROOM_ID}_policy"] = saved_cps[-1]["policy"]
        st.session_state[f"{ROOM_ID}_values"] = saved_cps[-1]["values"]
        st.session_state[f"{ROOM_ID}_env"] = {
            "slip_prob": saved.get("slip_prob", 0.25),
            "trap_reward": saved.get("trap_reward", -20.0),
            "enemy_reward": saved.get("enemy_reward", ROOM4_ENEMY_REWARD),
        }

st.header("Room 4 controls")
training = st.session_state[f"{ROOM_ID}_training"]

env_col, algo_col, explore_col = st.columns(3)
with env_col:
    st.markdown("**Environment**")
    slip_prob = st.slider("Slip probability", 0.0, 0.9, 0.25, 0.05, disabled=training)
    trap_reward = st.slider("Trap reward", -100.0, -1.0, -20.0, 1.0, disabled=training,
                            help="Non-terminal trap cost (stepping on a trap doesn't end the episode).")
    enemy_reward = st.slider("Enemy reward", -200.0, -20.0, float(ROOM4_ENEMY_REWARD), 10.0,
                             disabled=training,
                             help="One-time penalty for colliding with the patrol — this DOES end the episode.")
    st.caption(f"Goal reward fixed at {ROOM4_GOAL_REWARD:.0f}. Board + patrol are fixed.")

with algo_col:
    st.markdown("**Q-learning**")
    alpha = st.slider("Learning rate α", 0.01, 1.0, 0.2, 0.01, disabled=training)
    decay_alpha = st.toggle("Decay α over training", value=False, disabled=training)
    gamma = st.slider("Gamma (γ)", 0.5, 1.0, 0.9, 0.01, disabled=training)
    episodes = st.slider("Episodes", 100, 15000, 5000, 100, disabled=training,
                         help="Larger than Room 3 — the (cell, phase) state space is bigger.")
    max_steps = st.slider("Max steps / episode", 20, 500, 200, 10, disabled=training)

with explore_col:
    st.markdown("**Exploration & replay**")
    eps_start = st.slider("ε start", 0.0, 1.0, 1.0, 0.05, disabled=training)
    eps_end = st.slider("ε end", 0.0, 0.5, 0.05, 0.01, disabled=training)
    eps_decay_fraction = st.slider("ε decay fraction", 0.1, 1.0, 0.8, 0.05, disabled=training)
    snapshot_interval = st.slider("Snapshot interval", 10, 1000, 100, 10, disabled=training)
    max_attempt_steps = st.slider("Max attempt steps", 20, 300, 150, 10)

start, stop = train_stop(training)

st.divider()

if start and not training:
    for key in ("history", "checkpoints", "policy", "values", "v_start",
                "attempt_path", "attempt_steps", "attempt_success", "attempt_g"):
        st.session_state[f"{ROOM_ID}_{key}"] = defaults[key]
    st.session_state[f"{ROOM_ID}_training"] = True
    st.session_state[f"{ROOM_ID}_gamma"] = gamma
    st.session_state[f"{ROOM_ID}_env"] = {
        "slip_prob": slip_prob, "trap_reward": trap_reward, "enemy_reward": enemy_reward,
    }
    cfg = {
        "slip_prob": slip_prob, "trap_reward": trap_reward, "enemy_reward": enemy_reward,
        "ql": {
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
            "enemy_reward": env["enemy_reward"],
        })
    if msg_type == "done":
        st.session_state[f"{ROOM_ID}_training"] = False

tab_info, tab_train, tab_board = st.tabs(["ℹ️ Info", "📈 Train Result", "🏁 Run episode"])

with tab_info:
    preview_grid = make_room4_grid(slip_prob=slip_prob, trap_reward=trap_reward,
                                   enemy_reward=enemy_reward)
    st.plotly_chart(render_grid(preview_grid, enemy_pos=preview_grid.enemy_cell(0),
                                patrol_path=preview_grid.patrol_path, title="Room 4 layout"),
                    use_container_width=True)
    st.markdown(Path("docs/room4.md").read_text(encoding="utf-8"))

with tab_train:
    history = st.session_state[f"{ROOM_ID}_history"]
    if st.session_state[f"{ROOM_ID}_training"]:
        st.info("🏃 Training… metrics update live below.")
    if history is None or not history["reward"]:
        st.caption("Configure parameters above, then press ▶ Train.")
    else:
        rewards, steps, td = history["reward"], history["steps"], history["td_error"]
        eps, alpha_hist, outcomes = history["epsilon"], history["alpha"], history["outcome"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes", len(rewards))
        recent = outcomes[-200:]
        c2.metric("Recent escape rate",
                  f"{100 * recent.count('goal') / len(recent):.0f}%" if recent else "—",
                  help="Share of the last 200 episodes that reached the goal.")
        v_start = st.session_state[f"{ROOM_ID}_v_start"]
        c3.metric("V(start)", "—" if v_start is None else f"{v_start:.2f}",
                  help="max_a Q((start, phase 0), a) from the learned Q-table.")

        if outcomes:
            escaped = outcomes.count("goal")
            failed = len(outcomes) - escaped
            s1, s2 = st.columns(2)
            s1.metric("✅ Succeeded (escaped)", f"{escaped:,}",
                      help=f"{100 * escaped / len(outcomes):.1f}% of episodes reached the goal.")
            s2.metric("❌ Failed", f"{failed:,}",
                      help="Episodes caught by the patrol or timed out without escaping.")

        st.plotly_chart(line_chart(_rolling_rates(outcomes),
                                   title="Outcome breakdown (rolling %)"),
                        use_container_width=True)
        st.caption("Escaped / caught / timed-out share over a moving window — watch 'caught' fall "
                   "as Q-learning learns to time its run past the patrol.")
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

with tab_board:
    checkpoints = st.session_state[f"{ROOM_ID}_checkpoints"]
    env = st.session_state[f"{ROOM_ID}_env"]
    if not checkpoints or env is None:
        st.info("Train the room first to inspect and replay the learned policy.")
    else:
        grid = make_room4_grid(env["slip_prob"], env["trap_reward"], env["enemy_reward"])
        mode = st.radio("View", ["📊 Checkpoint snapshot", "🏁 Escape attempt"], horizontal=True)

        if mode == "📊 Checkpoint snapshot":
            idx = st.slider("Checkpoint", 0, len(checkpoints) - 1, len(checkpoints) - 1)
            phase = st.slider("Enemy phase", 0, grid.period - 1, 0,
                              help="The value function depends on where the patrol is — scrub to "
                                   "see V and the policy for each enemy phase.")
            cp = checkpoints[idx]
            fig = render_grid(grid, values=_project(cp["values"], phase),
                              policy=_project(cp["policy"], phase),
                              enemy_pos=grid.enemy_cell(phase), patrol_path=grid.patrol_path,
                              title=f"After {cp['episode']} episodes — enemy phase {phase}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Q-derived value + greedy policy for this enemy phase. The same cell can be "
                       "worth more or less depending on where the patrol is.")
        else:
            policy = st.session_state[f"{ROOM_ID}_policy"] or checkpoints[-1]["policy"]
            values = st.session_state[f"{ROOM_ID}_values"] or checkpoints[-1]["values"]
            if st.button("🏁 Run escape attempt"):
                path, rewards, steps, success = run_episode(grid, policy, max_attempt_steps)
                solved_gamma = st.session_state[f"{ROOM_ID}_gamma"] or gamma
                st.session_state[f"{ROOM_ID}_attempt_path"] = path
                st.session_state[f"{ROOM_ID}_attempt_steps"] = steps
                st.session_state[f"{ROOM_ID}_attempt_success"] = success
                st.session_state[f"{ROOM_ID}_attempt_g"] = discounted_return(rewards, solved_gamma)

            path = st.session_state[f"{ROOM_ID}_attempt_path"]
            if path is None:
                st.caption("Click 'Run escape attempt' to roll out the greedy policy once — the "
                           "enemy moves alongside the agent, so watch the timing.")
            else:
                step_idx = st.slider("Step", 0, len(path) - 1, 0)
                agent_cell, phase = path[step_idx]
                fig = render_grid(grid, values=_project(values, phase),
                                  policy=_project(policy, phase),
                                  agent_pos=agent_cell, path=[c for c, _ph in path],
                                  enemy_pos=grid.enemy_cell(phase), patrol_path=grid.patrol_path,
                                  title="Escape attempt")
                st.plotly_chart(fig, use_container_width=True)
                st.info(
                    f"ℹ️ The arrows and values shown are the policy **for this one enemy "
                    f"position** (phase {phase}). Because the state is `(agent cell, enemy "
                    f"position)`, the agent has a *different* map for each place the 👾 can "
                    f"be — so the arrows redraw every time you step the slider and the enemy "
                    f"moves. A cell can point one way when the enemy is near and another when "
                    f"it's clear."
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Steps taken", st.session_state[f"{ROOM_ID}_attempt_steps"])
                result = ("✅ Escaped" if st.session_state[f"{ROOM_ID}_attempt_success"]
                          else "❌ Caught / timed out")
                m2.metric("Result", result)
                m3.metric("G (this episode)", f"{st.session_state[f'{ROOM_ID}_attempt_g']:.2f}",
                          help="The actual discounted return collected during this one rollout.")
                v_start = st.session_state[f"{ROOM_ID}_v_start"]
                m4.metric("V(start)", "—" if v_start is None else f"{v_start:.2f}",
                          help="What training predicted the start was worth — compare against G.")

if st.session_state[f"{ROOM_ID}_training"]:
    import time
    time.sleep(0.4)
    st.rerun()
