"""EscapeRoomRL — lobby page.

Six rooms, one RL algorithm each, increasing in difficulty. Every room is reachable any
time from the sidebar nav; this page just shows what each room is and its trained V(start)
so far — the score of the training itself, not any one noisy rollout's G. See plan.md for
the full design and SPRINTS.md for the build plan.
"""
import streamlit as st

from engine.storage import load_history

st.set_page_config(page_title="EscapeRoomRL", page_icon="🔑", layout="wide")

ROOMS = [
    ("room1", "pages/1_room1_dynamic_programming.py", "Dynamic Programming", "🧮"),
    ("room2", "pages/2_room2_monte_carlo.py", "Monte Carlo", "🎲"),
    ("room3", "pages/3_room3_sarsa.py", "SARSA", "🔁"),
    ("room4", "pages/4_room4_q_learning.py", "Q-Learning", "🎯"),
    ("room5", "pages/5_room5_continuous_dqn.py", "Continuous DQN", "🚀"),
    ("room6", "pages/6_room6_dynamic_obstacles.py", "Dynamic Obstacles", "🕳️"),
]


def lobby() -> None:
    st.title("🔑 EscapeRoomRL")
    st.write(
        "Six rooms, six algorithms, increasing difficulty. Train an agent in each room "
        "from its own sidebar, then watch it escape on the Board tab — the faster the "
        "escape, the higher the score."
    )
    st.subheader("Scoreboard")
    cols = st.columns(len(ROOMS))
    for col, (room_id, _, title, icon) in zip(cols, ROOMS):
        with col:
            st.markdown(f"**{icon} {title}**")
            history = load_history(room_id)
            if history and "v_start" in history:
                st.metric("V(start)", f"{history['v_start']:.2f}")
            else:
                st.caption("Not trained yet")


nav = st.navigation(
    [st.Page(lobby, title="Lobby", icon="🔑", default=True)]
    + [st.Page(path, title=title, icon=icon) for _, path, title, icon in ROOMS]
)
nav.run()
