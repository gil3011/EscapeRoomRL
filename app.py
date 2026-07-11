"""EscapeRoomRL — lobby page.

Six rooms, one RL algorithm each, increasing in difficulty. Every room is reachable any
time from the top navigation bar; this page just explains what each room is and what to
expect inside it. See plan.md for the full design and SPRINTS.md for the build plan.
"""
import streamlit as st

st.set_page_config(page_title="EscapeRoomRL", page_icon="🔑", layout="wide")

# In top navigation the default (Lobby) page isn't listed as a tab — the logo is the
# idiomatic "home" control, and clicking it returns to the default page.
st.logo("assets/logo.svg", link=None)

ROOMS = [
    {
        "id": "room1", "path": "pages/1_room1_dynamic_programming.py",
        "title": "Dynamic Programming", "icon": "🧮", "name": "The Frozen Vault",
        "ready": True,
        "blurb": "The environment's rules are fully known. Instead of trial-and-error, the "
                 "agent solves for the optimal escape directly with value and policy "
                 "iteration — the Bellman equation in action.",
    },
    {
        "id": "room2", "path": "pages/2_room2_monte_carlo.py",
        "title": "Monte Carlo", "icon": "🎲", "name": "The Unknown Vault",
        "ready": False,
        "blurb": "The rules are now hidden. The agent learns only from complete episodes — "
                 "averaging the returns it actually experiences, with an exploring-starts "
                 "option.",
    },
    {
        "id": "room3", "path": "pages/3_room3_sarsa.py",
        "title": "SARSA", "icon": "🔁", "name": "The On-Policy Corridor",
        "ready": True,
        "blurb": "Model-free and on-policy: the agent updates its value estimate after every "
                 "step, learning from the action its exploration policy actually took. Adds a "
                 "slippery corridor and a shortcut tile to discover.",
    },
    {
        "id": "room4", "path": "pages/4_room4_q_learning.py",
        "title": "Q-Learning", "icon": "🎯", "name": "The Off-Policy Cellar",
        "ready": True,
        "blurb": "The hardest grid room — the same board as Room 3, but off-policy learning "
                 "that always bootstraps from the best next move, while timing a crossing past "
                 "a patrolling enemy.",
    },
    {
        "id": "room5", "path": "pages/5_room5_continuous_dqn.py",
        "title": "Continuous DQN", "icon": "🚀", "name": "The Open Floor",
        "ready": False,
        "blurb": "No grid at all — a continuous room where position is real-valued. A small "
                 "neural network (DQN) approximates the value function the table could no "
                 "longer hold.",
    },
    {
        "id": "room6", "path": "pages/6_room6_dynamic_obstacles.py",
        "title": "Dynamic Obstacles", "icon": "🕳️", "name": "The Obstacle Gauntlet",
        "ready": False,
        "blurb": "Obstacles reshuffle every episode and the agent only senses the nearest few "
                 "— a test of whether a learned policy can generalize to a room it has never "
                 "seen before.",
    },
]


def lobby() -> None:
    st.title("🔑 EscapeRoomRL")
    st.write(
        "Six rooms, six reinforcement-learning algorithms, increasing in difficulty. Each "
        "room is a self-contained escape puzzle that introduces one new idea. Pick a room "
        "from the navigation bar at the top, train an agent, then watch it escape."
    )
    st.divider()

    cols = st.columns(3)
    for i, room in enumerate(ROOMS):
        with cols[i % 3]:
            st.markdown(f"### {room['icon']} {room['name']}")
            status = "" if room["ready"] else " · 🚧 _coming soon_"
            st.caption(f"{room['title']}{status}")
            st.write(room["blurb"])


nav = st.navigation(
    [st.Page(lobby, title="Lobby", icon="🔑", default=True)]
    + [st.Page(r["path"], title=r["title"], icon=r["icon"]) for r in ROOMS],
    position="top",
)
nav.run()
