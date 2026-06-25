"""Room 5 — Continuous state, DQN. Placeholder until Sprints 6-7 (see SPRINTS.md)."""
import streamlit as st

st.title("🚀 Room 5 — The Open Floor")
st.info("🚧 Not implemented yet — coming in **Sprints 6-7**.")
st.write(
    "No more grid: a continuous 10x10 m room with position (x, y) and discrete velocity "
    "choices (vx, vy ∈ {-1, 0, 1}). The state space is too large to tabulate, so a small "
    "DQN (adapted from code examples/dql/dqn.py) learns the policy instead."
)
