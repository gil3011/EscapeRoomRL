"""Shared sidebar controls reused by every room."""
from __future__ import annotations

import streamlit as st


def train_stop_reset(training: bool) -> tuple[bool, bool, bool]:
    """The standard ▶ Train / ⏹ Stop / 🔄 Reset button row. Returns the three click flags."""
    col1, col2 = st.columns(2)
    start = col1.button("▶ Train", width="stretch", disabled=training)
    stop = col2.button("⏹ Stop", width="stretch", disabled=not training)
    reset = st.button("🔄 Reset", width="stretch", disabled=training)
    return start, stop, reset
