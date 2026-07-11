"""Shared sidebar controls reused by every room."""
from __future__ import annotations

import streamlit as st


def train_stop(training: bool) -> tuple[bool, bool]:
    """The standard ▶ Train / ⏹ Stop button row. Returns the two click flags."""
    col1, col2 = st.columns(2)
    start = col1.button("▶ Train", use_container_width=True, disabled=training)
    stop = col2.button("⏹ Stop", use_container_width=True, disabled=not training)
    return start, stop
