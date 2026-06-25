"""Generic background-thread training runner shared by every room.

Generalizes the thread + queue.Queue + Streamlit session_state pattern from
code examples/dql/streamlit_app.py so each room only supplies a train_fn — the thread
lifecycle, stop flag, and queue draining are shared code instead of being rewritten
six times.

Usage in a room page:

    runner = st.session_state.setdefault("runner", TrainingRunner())
    if start_btn:
        runner.start(train_fn, cfg)
    for msg_type, payload in runner.drain():
        ...  # update st.session_state metrics lists
"""
from __future__ import annotations

import queue
import threading
from typing import Callable


class TrainingRunner:
    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.log_q: queue.Queue | None = None
        self._stop_flag_ref: list[bool] | None = None

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, train_fn: Callable, cfg: dict) -> None:
        """Run train_fn(cfg, emit, stop_flag_ref) on a daemon thread.

        train_fn should periodically call emit(msg_type, payload) to report progress
        and check stop_flag_ref[0] to stop early.
        """
        self.log_q = queue.Queue()
        self._stop_flag_ref = [False]
        log_q = self.log_q
        stop_flag_ref = self._stop_flag_ref

        def emit(msg_type: str, payload=None) -> None:
            log_q.put((msg_type, payload))

        def run() -> None:
            train_fn(cfg, emit, stop_flag_ref)
            emit("done", None)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self._stop_flag_ref is not None:
            self._stop_flag_ref[0] = True

    def drain(self) -> list[tuple[str, object]]:
        """Return all currently-queued (msg_type, payload) messages without blocking."""
        messages = []
        if self.log_q is None:
            return messages
        try:
            while True:
                messages.append(self.log_q.get_nowait())
        except queue.Empty:
            pass
        return messages
