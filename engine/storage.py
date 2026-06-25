"""Disk persistence for training runs, shared by every room.

Layout per room:
    runs/<room_id>/history.json            per-episode/iteration metrics
    runs/<room_id>/checkpoints/<tag>.pkl   periodic policy/Q-table/weights snapshots
    runs/<room_id>/best.json               best Escape Score seen so far

Lets a room's Train/Board tabs reload past results after the app restarts, instead of
only living in st.session_state for the current run.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def room_dir(room_id: str) -> Path:
    d = RUNS_DIR / room_id
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    return d


def save_history(room_id: str, history: dict) -> None:
    with open(room_dir(room_id) / "history.json", "w") as f:
        json.dump(history, f)


def load_history(room_id: str) -> dict | None:
    path = room_dir(room_id) / "history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_checkpoint(room_id: str, tag: str, payload) -> None:
    with open(room_dir(room_id) / "checkpoints" / f"{tag}.pkl", "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(room_id: str, tag: str):
    path = room_dir(room_id) / "checkpoints" / f"{tag}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def list_checkpoints(room_id: str) -> list[str]:
    d = room_dir(room_id) / "checkpoints"
    return sorted((p.stem for p in d.glob("*.pkl")), key=lambda t: (len(t), t))


def load_best(room_id: str) -> dict | None:
    path = room_dir(room_id) / "best.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_best_if_higher(room_id: str, score: int, steps: int, success: bool) -> None:
    current = load_best(room_id)
    if current is not None and current["score"] >= score:
        return
    with open(room_dir(room_id) / "best.json", "w") as f:
        json.dump({"score": score, "steps": steps, "success": success}, f)
