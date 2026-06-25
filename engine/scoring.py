"""Escape Score — a display/leaderboard metric computed only at evaluation time.

Distinct from the training reward (see plan.md §2.2): a faster solve already earns more
training reward because every step costs a little, so this score is never fed back into
training. It only exists to be shown on a room's Board tab and the lobby scoreboard.
"""


def escape_score(steps_taken: int, par_steps: int, success: bool) -> int:
    """0-1000: 1000 for an instant exit, 0 for failure (timeout/trap/collision)."""
    if not success or steps_taken <= 0 or par_steps <= 0:
        return 0
    return max(0, round(1000 * (1 - steps_taken / par_steps)))
