"""Plotly rendering for the 10x10 grid rooms (Rooms 1-4): a value heatmap with
policy arrows and start/goal/slippery/trap/wall markers overlaid. One renderer reused
by all four grid rooms instead of four near-identical ones.

Color language, kept deliberately non-overlapping so no two signals share a hue:
walls = dark slate (structural), slippery = blue tint, traps = red tint, goal =
solid green, start = purple dot, the replayed agent = pink dot. The heatmap itself
uses orange/brown so it never competes with any of those categorical colors —
earlier it used blue, the same hue as the slippery overlay, which was the main
source of confusion.
"""
from __future__ import annotations

import plotly.graph_objects as go

from engine.grid_world import ACTION_ARROWS


def render_grid(grid, values: dict | None = None, policy: dict | None = None,
                 agent_pos: tuple[int, int] | None = None, title: str = "") -> go.Figure:
    size = grid.size
    z = [[0.0] * size for _ in range(size)]
    if values:
        for (i, j), v in values.items():
            z[i][j] = v

    fig = go.Figure(data=go.Heatmap(z=z, colorscale="Oranges", showscale=True,
                                     colorbar=dict(title="V")))

    def cell_rect(i, j, color):
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                      fillcolor=color, line_width=0, layer="above")

    for (i, j) in grid.walls:
        cell_rect(i, j, "#1f2937")
    for (i, j) in grid.slippery:
        cell_rect(i, j, "rgba(59,130,246,0.35)")
    for (i, j) in grid.traps:
        cell_rect(i, j, "rgba(239,68,68,0.35)")
        fig.add_annotation(x=j, y=i, text="✕", showarrow=False, font=dict(size=20, color="#7f1d1d"))

    # the goal's stored V is always 0 (terminal states are never updated by the
    # solver), which would otherwise paint it as the *lowest*-value cell on the
    # heatmap — give it its own solid marker instead of relying on that number.
    gi, gj = grid.goal
    cell_rect(gi, gj, "#16a34a")
    fig.add_annotation(x=gj, y=gi, text="🚪", showarrow=False, font=dict(size=20))

    si, sj = grid.start
    fig.add_annotation(x=sj, y=si, text="●", showarrow=False, font=dict(size=14, color="#7c3aed"))

    if policy:
        # traps aren't terminal, so the policy still recommends a move from one —
        # only the goal itself has no action to show.
        for (i, j), a in policy.items():
            if (i, j) == grid.goal:
                continue
            fig.add_annotation(x=j, y=i, text=ACTION_ARROWS[a], showarrow=False,
                               font=dict(size=16, color="#111827"))

    if agent_pos:
        # an annotation, not a trace, so it always renders above every shape
        # above regardless of layering — the agent marker must never be hidden.
        ai, aj = agent_pos
        fig.add_annotation(x=aj, y=ai, text="●", showarrow=False, font=dict(size=24, color="#ec4899"))

    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title=title, height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig
