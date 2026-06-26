"""Plotly rendering for the 10x10 grid rooms (Rooms 1-4): a value heatmap with
policy arrows and start/goal/slippery/trap/wall markers overlaid. One renderer reused
by all four grid rooms instead of four near-identical ones.
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

    fig = go.Figure(data=go.Heatmap(z=z, colorscale="Blues", showscale=True))

    for (i, j) in grid.walls:
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                      fillcolor="#1e293b", line_width=0)
    for (i, j) in grid.slippery:
        fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                      fillcolor="rgba(56,189,248,0.20)", line_width=0)
    for (i, j) in grid.traps:
        fig.add_annotation(x=j, y=i, text="✕", showarrow=False, font=dict(size=20, color="#ef4444"))

    gi, gj = grid.goal
    fig.add_annotation(x=gj, y=gi, text="🚪", showarrow=False, font=dict(size=20))
    si, sj = grid.start
    fig.add_annotation(x=sj, y=si, text="●", showarrow=False, font=dict(size=12, color="#22c55e"))

    if policy:
        # traps aren't terminal, so the policy still recommends a move from one —
        # only the goal itself has no action to show.
        for (i, j), a in policy.items():
            if (i, j) == grid.goal:
                continue
            fig.add_annotation(x=j, y=i, text=ACTION_ARROWS[a], showarrow=False, font=dict(size=16))

    if agent_pos:
        ai, aj = agent_pos
        fig.add_trace(go.Scatter(x=[aj], y=[ai], mode="markers",
                                 marker=dict(size=16, color="#f472b6"), showlegend=False))

    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(title=title, height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig
