"""Generic Plotly line-chart helper reused by every room's Train tab."""
from __future__ import annotations

import plotly.graph_objects as go


def line_chart(series: dict[str, list[float]], title: str = "", log_y: bool = False) -> go.Figure:
    """series: {trace_name: y_values}, each plotted against its own 1..N x-axis."""
    fig = go.Figure()
    for name, ys in series.items():
        fig.add_trace(go.Scatter(x=list(range(1, len(ys) + 1)), y=ys, mode="lines", name=name))
    fig.update_layout(title=title, height=320, margin=dict(l=40, r=20, t=40, b=30),
                      legend=dict(orientation="h", y=1.15))
    if log_y:
        fig.update_yaxes(type="log")
    return fig
