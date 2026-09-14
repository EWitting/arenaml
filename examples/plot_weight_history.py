"""Interactive Plotly view of the surrogate task weights over the course of a search.

Reads the ``credit_g_weight_history.csv`` produced by ``credit_g_history.py`` (or any other
``weight_history_`` dump with the same wide layout: a ``step`` column plus one ``perf:<task>``
column and, if the runtime surrogate was used, one ``cost:<task>`` column per benchmark task)
and writes a self-contained interactive HTML with one line per task, weight against evaluation
step. A dropdown switches between the performance-surrogate weights and the runtime-surrogate
weights.

Run after ``credit_g_history.py`` (it reuses that script's output), or point ``WEIGHT_CSV`` at
any other ``weight_history_.to_csv(...)`` file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

OUT = Path(__file__).parent / "output"
WEIGHT_CSV = OUT / "credit_g_weight_history.csv"
TARGET_TASK = "credit-g"  # the target dataset's own TabArena task, highlighted if present
N_HIGHLIGHT = 4  # additional tasks to highlight, by final |weight|

# Categorical palette (validated, colour-vision-deficiency-safe) for the highlighted lines;
# every other task recedes into a single muted grey so 40+ lines stay legible.
HIGHLIGHT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
OTHER_COLOR = "#c9c8c0"
INK, MUTED, GRID, SURFACE = "#1a1a19", "#6b6a63", "#e6e5df", "#fcfcfb"


def _task_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c[len(prefix) :] for c in df.columns if c.startswith(prefix)]


def _pick_highlights(df: pd.DataFrame, tasks: list[str], prefix: str) -> list[str]:
    """Target task (if present) first, then the tasks with the largest final |weight|."""
    final = df[[f"{prefix}{t}" for t in tasks]].iloc[-1]
    final.index = tasks
    ordered = list(final.abs().sort_values(ascending=False).index)
    highlights = [TARGET_TASK] if TARGET_TASK in tasks else []
    for t in ordered:
        if len(highlights) >= 1 + N_HIGHLIGHT:
            break
        if t not in highlights:
            highlights.append(t)
    return highlights


def _add_task_traces(
    fig: go.Figure, df: pd.DataFrame, tasks: list[str], prefix: str, visible: bool
) -> tuple[list[int], str]:
    """Add one background-grey trace per non-highlighted task, then one coloured trace per
    highlighted task (drawn last so they sit on top). Returns the trace indices added and the
    name of the single legend group used for the "other tasks" line so it can be excluded from
    the highlight legend.
    """
    highlights = _pick_highlights(df, tasks, prefix)
    others = [t for t in tasks if t not in highlights]
    start = len(fig.data)

    for i, t in enumerate(others):
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df[f"{prefix}{t}"],
                mode="lines",
                line={"color": OTHER_COLOR, "width": 1},
                opacity=0.45,
                name=f"other tasks ({len(others)})",
                legendgroup=f"{prefix}other",
                showlegend=(i == 0),
                hovertemplate=f"{t}<br>step %{{x}}<br>weight %{{y:.4f}}<extra></extra>",
                visible=visible,
            )
        )
    for t, color in zip(highlights, HIGHLIGHT_COLORS, strict=False):
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df[f"{prefix}{t}"],
                mode="lines",
                line={"color": color, "width": 2.6},
                name=t,
                legendgroup=f"{prefix}{t}",
                hovertemplate=f"<b>{t}</b><br>step %{{x}}<br>weight %{{y:.4f}}<extra></extra>",
                visible=visible,
            )
        )
    return list(range(start, len(fig.data))), f"{prefix}other"


def build_figure(df: pd.DataFrame) -> go.Figure:
    perf_tasks = _task_columns(df, "perf:")
    cost_tasks = _task_columns(df, "cost:")
    n_tasks = len(perf_tasks)
    uniform = 1.0 / n_tasks

    fig = go.Figure()
    perf_idx, _ = _add_task_traces(fig, df, perf_tasks, "perf:", visible=True)
    cost_idx: list[int] = []
    if cost_tasks:
        cost_idx, _ = _add_task_traces(fig, df, cost_tasks, "cost:", visible=False)

    # Reference line: the uniform cold-start weight (1 / n_tasks), shown for the performance view.
    fig.add_trace(
        go.Scatter(
            x=[df["step"].min(), df["step"].max()],
            y=[uniform, uniform],
            mode="lines",
            line={"color": MUTED, "width": 1, "dash": "dash"},
            name=f"uniform cold start (1/{n_tasks})",
            hoverinfo="skip",
            visible=True,
        )
    )
    uniform_idx = len(fig.data) - 1

    all_idx = list(range(len(fig.data)))
    perf_visible = [i in perf_idx or i == uniform_idx for i in all_idx]
    cost_visible_arr = [i in cost_idx for i in all_idx]

    fig.update_layout(
        template="plotly_white",
        title={
            "text": "Learned task weights over the search (credit-g)",
            "x": 0.01,
            "xanchor": "left",
            "font": {"color": INK, "size": 17},
        },
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK, "size": 12},
        hovermode="x unified",
        xaxis={
            "title": "evaluation step",
            "gridcolor": GRID,
            "zeroline": False,
            "color": MUTED,
            "title_font": {"color": MUTED},
        },
        yaxis={
            "title": "performance-surrogate weight (per TabArena task)",
            "gridcolor": GRID,
            "zeroline": False,
            "color": MUTED,
            "title_font": {"color": MUTED},
        },
        legend={
            "title": "task (click to isolate; double-click to solo)",
            "font": {"size": 10},
            "bgcolor": SURFACE,
            "bordercolor": GRID,
            "borderwidth": 1,
        },
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.12,
                "xanchor": "left",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Performance weights",
                        "method": "update",
                        "args": [
                            {"visible": perf_visible},
                            {
                                "yaxis": {
                                    "title": "performance-surrogate weight (per TabArena task)",
                                    "gridcolor": GRID,
                                    "zeroline": False,
                                    "color": MUTED,
                                }
                            },
                        ],
                    },
                    {
                        "label": "Runtime weights" if cost_tasks else "Runtime weights (n/a)",
                        "method": "update",
                        "args": [
                            {"visible": cost_visible_arr},
                            {
                                "yaxis": {
                                    "title": "runtime-surrogate weight (per TabArena task, log space)",
                                    "gridcolor": GRID,
                                    "zeroline": False,
                                    "color": MUTED,
                                }
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return fig


if __name__ == "__main__":
    if not WEIGHT_CSV.exists():
        raise SystemExit(f"{WEIGHT_CSV} not found; run examples/credit_g_history.py first.")
    weights = pd.read_csv(WEIGHT_CSV)
    figure = build_figure(weights)
    out_path = OUT / "credit_g_weight_history.html"
    figure.write_html(out_path, include_plotlyjs=True, full_html=True)
    print("saved", out_path)
