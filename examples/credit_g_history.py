"""Run a 5-minute search on OpenML credit-g, save the history and plot the search trajectory.

Produces ``examples/output/credit_g_history.csv`` and ``examples/output/credit_g_history.png``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

import arenaml
from arenaml import CVStrategy

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
METHODS = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "ExtraTrees", "LinearModel", "KNeighbors"]
# Categorical palette, fixed order per model family (validated for colour-vision deficiency).
COLORS = dict(
    zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"], strict=True)
)
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"


def run() -> arenaml.ArenaSearch:
    X, y = fetch_openml("credit-g", version=1, return_X_y=True, as_frame=True)
    search = arenaml.search(
        X,
        y,
        time_budget=300,
        models=METHODS,
        cv=CVStrategy("holdout", holdout_frac=0.25),
        log_cost=False,  # raw-seconds runtime surrogate for EI/cost, instead of the log-space default
        output_dir=OUT / "runs",
        verbosity=1,
    )
    search.history_.to_csv(OUT / "credit_g_history.csv", index=False)
    # Task weights of both surrogates after every evaluation (columns perf:<task> and cost:<task>).
    search.weight_history_.to_csv(OUT / "credit_g_weight_history.csv", index=False)
    return search


def plot(history: pd.DataFrame, dummy_error: float, lb_best: float | None) -> None:
    h = history[history["status"] == "ok"].copy()
    h["pred_error"] = -h["pred_mean"] * dummy_error
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # 1. trajectory: every evaluation and the incumbent, over wall-clock time
    ax = axes[0]
    for m in METHODS:
        sub = h[h["method"] == m]
        if len(sub):
            ax.scatter(
                sub["elapsed"],
                sub["metric_error_val"],
                s=22,
                color=COLORS[m],
                label=m,
                zorder=3,
                edgecolor="white",
                linewidth=0.6,
            )
    ax.step(
        h["elapsed"],
        h["best_error_so_far"],
        where="post",
        color=INK,
        linewidth=2,
        label="best so far",
        zorder=4,
    )
    if lb_best is not None:
        ax.axhline(
            lb_best,
            color=MUTED,
            linewidth=1.2,
            linestyle="--",
            zorder=2,
            label="best TabArena config on credit-g (leaderboard split)",
        )
    ax.set_xlabel("elapsed seconds", color=MUTED)
    ax.set_ylabel("1 - AUC (holdout)", color=MUTED)
    ax.set_title("Validation error of each evaluated configuration", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2, labelcolor=INK)

    # 2. regret relative to the best configuration found, per evaluation
    ax = axes[1]
    final_best = h["metric_error_val"].min()
    ax.plot(h["step"], h["best_error_so_far"] - final_best, color=INK, linewidth=2, drawstyle="steps-post")
    ax.set_xlabel("evaluation", color=MUTED)
    ax.set_ylabel("regret to best found", color=MUTED)
    ax.set_title("Regret (incumbent error minus final best)", color=INK, fontsize=11, loc="left")
    ax.set_ylim(bottom=0)

    # 3. surrogate calibration: predicted vs. observed error, with prediction std as error bar
    ax = axes[2]
    for m in METHODS:
        sub = h[h["method"] == m]
        if len(sub):
            ax.errorbar(
                sub["pred_error"],
                sub["metric_error_val"],
                xerr=sub["pred_std"] * dummy_error,
                fmt="o",
                ms=4.5,
                color=COLORS[m],
                ecolor=COLORS[m],
                elinewidth=0.8,
                alpha=0.9,
                label=m,
            )
    lo, hi = (
        h[["pred_error", "metric_error_val"]].min().min(),
        h[["pred_error", "metric_error_val"]].max().max(),
    )
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("predicted 1 - AUC at selection time", color=MUTED)
    ax.set_ylabel("observed 1 - AUC", color=MUTED)
    ax.set_title("Surrogate prediction vs. observation", color=INK, fontsize=11, loc="left")

    fig.suptitle(
        "arenaml on credit-g, 300 s budget, holdout validation", color=INK, fontsize=12, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(OUT / "credit_g_history.png", dpi=150)


if __name__ == "__main__":
    s = run()
    print(s.summary())
    print(s.history_["method"].value_counts())
    lb_best = None
    if "credit-g" in s.leaderboard_.task_names:
        col = s.leaderboard_.error.loc[s.candidates_, "credit-g"]
        lb_best = float(np.nanmin(col))
    plot(s.history_, s.dummy_error_, lb_best)
    print("saved", OUT / "credit_g_history.csv", OUT / "credit_g_history.png")


def replot_from_csv() -> None:
    """Redraw the figure from a saved history without rerunning the search."""
    from arenaml import load_leaderboard

    history = pd.read_csv(OUT / "credit_g_history.csv")
    lb = load_leaderboard(methods=METHODS)
    plot(history, 0.5, float(np.nanmin(lb.error["credit-g"])))
