"""Minimal example: find the best TabArena configuration for a dataset in a fixed time budget."""

from __future__ import annotations

import logging

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import arenaml
from arenaml import CVStrategy

logging.basicConfig(level=logging.INFO, format="%(message)s")

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# One line: cost-aware Bayesian optimisation over every applicable leaderboard configuration.
# The first call downloads and caches the TabArena result tables (a few MB per model family).
search = arenaml.search(
    X_train,
    y_train,
    time_budget=600,  # seconds of wall-clock, model fits included
    models="cpu",  # or "all", "gpu", or e.g. ["LightGBM", "CatBoost", "TabPFNv2_GPU"]
    cv=CVStrategy("kfold", n_folds=8),  # TabArena's protocol; CVStrategy("holdout") is cheaper
)

print(search.summary())
print("test accuracy:", (search.predict(X_test).values == y_test.values).mean())

# What the surrogate currently believes about every candidate, and how the benchmark tasks
# were weighted for this dataset.
print(search.predictions().sort_values("pred_mean", ascending=False).head(10))
print(search.task_weights().sort_values("performance", ascending=False).head(10))

# The fitted AutoGluon TabularPredictor of the best configuration is available directly.
predictor = search.best_predictor_
print(predictor.leaderboard())
