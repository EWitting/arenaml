"""End-to-end tests that fit real models through AutoGluon (slow, CPU only)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes

import arenaml
from arenaml.evaluate import CVStrategy, run_config
from arenaml.leaderboard import load_leaderboard

pytestmark = [pytest.mark.live, pytest.mark.artifacts]

METHODS = ["XGBoost", "LightGBM", "RandomForest", "LinearModel"]


@pytest.fixture(scope="module")
def lb():
    return load_leaderboard(methods=METHODS)


def test_run_config_holdout_and_kfold(lb, tmp_path):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    spec = lb.configs.loc["XGBoost_c1_BAG_L1"]
    for cv in (CVStrategy("holdout", holdout_frac=0.25), CVStrategy("kfold", n_folds=3)):
        res = run_config(
            "XGBoost_c1_BAG_L1",
            spec["model_cls"],
            spec["hyperparameters"],
            X,
            y,
            "binary",
            "roc_auc",
            cv=cv,
            time_limit=120,
            output_dir=tmp_path,
        )
        assert res.ok, res.error
        assert 0.0 <= res.metric_error_val < 0.2
        assert res.seconds > 0
        assert len(res.predictor.predict(X.head(5))) == 5
        res.cleanup()
        assert res.predictor is None


def test_search_binary(lb, tmp_path):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    s = arenaml.search(
        X,
        y,
        time_budget=180,
        max_evals=4,
        leaderboard=lb,
        cv=CVStrategy("holdout", holdout_frac=0.25),
        output_dir=tmp_path,
        verbosity=0,
    )
    assert s.problem_type_ == "binary" and s.eval_metric_ == "roc_auc"
    assert 1 <= len(s.history_) <= 4
    assert s.history_["is_best"].sum() >= 1
    assert s.best_error_ == s.history_.loc[s.history_["status"] == "ok", "metric_error_val"].min()
    proba = s.predict_proba(X.head(3))
    assert proba.shape == (3, 2)
    assert s.predictions().shape[0] == len(s.candidates_)
    weights = s.task_weights()
    assert weights.shape[0] == 51 and np.isfinite(weights["performance"]).all()


def test_search_regression_same_type_tasks(lb, tmp_path):
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    s = arenaml.search(
        X,
        y,
        time_budget=120,
        max_evals=2,
        leaderboard=lb,
        models=["LightGBM", "RandomForest"],
        cv=CVStrategy("holdout"),
        source_tasks="same_type",
        output_dir=tmp_path,
        verbosity=0,
    )
    assert s.problem_type_ == "regression" and s.eval_metric_ == "rmse"
    assert s.task_weights().shape[0] == 13
    assert set(s.history_["method"]) <= {"LightGBM", "RandomForest"}
    assert len(s.predict(X.head(4))) == 4
