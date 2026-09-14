"""Tests against the real TabArena artifacts (cached under ~/.cache/tabarena or downloaded)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arenaml.applicability import TargetProfile, applicability, resolve_model_cls
from arenaml.leaderboard import build_leaderboard
from arenaml.normalization import DUMMY_LEVEL

pytestmark = pytest.mark.artifacts

METHODS = ["XGBoost", "LightGBM", "TabICL_GPU", "TabPFNv2_GPU", "LinearModel"]


@pytest.fixture(scope="module")
def lb():
    return build_leaderboard(methods=METHODS)


def test_shapes_and_dummy(lb):
    assert lb.error.shape[1] == 51
    assert lb.error.shape == lb.cost.shape
    assert list(lb.error.index) == list(lb.configs.index)
    assert set(lb.configs["method"]) == set(METHODS)
    assert (
        lb.tasks["problem_type"].value_counts()
        == pd.Series({"binary": 30, "regression": 13, "multiclass": 8})
    ).all()
    binary = lb.tasks["problem_type"] == "binary"
    assert np.allclose(lb.dummy_error[binary], 0.5)
    assert (lb.dummy_error > 0).all()


def test_imputed_rows_are_missing_not_random_forest(lb):
    support = lb.method_support()
    assert support.loc["TabICL_GPU", "regression"] == 0.0  # classification-only
    assert support.loc["XGBoost"].min() == 1.0
    assert support.loc["TabPFNv2_GPU", "binary"] < 1.0  # size constraints
    perf = lb.normalized_performance()
    assert perf.max().max() <= 0.0 + 1e-9  # error / dummy >= 0 -> performance <= 0
    assert perf.isna().sum().sum() == lb.error.isna().sum().sum()


def test_config_specs_are_runnable(lb):
    spec = lb.configs.loc["XGBoost_c1_BAG_L1"]
    assert spec["model_cls"] == "XGBoostModel"
    assert "ag_args" in spec["hyperparameters"]
    cls = resolve_model_cls(spec["model_cls"])
    assert cls.__name__ == "XGBoostModel"


def test_applicability_rules(lb):
    reg = TargetProfile("regression", n_samples=1000, n_features=10, n_classes=0, n_train_per_fold=875)
    table = applicability(lb, reg, include_gpu_models=True, check_dependencies=False)
    reasons = table.groupby("method")["reason"].agg(lambda s: set(s.dropna()))
    assert reasons["TabICL_GPU"] == {"problem_type_unsupported"}
    assert reasons["XGBoost"] == set()
    big = TargetProfile("binary", n_samples=50000, n_features=10, n_classes=2, n_train_per_fold=43750)
    table = applicability(lb, big, include_gpu_models=True, check_dependencies=False)
    assert set(table.loc[table["method"] == "TabPFNv2_GPU", "reason"]) == {"dataset_size_constraint"}
    table = applicability(lb, big, include_gpu_models=False, check_dependencies=False)
    assert set(table.loc[table["method"] == "TabICL_GPU", "reason"]) == {"gpu_required"}
    table = applicability(lb, big, methods=["XGBoost"], include_gpu_models=False, check_dependencies=False)
    assert table["applicable"].sum() == 201
    with pytest.raises(ValueError):
        applicability(lb, big, methods=["NotAMethod"])


def test_dummy_level_fill(lb):
    perf = lb.normalized_performance().fillna(DUMMY_LEVEL)
    assert perf.min().min() >= DUMMY_LEVEL - 10  # failed runs can be worse than dummy but bounded
