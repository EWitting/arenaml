"""Unit tests that need neither TabArena artifacts nor model fits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import arenaml  # noqa: F401 -- import itself must trigger the warm-up, see the test below
from arenaml.acquisition import cost_cooled, expected_improvement
from arenaml.applicability import gpu_available
from arenaml.evaluate import CVStrategy, prepare_hyperparameters
from arenaml.normalization import (
    DUMMY_LEVEL,
    default_metric,
    dummy_error,
    infer_problem_type,
    loss_rescaled,
    normalize_error,
)
from arenaml.optimizer import Optimizer
from arenaml.surrogate import LinearCostSurrogate, LinearPerfSurrogate

# ---------------------------------------------------------------------------- normalisation


def test_default_metrics_match_tabarena():
    assert default_metric("binary") == "roc_auc"
    assert default_metric("multiclass") == "log_loss"
    assert default_metric("regression") == "rmse"
    with pytest.raises(ValueError):
        default_metric("ranking")


def test_normalize_error_direction_and_dummy_level():
    dummy = pd.Series({"a": 0.5, "b": 2.0})
    err = pd.DataFrame({"a": [0.5, 0.1], "b": [2.0, 1.0]}, index=["dummy_like", "better"])
    perf = normalize_error(err, dummy)
    assert np.allclose(perf.loc["dummy_like"], DUMMY_LEVEL)
    assert (perf.loc["better"] > perf.loc["dummy_like"]).all()
    assert perf.loc["better", "a"] == pytest.approx(-0.2)


def test_dummy_error_closed_forms():
    y_bin = pd.Series(["neg"] * 70 + ["pos"] * 30)
    assert dummy_error(y_bin, "binary") == pytest.approx(0.5)
    prior = np.array([0.5, 0.3, 0.2])
    y_multi = pd.Series(np.repeat([0, 1, 2], (prior * 100).astype(int)))
    assert dummy_error(y_multi, "multiclass") == pytest.approx(float(-(prior * np.log(prior)).sum()))
    y_reg = pd.Series(np.random.default_rng(0).normal(3.0, 2.0, 500))
    assert dummy_error(y_reg, "regression") == pytest.approx(float(y_reg.std(ddof=0)))
    assert dummy_error(y_bin, "binary", metric="accuracy") == pytest.approx(0.3)


def test_infer_problem_type():
    assert infer_problem_type(pd.Series(["a", "b", "a"])) == "binary"
    assert infer_problem_type(pd.Series(["x", "y", "z"] * 20)) == "multiclass"
    assert infer_problem_type(pd.Series(np.linspace(0, 1, 200))) == "regression"


def test_loss_rescaled():
    err = pd.Series([0.1, 0.3, 0.5])
    assert loss_rescaled(err).tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert loss_rescaled(pd.Series([0.2, 0.2])).tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------- acquisition


def test_expected_improvement_properties():
    mu = np.array([0.0, 0.0, 1.0, -1.0])
    sigma = np.array([1.0, 0.0, 0.0, 0.5])
    ei = expected_improvement(mu, sigma, f_best=0.0)
    assert ei[0] > 0  # uncertain candidate at the incumbent still has value
    assert ei[1] == 0  # certain and equal to the incumbent
    assert ei[2] == pytest.approx(1.0)  # certain improvement
    assert 0 < ei[3] < ei[0]


def test_cost_cooled():
    scores = np.array([1.0, 1.0, 1.0])
    cost = np.array([1.0, 4.0, np.nan])
    cooled = cost_cooled(scores, cost, alpha=1.0)
    assert cooled[0] == 1.0 and cooled[1] == 0.25
    assert cooled[2] == pytest.approx(1.0 / 2.5)  # NaN replaced by the median of valid costs
    assert np.array_equal(cost_cooled(scores, cost, alpha=0.0), scores)
    assert np.allclose(cost_cooled(scores, cost, alpha=0.5)[1], 0.5)


# ---------------------------------------------------------------------------- surrogates


def test_perf_surrogate_cold_start_is_uniform_average():
    features = np.array([[1.0, 3.0], [2.0, 2.0], [-1.0, 1.0]])
    surrogate = LinearPerfSurrogate(features)
    mu, sigma = surrogate.predict()
    assert np.allclose(mu, features.mean(axis=1))
    assert np.allclose(sigma, features.std(axis=1))
    assert np.allclose(surrogate.weights, [0.5, 0.5])
    surrogate.observe(0, 2.0)
    assert not surrogate.is_fitted  # one observation is not enough
    assert np.allclose(surrogate.predict()[0], features.mean(axis=1))


def test_perf_surrogate_learns_task_weights_without_intercept():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 6))
    true_w = np.array([0.6, 0.0, 0.3, 0.0, 0.1, 0.0])
    target = features @ true_w
    surrogate = LinearPerfSurrogate(features, fit_intercept=False)
    for i in range(30):
        surrogate.observe(i, target[i] + rng.normal(scale=0.01))
    assert surrogate.is_fitted
    assert np.allclose(surrogate.weights, true_w, atol=0.05)
    mu, sigma = surrogate.predict()
    assert np.allclose(mu, target, atol=0.1)
    assert (sigma > 0).all()


def test_cost_surrogate_cold_start_geometric_mean_and_scale():
    cost = np.array([[1.0, 100.0], [10.0, np.nan]])
    surrogate = LinearCostSurrogate(cost, log_space=True, scale=0.5)
    est = surrogate.predict()
    assert est[0] == pytest.approx(10.0 * 0.5)  # geometric mean of (1, 100) times scale
    # NaN cell imputed with the per-task median of the column (100) -> geometric mean of (10, 100)
    assert est[1] == pytest.approx(np.sqrt(10 * 100) * 0.5)
    arithmetic = LinearCostSurrogate(cost, log_space=False, scale=1.0).predict()
    assert arithmetic[0] == pytest.approx(50.5)


# ---------------------------------------------------------------------------- optimiser


def _synthetic_problem(seed: int = 1, n: int = 300, n_tasks: int = 20):
    rng = np.random.default_rng(seed)
    quality = rng.normal(0.0, 0.2, n)
    scale = rng.uniform(0.2, 1.0, n_tasks)
    features = -(scale[None, :] * (1.0 + quality[:, None])) + rng.normal(0, 0.05, (n, n_tasks))
    truth = features.mean(axis=1) + rng.normal(0, 0.02, n)
    cost = np.exp(rng.normal(3, 1, (n, 1)) + rng.normal(0, 0.3, (n, n_tasks)))
    return features, truth, cost


def test_optimizer_respects_budget_and_improves():
    features, truth, cost = _synthetic_problem()
    names = [f"c{i}" for i in range(len(truth))]
    opt = Optimizer(names, features, cost, seed=0)
    first = opt.suggest(remaining_seconds=1e9)
    assert first.f_best == DUMMY_LEVEL and first.pred_cost > 0
    budget, spent, steps = 2000.0, 0.0, 0
    while (s := opt.suggest(remaining_seconds=budget - spent)) is not None:
        assert s.pred_cost <= budget - spent
        actual = float(np.exp(np.log(cost[s.index]).mean()))
        opt.observe(s.index, truth[s.index], actual)
        spent += actual
        steps += 1
    assert steps > 5
    assert opt.n_evaluated == steps
    # The incumbent is among the top 2% of all candidates.
    rank = int((truth > opt.f_best).sum())
    assert rank <= len(truth) * 0.02
    assert opt.perf.is_fitted and opt.cost.is_fitted
    table = opt.predict()
    assert table["evaluated"].sum() == steps
    assert opt.suggest(remaining_seconds=None) is not None  # unlimited budget still has candidates


def test_optimizer_never_repeats_and_handles_no_cost():
    features, truth, _ = _synthetic_problem(n=10, n_tasks=5)
    opt = Optimizer([str(i) for i in range(10)], features, None, seed=3)
    seen = set()
    while (s := opt.suggest()) is not None:
        assert s.index not in seen
        seen.add(s.index)
        opt.observe(s.index, truth[s.index], None)
    assert len(seen) == 10


# ---------------------------------------------------------------------------- evaluation helpers


def test_import_arenaml_warms_gpu_available():
    # `import arenaml` (already done at module load, above) must have made this a cache hit:
    # a caller's first ArenaSearch.fit() should never pay gpu_available()'s own first-call
    # cost (torch's import plus its first torch.cuda.is_available()), several seconds on a
    # cold process -- see the module docstring of arenaml.applicability.warm_up.
    info = gpu_available.cache_info()
    assert info.currsize == 1  # populated exactly once, regardless of hits below
    gpu_available()
    gpu_available()
    assert gpu_available.cache_info().hits >= info.hits + 2


def test_cv_strategy_validation_and_counts():
    kfold = CVStrategy("kfold", n_folds=8)
    assert kfold.n_fits == 8 and kfold.n_train_per_fit(800) == 700
    holdout = CVStrategy("holdout", holdout_frac=0.25)
    assert holdout.n_fits == 1 and holdout.n_train_per_fit(100) == 75
    with pytest.raises(ValueError):
        CVStrategy("kfold", n_folds=1)
    with pytest.raises(ValueError):
        CVStrategy("loo")


def test_prepare_hyperparameters_replaces_tabarena_time_limit():
    hp = {"learning_rate": 0.1, "ag_args_ensemble": {"ag.max_time_limit": 3600, "refit_folds": True}}
    bagged = prepare_hyperparameters(hp, CVStrategy("kfold"), time_limit=120, num_gpus=0)
    assert bagged["ag_args_ensemble"] == {"refit_folds": True, "ag.max_time_limit": 120.0}
    assert bagged["ag_args_fit"] == {"num_gpus": 0}
    assert hp["ag_args_ensemble"]["ag.max_time_limit"] == 3600  # input untouched
    holdout = prepare_hyperparameters(hp, CVStrategy("holdout"), time_limit=60)
    assert holdout["ag.max_time_limit"] == 60.0
    assert "ag.max_time_limit" not in holdout["ag_args_ensemble"]
    unlimited = prepare_hyperparameters({"ag_args_ensemble": {"ag.max_time_limit": 3600}}, CVStrategy(), None)
    assert "ag_args_ensemble" not in unlimited
