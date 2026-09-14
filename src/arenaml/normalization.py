"""Metric handling and dummy-score normalisation.

TabArena reports every result as a *metric error* that is lower-is-better for all task types:
``1 - roc_auc`` for binary classification, ``log_loss`` for multiclass classification and
``rmse`` for regression.  The leaderboard rescales these per task using the best and worst
score (``loss_rescaled``), which leaks the optimum and is therefore only used for reporting.

During the search we instead follow the thesis protocol: every error is divided by the error
of a constant "dummy" predictor for the same task (majority class / class prior / mean) and
negated, so that

* every value is higher-is-better,
* ``-1.0`` means "as good as the dummy predictor" on every task and every task type,
* ``0.0`` is a perfect score,
* nothing about the achievable optimum on the target task is used.

The dummy error of the target dataset is computed here from the labels alone, using the same
AutoGluon scorer that TabArena uses to evaluate models, so the target observations land on the
same scale as the benchmark columns.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

#: Metric used by TabArena for each problem type (``metric_error`` column of the artifacts).
METRIC_BY_PROBLEM_TYPE = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}

#: Normalised performance of the dummy predictor: ``-(dummy_error / dummy_error)``.
DUMMY_LEVEL = -1.0

PROBLEM_TYPES = ("binary", "multiclass", "regression")


def default_metric(problem_type: str) -> str:
    """Return TabArena's evaluation metric for ``problem_type``."""
    try:
        return METRIC_BY_PROBLEM_TYPE[problem_type]
    except KeyError as exc:
        raise ValueError(f"Unknown problem_type {problem_type!r}; expected one of {PROBLEM_TYPES}") from exc


def normalize_error(error, dummy_error):
    """Map metric errors to higher-is-better performance relative to the dummy predictor.

    Works element-wise on scalars, arrays, Series and DataFrames.  For a DataFrame with tasks as
    columns, ``dummy_error`` must be a Series indexed by task.  NaN stays NaN.
    """
    return -(error / dummy_error)


def denormalize_performance(performance, dummy_error):
    """Inverse of :func:`normalize_error`."""
    return -performance * dummy_error


def infer_problem_type(y: pd.Series | np.ndarray | Iterable) -> str:
    """Infer ``binary``/``multiclass``/``regression`` from the labels.

    Uses AutoGluon's inference when available (identical to what ``TabularPredictor`` would do)
    and falls back to a simple rule otherwise.
    """
    y = pd.Series(y)
    try:
        from autogluon.core.utils.utils import infer_problem_type as _ag_infer

        return _ag_infer(y, silent=True)
    except Exception:  # pragma: no cover - only hit without AutoGluon
        n_unique = y.nunique(dropna=True)
        if n_unique == 2:
            return "binary"
        if pd.api.types.is_numeric_dtype(y) and n_unique > 10:
            return "regression"
        return "multiclass"


def _encode_labels(y: pd.Series, problem_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (integer codes, class values) for classification, or (float values, empty) otherwise."""
    y = pd.Series(y).reset_index(drop=True)
    if problem_type == "regression":
        return y.astype(float).to_numpy(), np.array([])
    codes, classes = pd.factorize(y, sort=True)
    if (codes < 0).any():
        raise ValueError("Labels contain missing values; drop or impute them before searching.")
    return codes.astype(int), np.asarray(classes)


def dummy_error(y, problem_type: str, metric: str | None = None) -> float:
    """Error of a constant predictor on labels ``y`` under the TabArena scorer for ``metric``.

    The constant predictor predicts the class prior (classification) or the mean (regression),
    matching TabArena's ``Dummy`` baseline.  For ``roc_auc`` this is always ``0.5``, for
    ``log_loss`` it is the entropy of the class prior and for ``rmse`` the population standard
    deviation of the target.

    Any AutoGluon metric name is accepted; the scorer's ``error`` (``optimum - score``) is
    returned so the value is on the same scale as TabArena's ``metric_error`` column.
    """
    metric = metric or default_metric(problem_type)
    y_true, classes = _encode_labels(y, problem_type)
    n = len(y_true)
    if n == 0:
        raise ValueError("y is empty")

    from autogluon.core.metrics import get_metric

    scorer = get_metric(metric, problem_type=problem_type)

    if problem_type == "regression":
        y_pred = np.full(n, y_true.mean())
        return float(scorer.error(y_true, y_pred))

    n_classes = len(classes)
    prior = np.bincount(y_true, minlength=n_classes).astype(float) / n
    y_pred_label = np.full(n, int(np.argmax(prior)))
    proba = np.tile(prior, (n, 1))
    if scorer.needs_pred:
        return float(scorer.error(y_true, y_pred_label))
    if problem_type == "binary":
        return float(scorer.error(y_true, proba[:, 1]))
    return float(scorer.error(y_true, proba))


def loss_rescaled(error: pd.Series) -> pd.Series:
    """TabArena's leaderboard rescaling: 0 for the best and 1 for the worst error on a task.

    Only meant for reporting; it uses the optimum and must not feed the search.
    """
    best, worst = error.min(), error.max()
    if not np.isfinite(best) or worst == best:
        return pd.Series(0.0, index=error.index)
    return (error - best) / (worst - best)
