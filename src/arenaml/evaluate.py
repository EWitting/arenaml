"""Run one TabArena leaderboard configuration on a dataset, the way TabArena ran it.

TabArena evaluates every configuration through AutoGluon's ``TabularPredictor`` with a
single model, no weighted ensemble, 8-fold bagging and the configuration's own
``ag_args_ensemble`` / ``ag_args_fit`` settings (fold fitting strategy, refit on the full data
for foundation models, GPU allocation, ...).  :func:`run_config` reproduces that call on the
user's data with a user-chosen validation protocol and returns the validation metric error,
the wall-clock cost and, optionally, the fitted predictor.
"""

from __future__ import annotations

import copy
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LABEL = "__label__"
TABARENA_BAG_FOLDS = 8


@dataclass(frozen=True)
class CVStrategy:
    """How each candidate is validated on the target dataset.

    ``kfold`` fits ``n_folds * n_repeats`` bagged models and scores the out-of-fold predictions
    (TabArena's protocol with ``n_folds=8``).  ``holdout`` fits one model on
    ``1 - holdout_frac`` of the rows and scores the rest; it is cheaper but noisier.
    """

    kind: Literal["kfold", "holdout"] = "kfold"
    n_folds: int = 8
    n_repeats: int = 1
    holdout_frac: float = 0.2

    def __post_init__(self) -> None:
        if self.kind not in ("kfold", "holdout"):
            raise ValueError("kind must be 'kfold' or 'holdout'")
        if self.kind == "kfold" and self.n_folds < 2:
            raise ValueError("n_folds must be >= 2 for kfold")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        if not 0.0 < self.holdout_frac < 1.0:
            raise ValueError("holdout_frac must be in (0, 1)")

    @property
    def n_fits(self) -> int:
        """Number of model fits per evaluation."""
        return self.n_folds * self.n_repeats if self.kind == "kfold" else 1

    def n_train_per_fit(self, n_samples: int) -> int:
        if self.kind == "kfold":
            return int(n_samples * (self.n_folds - 1) / self.n_folds)
        return int(n_samples * (1.0 - self.holdout_frac))

    @property
    def is_bagged(self) -> bool:
        return self.kind == "kfold"


@dataclass
class EvalResult:
    """Outcome of running one configuration on the target dataset."""

    config: str
    status: str  # "ok" or "failed"
    metric_error_val: float = float("nan")
    seconds: float = float("nan")
    fit_seconds: float = float("nan")
    infer_seconds: float = float("nan")
    time_limit: float | None = None
    predictor: Any = None
    path: str | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and np.isfinite(self.metric_error_val)

    def cleanup(self) -> None:
        """Drop the predictor and delete its files from disk."""
        self.predictor = None
        if self.path:
            shutil.rmtree(self.path, ignore_errors=True)


def prepare_hyperparameters(
    hyperparameters: dict,
    cv: CVStrategy,
    time_limit: float | None,
    num_gpus: int | None = None,
    num_cpus: int | None = None,
) -> dict:
    """Copy a leaderboard hyperparameter dict and inject the time limit and resources.

    The leaderboard entries carry ``ag_args_ensemble["ag.max_time_limit"] = 3600`` (TabArena's
    per-fit limit).  It is replaced by ``time_limit``, placed where TabArena's experiment
    constructor puts it: inside ``ag_args_ensemble`` for a bagged fit, top-level otherwise.
    """
    hp = copy.deepcopy(hyperparameters or {})
    ens = dict(hp.get("ag_args_ensemble", {}) or {})
    ens.pop("ag.max_time_limit", None)
    hp.pop("ag.max_time_limit", None)
    if time_limit is not None:
        if cv.is_bagged:
            ens["ag.max_time_limit"] = float(time_limit)
        else:
            hp["ag.max_time_limit"] = float(time_limit)
    if ens:
        hp["ag_args_ensemble"] = ens
    else:
        hp.pop("ag_args_ensemble", None)
    if num_gpus is not None or num_cpus is not None:
        fit_args = dict(hp.get("ag_args_fit", {}) or {})
        if num_gpus is not None:
            fit_args["num_gpus"] = num_gpus
        if num_cpus is not None:
            fit_args["num_cpus"] = num_cpus
        hp["ag_args_fit"] = fit_args
    return hp


def _fit_kwargs(cv: CVStrategy, num_cpus: int | None, num_gpus: int | None) -> dict:
    kwargs: dict[str, Any] = {
        "fit_weighted_ensemble": False,
        "calibrate": False,
        "raise_on_model_failure": True,
    }
    if cv.is_bagged:
        kwargs["num_bag_folds"] = cv.n_folds
        kwargs["num_bag_sets"] = cv.n_repeats
    else:
        kwargs["num_bag_folds"] = 0
        kwargs["holdout_frac"] = cv.holdout_frac
    if num_cpus is not None:
        kwargs["num_cpus"] = num_cpus
    if num_gpus is not None:
        kwargs["num_gpus"] = num_gpus
    return kwargs


def run_config(
    config: str,
    model_cls: Any,
    hyperparameters: dict,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    eval_metric: str,
    cv: CVStrategy | None = None,
    time_limit: float | None = None,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    output_dir: str | Path | None = None,
    keep_predictor: bool = True,
    verbosity: int = 0,
) -> EvalResult:
    """Fit one leaderboard configuration on ``(X, y)`` and score it out-of-fold.

    Args:
        config: name of the configuration (used for the output folder and logging).
        model_cls: AutoGluon model class, or its name as stored in the TabArena artifacts.
        hyperparameters: the configuration's hyperparameters as stored in the artifacts.
        X, y: features and labels; the index is ignored.
        problem_type: ``binary``, ``multiclass`` or ``regression``.
        eval_metric: AutoGluon metric name; the returned error is ``optimum - score``.
        cv: validation protocol.
        time_limit: seconds for the (bagged) fit, ``None`` for no limit.
        num_cpus, num_gpus: resources passed to AutoGluon (``None`` = AutoGluon's default).
        output_dir: parent folder for the predictor files (a temporary folder if ``None``).
        keep_predictor: keep the fitted predictor and its files; otherwise they are deleted.
        verbosity: AutoGluon verbosity.
    """
    from autogluon.tabular import TabularPredictor

    from arenaml.applicability import resolve_model_cls

    cv = cv or CVStrategy()
    if isinstance(model_cls, str):
        model_cls = resolve_model_cls(model_cls)
    hp = prepare_hyperparameters(hyperparameters, cv, time_limit, num_gpus=num_gpus, num_cpus=num_cpus)

    X = pd.DataFrame(X).reset_index(drop=True)
    if LABEL in X.columns:
        raise ValueError(f"X must not contain a column named {LABEL!r}")
    train = X.copy()
    train[LABEL] = pd.Series(y).reset_index(drop=True).values

    root = Path(output_dir) if output_dir is not None else Path.cwd() / "arenaml_runs"
    path = root / f"{config}_{uuid.uuid4().hex[:8]}"
    result = EvalResult(config=config, status="failed", time_limit=time_limit, path=str(path))

    start = time.perf_counter()
    try:
        predictor = TabularPredictor(
            label=LABEL,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=str(path),
            verbosity=verbosity,
        )
        predictor.fit(
            train_data=train, hyperparameters={model_cls: hp}, **_fit_kwargs(cv, num_cpus, num_gpus)
        )
        fit_seconds = time.perf_counter() - start
        board = predictor.leaderboard(score_format="error", set_refit_score_to_parent=True).set_index("model")
        best = predictor.model_best
        row = board.loc[best]
        result.metric_error_val = float(row["metric_error_val"])
        result.fit_seconds = fit_seconds
        result.infer_seconds = float(row.get("pred_time_val", np.nan))
        result.seconds = fit_seconds + (result.infer_seconds if np.isfinite(result.infer_seconds) else 0.0)
        result.status = "ok"
        result.extra = {"model_name": best, "ag_fit_time": float(row.get("fit_time", np.nan))}
        if keep_predictor:
            result.predictor = predictor
        else:
            result.cleanup()
    except Exception as exc:  # AutoGluon raises many different types
        result.seconds = time.perf_counter() - start
        result.error = f"{type(exc).__name__}: {exc}"
        log.warning("%s failed after %.1fs: %s", config, result.seconds, result.error)
        result.cleanup()
    return result
