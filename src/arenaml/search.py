"""User-facing search: pick the best TabArena configuration for a dataset within a budget."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from arenaml.applicability import TargetProfile, applicability, resolve_model_cls
from arenaml.evaluate import TABARENA_BAG_FOLDS, CVStrategy, EvalResult, run_config
from arenaml.leaderboard import Leaderboard, load_leaderboard
from arenaml.normalization import (
    DUMMY_LEVEL,
    default_metric,
    dummy_error,
    infer_problem_type,
    normalize_error,
)
from arenaml.optimizer import Optimizer

log = logging.getLogger(__name__)

_HISTORY_COLUMNS = [
    "step",
    "config",
    "method",
    "status",
    "metric_error_val",
    "normalized_perf",
    "seconds",
    "time_limit",
    "pred_mean",
    "pred_std",
    "pred_cost",
    "acquisition",
    "f_best_before",
    "best_error_so_far",
    "elapsed",
    "is_best",
]


class ArenaSearch:
    """Cost-aware Bayesian optimisation over TabArena leaderboard configurations.

    Args:
        time_budget: total wall-clock seconds for the search (model fits included).
        max_evals: optional cap on the number of configurations evaluated.
        models: ``"all"`` (default), ``"cpu"``, ``"gpu"``, or a list of TabArena method names
            such as ``["LightGBM", "CatBoost", "TabPFNv2_GPU"]``.
        cv: validation protocol used for every candidate (default: 8-fold bagging like TabArena).
        eval_metric: AutoGluon metric to optimise; defaults to TabArena's metric for the problem
            type (``roc_auc``, ``log_loss``, ``rmse``).
        cost_alpha: exponent of the runtime penalty in the acquisition (``1``: EI per second,
            ``0``: plain EI).
        xi: exploration bonus of expected improvement.
        source_tasks: ``"all"`` uses every TabArena task as a feature, ``"same_type"`` only the
            tasks with the target's problem type.
        include_gpu_models: force GPU-benchmarked models in or out; by default they are used
            only when CUDA is available.
        max_eval_time: cap on the time limit given to a single fit (defaults to the remaining
            budget).
        min_eval_time: stop searching when less than this many seconds remain.
        budget_margin: a candidate is eligible only if its predicted runtime is at most this
            fraction of the remaining budget.
        penalize_failures: score crashed or timed-out fits at the dummy level so the surrogate
            learns to avoid them; otherwise they are just skipped.
        refit_full: after the search, refit the best configuration on all rows.
        num_cpus, num_gpus: resources for AutoGluon (``None`` = AutoGluon's default).
        output_dir: folder for AutoGluon predictor files (default ``./arenaml_runs``).
        leaderboard: a pre-loaded :class:`Leaderboard`; otherwise loaded (and cached) for
            ``models``.
        score_column: ``"test"`` (leaderboard number) or ``"val"`` benchmark scores.
        cache_dir: cache folder for the aggregated leaderboard.
        seed: random seed for tie-breaking.
        verbosity: 0 silent, 1 progress log lines, 2 also AutoGluon output.
    """

    def __init__(
        self,
        time_budget: float = 3600.0,
        max_evals: int | None = None,
        models: str | Iterable[str] = "all",
        cv: CVStrategy | None = None,
        eval_metric: str | None = None,
        cost_alpha: float = 1.0,
        xi: float = 0.0,
        source_tasks: Literal["all", "same_type"] = "all",
        include_gpu_models: bool | None = None,
        max_eval_time: float | None = None,
        min_eval_time: float = 5.0,
        budget_margin: float = 1.0,
        penalize_failures: bool = True,
        refit_full: bool = False,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        output_dir: str | Path | None = None,
        leaderboard: Leaderboard | None = None,
        score_column: Literal["test", "val"] = "test",
        cache_dir: str | Path | None = None,
        seed: int | None = 0,
        verbosity: int = 1,
    ):
        if time_budget <= 0:
            raise ValueError("time_budget must be positive")
        if source_tasks not in ("all", "same_type"):
            raise ValueError("source_tasks must be 'all' or 'same_type'")
        self.time_budget = float(time_budget)
        self.max_evals = max_evals
        self.models = models if isinstance(models, str) else list(models)
        self.cv = cv or CVStrategy()
        self.eval_metric = eval_metric
        self.cost_alpha = cost_alpha
        self.xi = xi
        self.source_tasks = source_tasks
        self.include_gpu_models = include_gpu_models
        self.max_eval_time = max_eval_time
        self.min_eval_time = min_eval_time
        self.budget_margin = budget_margin
        self.penalize_failures = penalize_failures
        self.refit_full = refit_full
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.output_dir = output_dir
        self.leaderboard = leaderboard
        self.score_column = score_column
        self.cache_dir = cache_dir
        self.seed = seed
        self.verbosity = verbosity

        # Fitted state
        self.problem_type_: str | None = None
        self.eval_metric_: str | None = None
        self.dummy_error_: float | None = None
        self.target_: TargetProfile | None = None
        self.leaderboard_: Leaderboard | None = None
        self.applicability_: pd.DataFrame | None = None
        self.candidates_: list[str] = []
        self.optimizer_: Optimizer | None = None
        self.history_: pd.DataFrame = pd.DataFrame(columns=_HISTORY_COLUMNS)
        self.best_config_: str | None = None
        self.best_error_: float = float("nan")
        self.best_result_: EvalResult | None = None
        self.elapsed_: float = 0.0

    # ------------------------------------------------------------------ setup helpers
    def _log(self, msg: str, *args) -> None:
        if self.verbosity >= 1:
            log.info(msg, *args)

    def _load_leaderboard(self) -> Leaderboard:
        if self.leaderboard is not None:
            return self.leaderboard
        methods = "all" if isinstance(self.models, str) else self.models
        return load_leaderboard(methods=methods, score_column=self.score_column, cache_dir=self.cache_dir)

    def _profile(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> TargetProfile:
        n_classes = int(pd.Series(y).nunique()) if problem_type != "regression" else 0
        return TargetProfile(
            problem_type=problem_type,
            n_samples=len(X),
            n_features=int(X.shape[1]),
            n_classes=n_classes,
            n_train_per_fold=self.cv.n_train_per_fit(len(X)),
        )

    def _build_optimizer(self, lb: Leaderboard, candidates: list[str]) -> Optimizer:
        tasks = lb.task_names
        if self.source_tasks == "same_type":
            tasks = [t for t in tasks if lb.tasks.loc[t, "problem_type"] == self.problem_type_]
        perf = lb.normalized_performance().loc[candidates, tasks].fillna(DUMMY_LEVEL).to_numpy()
        cost = lb.cost.loc[candidates, tasks].to_numpy()
        return Optimizer(
            names=candidates,
            perf_features=perf,
            cost_matrix=cost,
            cost_alpha=self.cost_alpha,
            xi=self.xi,
            cost_scale=self.cv.n_fits / TABARENA_BAG_FOLDS,
            budget_margin=self.budget_margin,
            seed=self.seed,
        )

    # ------------------------------------------------------------------ main entry point
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray, problem_type: str | None = None) -> ArenaSearch:
        """Search for the best configuration on ``(X, y)`` and fit it."""
        start = time.perf_counter()
        deadline = start + self.time_budget
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        if len(X) != len(y):
            raise ValueError("X and y have different lengths")
        if y.isna().any():
            raise ValueError("y contains missing values")

        self.problem_type_ = problem_type or infer_problem_type(y)
        self.eval_metric_ = self.eval_metric or default_metric(self.problem_type_)
        self.dummy_error_ = dummy_error(y, self.problem_type_, self.eval_metric_)
        self.target_ = self._profile(X, y, self.problem_type_)
        self._log(
            "Target: %s, %d rows, %d features, metric %s, dummy error %.4f",
            self.problem_type_,
            len(X),
            X.shape[1],
            self.eval_metric_,
            self.dummy_error_,
        )

        lb = self._load_leaderboard()
        self.leaderboard_ = lb
        self.applicability_ = applicability(
            lb, self.target_, methods=self.models, include_gpu_models=self.include_gpu_models
        )
        self.candidates_ = list(self.applicability_.index[self.applicability_["applicable"]])
        if not self.candidates_:
            raise RuntimeError(
                "No applicable configurations. Reasons:\n"
                + self.applicability_["reason"].value_counts().to_string()
            )
        excluded = self.applicability_.loc[~self.applicability_["applicable"], "reason"].value_counts()
        self._log(
            "%d candidate configurations from %d methods (%d excluded: %s)",
            len(self.candidates_),
            self.applicability_.loc[self.candidates_, "method"].nunique(),
            int(excluded.sum()),
            ", ".join(f"{k}={v}" for k, v in excluded.items()) or "none",
        )

        self.optimizer_ = self._build_optimizer(lb, self.candidates_)
        rows: list[dict] = []
        self.best_config_, self.best_error_, self.best_result_ = None, float("nan"), None
        step = 0
        while True:
            if self.max_evals is not None and step >= self.max_evals:
                self._log("Stopping: reached max_evals=%d", self.max_evals)
                break
            remaining = deadline - time.perf_counter()
            if remaining < self.min_eval_time:
                self._log("Stopping: %.0fs left", max(remaining, 0.0))
                break
            suggestion = self.optimizer_.suggest(remaining_seconds=remaining)
            if suggestion is None:
                self._log("Stopping: no remaining candidate fits in the %.0fs left", remaining)
                break
            time_limit = remaining if self.max_eval_time is None else min(remaining, self.max_eval_time)
            name = suggestion.name
            spec = lb.configs.loc[name]
            self._log(
                "[%d] %s (pred score %.3f +- %.3f, pred %.0fs, %.0fs left)",
                step + 1,
                name,
                suggestion.pred_mean,
                suggestion.pred_std,
                suggestion.pred_cost,
                remaining,
            )
            result = run_config(
                config=name,
                model_cls=resolve_model_cls(
                    str(spec["model_cls"]), str(spec.get("config_ag_key") or spec["ag_key"])
                ),
                hyperparameters=spec["hyperparameters"],
                X=X,
                y=y,
                problem_type=self.problem_type_,
                eval_metric=self.eval_metric_,
                cv=self.cv,
                time_limit=time_limit,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
                output_dir=self.output_dir,
                keep_predictor=True,
                verbosity=max(self.verbosity - 1, 0) * 2,
            )
            step += 1
            if result.ok:
                perf = float(normalize_error(result.metric_error_val, self.dummy_error_))
                self.optimizer_.observe(suggestion.index, perf, result.seconds)
            else:
                perf = DUMMY_LEVEL
                if self.penalize_failures:
                    self.optimizer_.observe(suggestion.index, DUMMY_LEVEL, result.seconds)
                else:
                    self.optimizer_.evaluated[suggestion.index] = np.nan
            is_best = result.ok and (self.best_result_ is None or result.metric_error_val < self.best_error_)
            if is_best:
                if self.best_result_ is not None:
                    self.best_result_.cleanup()
                self.best_config_, self.best_error_, self.best_result_ = name, result.metric_error_val, result
            else:
                result.cleanup()
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "step": step,
                    "config": name,
                    "method": spec["method"],
                    "status": result.status,
                    "metric_error_val": result.metric_error_val,
                    "normalized_perf": perf,
                    "seconds": result.seconds,
                    "time_limit": time_limit,
                    "pred_mean": suggestion.pred_mean,
                    "pred_std": suggestion.pred_std,
                    "pred_cost": suggestion.pred_cost,
                    "acquisition": suggestion.acquisition,
                    "f_best_before": suggestion.f_best,
                    "best_error_so_far": self.best_error_,
                    "elapsed": elapsed,
                    "is_best": is_best,
                }
            )
            self.history_ = pd.DataFrame(rows, columns=_HISTORY_COLUMNS)
            if result.ok:
                self._log(
                    "    %s = %.4f in %.0fs%s",
                    self.eval_metric_ + " error",
                    result.metric_error_val,
                    result.seconds,
                    "  (new best)" if is_best else "",
                )
            else:
                self._log("    failed in %.0fs: %s", result.seconds, result.error)

        self.history_ = pd.DataFrame(rows, columns=_HISTORY_COLUMNS)
        if self.best_result_ is None:
            raise RuntimeError("No configuration finished successfully within the budget")
        if self.refit_full and self.best_result_.predictor is not None:
            self._log("Refitting %s on all rows", self.best_config_)
            self.best_result_.predictor.refit_full()
        self.elapsed_ = time.perf_counter() - start
        self._log(
            "Done: best %s with %s error %.4f after %d evaluations in %.0fs",
            self.best_config_,
            self.eval_metric_,
            self.best_error_,
            step,
            self.elapsed_,
        )
        return self

    # ------------------------------------------------------------------ results
    @property
    def best_predictor_(self):
        """The AutoGluon ``TabularPredictor`` fitted with the best configuration."""
        if self.best_result_ is None or self.best_result_.predictor is None:
            raise RuntimeError("No fitted predictor; call fit() first")
        return self.best_result_.predictor

    @property
    def best_hyperparameters_(self) -> dict:
        return dict(self.leaderboard_.configs.loc[self.best_config_, "hyperparameters"])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.best_predictor_.predict(pd.DataFrame(X))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.best_predictor_.predict_proba(pd.DataFrame(X))

    def predictions(self) -> pd.DataFrame:
        """Current surrogate predictions (normalised score and runtime) for every candidate."""
        if self.optimizer_ is None:
            raise RuntimeError("call fit() first")
        table = self.optimizer_.predict()
        table["pred_error"] = -table["pred_mean"] * self.dummy_error_
        table["method"] = self.leaderboard_.configs.loc[table.index, "method"].values
        return table

    def task_weights(self) -> pd.DataFrame:
        """Learned weight per TabArena task for the performance and runtime surrogates."""
        if self.optimizer_ is None:
            raise RuntimeError("call fit() first")
        tasks = self.leaderboard_.task_names
        if self.source_tasks == "same_type":
            tasks = [t for t in tasks if self.leaderboard_.tasks.loc[t, "problem_type"] == self.problem_type_]
        data: dict[str, Any] = {"performance": self.optimizer_.perf.weights}
        if self.optimizer_.cost is not None:
            data["runtime"] = self.optimizer_.cost.weights
        return pd.DataFrame(data, index=pd.Index(tasks, name="task"))

    def summary(self) -> str:
        lines = [
            f"ArenaSearch: {len(self.history_)} evaluations in {self.elapsed_:.0f}s "
            f"({self.problem_type_}, metric {self.eval_metric_})",
            f"best: {self.best_config_} with {self.eval_metric_} error {self.best_error_:.5f}",
        ]
        if len(self.history_):
            cols = ["step", "config", "status", "metric_error_val", "seconds", "pred_cost"]
            lines.append(self.history_[cols].to_string(index=False))
        return "\n".join(lines)


def search(
    X: pd.DataFrame, y: pd.Series | np.ndarray, problem_type: str | None = None, **kwargs
) -> ArenaSearch:
    """One-liner: ``arenaml.search(X, y, time_budget=600).predict(X_new)``.

    Keyword arguments are passed to :class:`ArenaSearch`.
    """
    return ArenaSearch(**kwargs).fit(X, y, problem_type=problem_type)
