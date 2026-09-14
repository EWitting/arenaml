"""Build the configuration-by-task matrices from the TabArena artifacts.

For every benchmarked method TabArena publishes a small results table with one row per
(dataset, fold, configuration) holding the metric error, the validation metric error and the
train / inference times, plus a JSON file with the exact AutoGluon hyperparameters of every
configuration.  This module aggregates those tables into

* ``error``: ``(n_configs, n_tasks)`` metric error, averaged over folds, NaN where the
  configuration was not (genuinely) evaluated on the task,
* ``cost``: ``(n_configs, n_tasks)`` seconds of training plus inference, same layout,
* ``configs``: one row per configuration with the model class and hyperparameters needed to
  run it again,
* ``tasks``: one row per benchmark task with its problem type, size and dummy error.

Rows that TabArena itself imputed (``imputed == True``, e.g. a classification-only model on
a regression task, filled with RandomForest's score by the leaderboard) are treated as missing
here, because they carry no information about the configuration and would otherwise make an
inapplicable model look competitive.

The result is cached as a pickle under ``~/.cache/arenaml`` (override with ``ARENAML_CACHE``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from arenaml.normalization import normalize_error

log = logging.getLogger(__name__)

DUMMY_METHOD = "Dummy"
ScoreColumn = Literal["test", "val"]

#: Methods of these TabArena types are ensembles or full AutoML systems, not single configurations.
_EXCLUDED_METHOD_TYPES = {"baseline", "portfolio"}


def default_cache_dir() -> Path:
    root = os.environ.get("ARENAML_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "arenaml"


@dataclass
class Leaderboard:
    """Aggregated TabArena results with one row per configuration and one column per task."""

    error: pd.DataFrame
    cost: pd.DataFrame
    configs: pd.DataFrame
    tasks: pd.DataFrame
    score_column: str = "test"

    # ------------------------------------------------------------------ basic accessors
    @property
    def config_names(self) -> list[str]:
        return list(self.error.index)

    @property
    def task_names(self) -> list[str]:
        return list(self.error.columns)

    @property
    def dummy_error(self) -> pd.Series:
        return self.tasks["dummy_error"]

    @property
    def methods(self) -> list[str]:
        return sorted(self.configs["method"].unique())

    def normalized_performance(self) -> pd.DataFrame:
        """Higher-is-better performance relative to the dummy predictor; NaN where not evaluated."""
        return normalize_error(self.error, self.dummy_error)

    def evaluated(self) -> pd.DataFrame:
        """Boolean mask of genuinely evaluated (config, task) cells."""
        return self.error.notna()

    def method_support(self) -> pd.DataFrame:
        """Fraction of tasks of each problem type on which each method has genuine results."""
        mask = self.evaluated()
        by_method = mask.groupby(self.configs["method"]).mean()
        return by_method.T.groupby(self.tasks["problem_type"]).mean().T

    def subset(self, configs: Iterable[str] | None = None, tasks: Iterable[str] | None = None) -> Leaderboard:
        configs = list(configs) if configs is not None else self.config_names
        tasks = list(tasks) if tasks is not None else self.task_names
        return Leaderboard(
            error=self.error.loc[configs, tasks],
            cost=self.cost.loc[configs, tasks],
            configs=self.configs.loc[configs],
            tasks=self.tasks.loc[tasks],
            score_column=self.score_column,
        )

    def __repr__(self) -> str:
        n_c, n_t = self.error.shape
        return f"Leaderboard(configs={n_c}, tasks={n_t}, methods={len(self.methods)}, score={self.score_column!r})"


# ---------------------------------------------------------------------- loading from TabArena


def list_config_methods(require_results: bool = True) -> list[str]:
    """Names of all TabArena methods that are single configurations (not portfolios/baselines)."""
    from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection

    names = []
    for md in tabarena_method_metadata_collection.method_metadata_lst:
        if md.method == DUMMY_METHOD or md.method_type in _EXCLUDED_METHOD_TYPES:
            continue
        if require_results and not md.has_results:
            continue
        names.append(md.method)
    return names


def _results_table(method_metadata) -> pd.DataFrame | None:
    """Load the per-(dataset, fold, config) results table of a method, downloading if needed."""
    results_file = Path(method_metadata.path_results) / "model_results.parquet"
    processed_file = Path(method_metadata.path_processed) / "configs.parquet"
    if results_file.exists():
        return pd.read_parquet(results_file)
    if processed_file.exists():
        return pd.read_parquet(processed_file)
    if method_metadata.has_results:
        try:
            method_metadata.method_downloader().download_results()
        except Exception as exc:  # network / permissions
            log.warning("Could not download results for %s: %s", method_metadata.method, exc)
    if results_file.exists():
        return pd.read_parquet(results_file)
    if method_metadata.has_processed and not processed_file.exists():
        try:
            method_metadata.method_downloader().download_processed()
        except Exception as exc:
            log.warning("Could not download processed data for %s: %s", method_metadata.method, exc)
    if processed_file.exists():
        return pd.read_parquet(processed_file)
    return None


def _aggregate_method(
    df: pd.DataFrame, score_column: ScoreColumn, tasks: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average a method's results over folds -> (error, cost) frames indexed by config."""
    config_col = "framework" if "framework" in df.columns else "method"
    err_col = "metric_error" if score_column == "test" else "metric_error_val"
    if err_col not in df.columns:
        raise KeyError(f"Results table has no {err_col!r} column; columns: {list(df.columns)}")
    df = df[df["dataset"].isin(tasks)]
    if "imputed" in df.columns:
        df = df[~df["imputed"].fillna(False).astype(bool)]
    df = df.replace([np.inf, -np.inf], np.nan)
    time_cols = [c for c in ("time_train_s", "time_infer_s") if c in df.columns]
    cost = df[time_cols].sum(axis=1) if time_cols else pd.Series(np.nan, index=df.index)
    work = pd.DataFrame(
        {
            "config": df[config_col].values,
            "dataset": df["dataset"].values,
            "error": df[err_col].values,
            "cost": cost.values,
        }
    )
    agg = work.groupby(["config", "dataset"]).mean()
    error = agg["error"].unstack("dataset")
    cost = agg["cost"].unstack("dataset")
    return error, cost


def _task_table() -> pd.DataFrame:
    from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

    meta = load_task_metadata(paper=True)
    tasks = pd.DataFrame(
        {
            "tid": meta["tid"].astype(int).values,
            "problem_type": meta["problem_type"].values,
            "n_samples": meta["NumberOfInstances"].astype(int).values,
            "n_train_per_fold": meta["n_samples_train_per_fold"].astype(int).values,
            "n_features": meta["n_features"].astype(int).values,
            "n_classes": meta["n_classes"].fillna(0).astype(int).values,
        },
        index=pd.Index(meta["dataset"].values, name="task"),
    )
    return tasks.sort_index()


def build_leaderboard(
    methods: Iterable[str] | Literal["all"] = "all", score_column: ScoreColumn = "test"
) -> Leaderboard:
    """Aggregate the TabArena artifacts into a :class:`Leaderboard` (no caching)."""
    from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection as collection

    if score_column not in ("test", "val"):
        raise ValueError("score_column must be 'test' or 'val'")
    method_names = list_config_methods() if methods == "all" else list(methods)
    tasks = _task_table()
    task_names = list(tasks.index)

    # Dummy baseline: the reference error per task.
    dummy_md = collection.get_method_metadata(method=DUMMY_METHOD)
    dummy_df = _results_table(dummy_md)
    if dummy_df is None:
        raise RuntimeError("Could not load the TabArena Dummy baseline results")
    dummy_error, _ = _aggregate_method(dummy_df, score_column, task_names)
    dummy_error = dummy_error.mean(axis=0).reindex(task_names)
    if dummy_error.isna().any():
        missing = list(dummy_error.index[dummy_error.isna()])
        raise RuntimeError(f"Dummy baseline missing for tasks {missing}")
    tasks["dummy_error"] = dummy_error.values

    errors, costs, config_rows = [], [], []
    for method in method_names:
        try:
            md = collection.get_method_metadata(method=method)
        except Exception as exc:
            log.warning("Skipping unknown method %s: %s", method, exc)
            continue
        df = _results_table(md)
        if df is None:
            log.warning("Skipping %s: no results available locally or for download", method)
            continue
        try:
            hyperparameters = md.load_configs_hyperparameters(holdout=False, download="auto")
        except Exception as exc:
            log.warning("Skipping %s: hyperparameters unavailable (%s)", method, exc)
            continue
        error, cost = _aggregate_method(df, score_column, task_names)
        keep = [c for c in error.index if c in hyperparameters]
        dropped = len(error.index) - len(keep)
        if dropped:
            log.warning("%s: %d configs without hyperparameters dropped", method, dropped)
        if not keep:
            continue
        error, cost = error.loc[keep], cost.loc[keep]
        for name in keep:
            spec = hyperparameters[name]
            config_rows.append(
                {
                    "config": name,
                    "method": method,
                    "model_cls": spec.get("model_cls"),
                    "ag_key": md.ag_key,
                    "config_ag_key": spec.get("ag_key"),
                    "compute": md.compute,
                    "artifact": md.artifact_name,
                    "hyperparameters": spec.get("hyperparameters", {}),
                }
            )
        errors.append(error)
        costs.append(cost)
        log.info("Loaded %s: %d configs", method, len(keep))

    if not errors:
        raise RuntimeError("No TabArena results could be loaded for the requested methods")
    error = pd.concat(errors).reindex(columns=task_names)
    cost = pd.concat(costs).reindex(columns=task_names)
    configs = pd.DataFrame(config_rows).set_index("config")
    error.index.name = cost.index.name = "config"
    error.columns.name = cost.columns.name = "task"
    return Leaderboard(error=error, cost=cost, configs=configs, tasks=tasks, score_column=score_column)


def _cache_key(methods, score_column: str) -> str:
    method_part = (
        "all" if methods == "all" else hashlib.sha256(repr(sorted(methods)).encode()).hexdigest()[:10]
    )
    return f"leaderboard_v2_{score_column}_{method_part}"


def load_leaderboard(
    methods: Iterable[str] | Literal["all"] = "all",
    score_column: ScoreColumn = "test",
    cache_dir: str | os.PathLike | None = None,
    refresh: bool = False,
) -> Leaderboard:
    """Load (or build and cache) the TabArena leaderboard matrices.

    Args:
        methods: ``"all"`` for every single-configuration method on the leaderboard, or a list of
            TabArena method names such as ``["XGBoost", "LightGBM", "TabPFNv2_GPU"]``.
        score_column: ``"test"`` uses TabArena's reported test ``metric_error`` (the leaderboard
            number); ``"val"`` uses the out-of-fold validation error, which is what a search on a
            new dataset observes.
        cache_dir: where to store the aggregated pickle. Defaults to ``~/.cache/arenaml``.
        refresh: rebuild even if a cached file exists.
    """
    if methods != "all":
        methods = list(methods)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache_dir / f"{_cache_key(methods, score_column)}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as fh:
            lb = pickle.load(fh)
        log.info("Loaded cached leaderboard from %s", path)
        return lb
    lb = build_leaderboard(methods=methods, score_column=score_column)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(lb, fh)
    log.info("Cached leaderboard to %s", path)
    return lb
