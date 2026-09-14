"""Discrete cost-aware Bayesian optimisation over leaderboard configurations.

The optimiser only sees numbers: a feature row per candidate (normalised benchmark scores),
the benchmark runtimes, and the scores and runtimes observed on the target so far.  Running a
candidate on the target dataset is the caller's job (see :mod:`arenaml.evaluate`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arenaml.acquisition import cost_cooled, expected_improvement
from arenaml.normalization import DUMMY_LEVEL
from arenaml.surrogate import LinearCostSurrogate, LinearPerfSurrogate


@dataclass
class Suggestion:
    index: int
    name: str
    pred_mean: float
    pred_std: float
    pred_cost: float
    acquisition: float
    f_best: float


class Optimizer:
    """Expected improvement per predicted second over a fixed candidate set.

    Args:
        names: candidate names, one per row of ``perf_features``.
        perf_features: ``(n_candidates, n_tasks)`` normalised benchmark performance (no NaN).
        cost_matrix: ``(n_candidates, n_tasks)`` benchmark runtimes in seconds (NaN allowed).
        cost_alpha: exponent of the cost penalty; ``1.0`` divides EI by the predicted runtime,
            ``0.0`` ignores cost.
        xi: exploration bonus of expected improvement.
        cost_scale: cold-start multiplier for predicted runtimes (see
            :class:`~arenaml.surrogate.LinearCostSurrogate`).
        log_cost: model runtimes in log space.
        fit_intercept: intercept for the performance surrogate (thesis default: none).
        budget_margin: a candidate is only eligible if ``pred_cost <= budget_margin * remaining``.
        seed: random seed used to break ties.
    """

    def __init__(
        self,
        names: list[str],
        perf_features: np.ndarray,
        cost_matrix: np.ndarray | None,
        cost_alpha: float = 1.0,
        xi: float = 0.0,
        cost_scale: float = 1.0,
        log_cost: bool = True,
        fit_intercept: bool = False,
        budget_margin: float = 1.0,
        seed: int | None = 0,
    ):
        self.names = list(names)
        n = len(self.names)
        perf_features = np.asarray(perf_features, dtype=float)
        if perf_features.shape[0] != n:
            raise ValueError("perf_features must have one row per candidate")
        self.perf = LinearPerfSurrogate(perf_features, fit_intercept=fit_intercept)
        self.cost: LinearCostSurrogate | None = None
        if cost_matrix is not None:
            cost_matrix = np.asarray(cost_matrix, dtype=float)
            if cost_matrix.shape[0] != n:
                raise ValueError("cost_matrix must have one row per candidate")
            self.cost = LinearCostSurrogate(cost_matrix, log_space=log_cost, scale=cost_scale)
        self.cost_alpha = float(cost_alpha)
        self.xi = float(xi)
        self.budget_margin = float(budget_margin)
        self.rng = np.random.default_rng(seed)
        self.evaluated: dict[int, float] = {}
        self._best: float = DUMMY_LEVEL

    # ------------------------------------------------------------------ state
    @property
    def f_best(self) -> float:
        """Best observed normalised score; the dummy level before any observation."""
        return self._best

    @property
    def n_evaluated(self) -> int:
        return len(self.evaluated)

    def remaining_candidates(self) -> np.ndarray:
        mask = np.ones(len(self.names), dtype=bool)
        if self.evaluated:
            mask[list(self.evaluated)] = False
        return mask

    # ------------------------------------------------------------------ predictions
    def predict(self) -> pd.DataFrame:
        """Current predictive mean / std of the normalised score and runtime for every candidate."""
        mu, sigma = self.perf.predict()
        cost = self.cost.predict() if self.cost is not None else np.full(len(self.names), np.nan)
        return pd.DataFrame(
            {
                "pred_mean": mu,
                "pred_std": sigma,
                "pred_cost": cost,
                "evaluated": ~self.remaining_candidates(),
            },
            index=pd.Index(self.names, name="config"),
        )

    def acquisition(self) -> np.ndarray:
        mu, sigma = self.perf.predict()
        ei = expected_improvement(mu, sigma, self.f_best, xi=self.xi)
        cost = self.cost.predict() if self.cost is not None else None
        return cost_cooled(ei, cost, alpha=self.cost_alpha)

    # ------------------------------------------------------------------ loop
    def suggest(self, remaining_seconds: float | None = None) -> Suggestion | None:
        """Pick the next candidate, or ``None`` when nothing eligible is left.

        Args:
            remaining_seconds: wall-clock budget left; candidates whose predicted runtime does
                not fit are skipped.  ``None`` disables the check.
        """
        mask = self.remaining_candidates()
        mu, sigma = self.perf.predict()
        cost = self.cost.predict() if self.cost is not None else None
        if remaining_seconds is not None and cost is not None:
            mask &= cost <= self.budget_margin * remaining_seconds
        if not mask.any():
            return None
        scores = cost_cooled(
            expected_improvement(mu, sigma, self.f_best, xi=self.xi), cost, alpha=self.cost_alpha
        )
        scores = np.where(mask, scores, -np.inf)
        top = np.flatnonzero(scores == scores.max())
        idx = int(self.rng.choice(top))
        return Suggestion(
            index=idx,
            name=self.names[idx],
            pred_mean=float(mu[idx]),
            pred_std=float(sigma[idx]),
            pred_cost=float(cost[idx]) if cost is not None else float("nan"),
            acquisition=float(scores[idx]),
            f_best=self.f_best,
        )

    def observe(self, index: int, performance: float, seconds: float | None) -> None:
        """Record the target result of a candidate: normalised score and runtime in seconds."""
        index = int(index)
        self.evaluated[index] = float(performance)
        self.perf.observe(index, float(performance))
        if self.cost is not None and seconds is not None and np.isfinite(seconds) and seconds > 0:
            self.cost.observe(index, float(seconds))
        self._best = max(self._best, float(performance))
