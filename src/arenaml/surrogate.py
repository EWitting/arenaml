"""Bayesian linear regression surrogates on performance vectors.

Each candidate configuration is represented by its (normalised) benchmark scores on the
TabArena tasks: a feature vector with one entry per benchmark task.  Given the scores observed
so far on the target dataset, a Bayesian ridge regression *without intercept* learns one
weight per benchmark task and predicts the target score of every other candidate together
with a predictive standard deviation.

Before enough target observations exist the weights are uniform (``1 / n_tasks``), so the
prediction of a configuration is simply its average over the benchmark tasks, and the
spread of its scores across tasks serves as the predictive uncertainty.

The runtime surrogate is the same model applied to the runtime matrix.  By default runtimes
are modelled in log space (runtimes span several orders of magnitude across tasks), so the
cold-start estimate is the geometric mean over tasks; ``log_space=False`` gives the
arithmetic mean instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import BayesianRidge


class LinearPerfSurrogate:
    """Bayesian ridge on benchmark-score features with a uniform-weight cold start.

    Args:
        features: ``(n_candidates, n_tasks)`` matrix without NaN.
        fit_intercept: whether the linear model has an intercept (thesis default: no).
        min_observations: number of target observations needed before the model is fitted.
        cold_start_std: ``"row_std"`` uses the per-candidate standard deviation across tasks as
            predictive uncertainty before the model is fitted, ``"none"`` returns zeros.
    """

    def __init__(
        self,
        features: np.ndarray,
        fit_intercept: bool = False,
        min_observations: int = 2,
        cold_start_std: str = "row_std",
    ):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("features must be 2-D (n_candidates, n_tasks)")
        if np.isnan(features).any():
            raise ValueError("features contain NaN; impute before constructing the surrogate")
        if cold_start_std not in ("row_std", "none"):
            raise ValueError("cold_start_std must be 'row_std' or 'none'")
        self.features = features
        self.fit_intercept = fit_intercept
        self.min_observations = max(1, int(min_observations))
        self.cold_start_std = cold_start_std
        self.model_: BayesianRidge | None = None
        self._obs_idx: list[int] = []
        self._obs_y: list[float] = []

    # ------------------------------------------------------------------ state
    @property
    def n_observations(self) -> int:
        return len(self._obs_idx)

    @property
    def is_fitted(self) -> bool:
        return self.model_ is not None

    @property
    def n_tasks(self) -> int:
        return self.features.shape[1]

    @property
    def weights(self) -> np.ndarray:
        """Weight per benchmark task (uniform before the model is fitted)."""
        if self.model_ is None:
            return np.full(self.n_tasks, 1.0 / self.n_tasks)
        return np.asarray(self.model_.coef_, dtype=float)

    # ------------------------------------------------------------------ fitting
    def observe(self, index: int, value: float) -> None:
        """Record the target score of candidate ``index`` and refit if possible."""
        if not np.isfinite(value):
            raise ValueError("observed value must be finite")
        self._obs_idx.append(int(index))
        self._obs_y.append(float(value))
        self._refit()

    def reset(self) -> None:
        self._obs_idx, self._obs_y, self.model_ = [], [], None

    def _refit(self) -> None:
        if self.n_observations < self.min_observations:
            self.model_ = None
            return
        X = self.features[self._obs_idx]
        y = np.asarray(self._obs_y)
        model = BayesianRidge(fit_intercept=self.fit_intercept)
        try:
            model.fit(X, y)
        except Exception:
            self.model_ = None
            return
        self.model_ = model

    # ------------------------------------------------------------------ prediction
    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and standard deviation for every candidate."""
        if self.model_ is None:
            mu = self.features.mean(axis=1)
            if self.cold_start_std == "row_std":
                sigma = self.features.std(axis=1)
            else:
                sigma = np.zeros_like(mu)
            return mu, sigma
        mu, sigma = self.model_.predict(self.features, return_std=True)
        return np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)


class LinearCostSurrogate:
    """Runtime surrogate: :class:`LinearPerfSurrogate` on (log) runtimes.

    Args:
        cost: ``(n_candidates, n_tasks)`` runtimes in seconds, NaN allowed (imputed by the
            per-task median).
        log_space: model ``log(cost)`` instead of ``cost``.
        scale: multiplicative prior correction applied to cold-start estimates, e.g. the number
            of model fits of the user's CV protocol relative to TabArena's 8-fold bagging.
        fit_intercept: intercept for the linear model (default on, it absorbs a constant
            hardware speed factor in log space).
    """

    def __init__(
        self,
        cost: np.ndarray,
        log_space: bool = True,
        scale: float = 1.0,
        fit_intercept: bool = True,
        min_observations: int = 2,
        floor: float = 1e-3,
    ):
        cost = np.asarray(cost, dtype=float)
        cost = np.where(np.isfinite(cost) & (cost > 0), cost, np.nan)
        col_median = np.nanmedian(cost, axis=0)
        col_median = np.where(np.isfinite(col_median), col_median, np.nanmedian(cost))
        cost = np.where(np.isnan(cost), col_median[None, :], cost)
        cost = np.maximum(cost, floor)
        self.log_space = log_space
        self.scale = float(scale)
        self.floor = floor
        features = np.log(cost) if log_space else cost
        self._inner = LinearPerfSurrogate(
            features, fit_intercept=fit_intercept, min_observations=min_observations, cold_start_std="none"
        )

    @property
    def n_observations(self) -> int:
        return self._inner.n_observations

    @property
    def is_fitted(self) -> bool:
        return self._inner.is_fitted

    @property
    def weights(self) -> np.ndarray:
        return self._inner.weights

    def observe(self, index: int, seconds: float) -> None:
        seconds = max(float(seconds), self.floor)
        self._inner.observe(index, np.log(seconds) if self.log_space else seconds)

    def predict(self) -> np.ndarray:
        """Expected runtime in seconds for every candidate."""
        mu, _ = self._inner.predict()
        if self.log_space:
            est = np.exp(mu)
        else:
            est = np.maximum(mu, self.floor)
        if not self._inner.is_fitted:
            est = est * self.scale
        return est
