"""Acquisition functions over a discrete candidate set."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.0) -> np.ndarray:
    """Closed-form expected improvement under a Gaussian predictive distribution (maximisation).

    Candidates with zero predictive variance get the deterministic improvement ``max(0, mu - f)``.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    threshold = f_best + xi
    ei = np.zeros_like(mu)
    pos = sigma > 0
    diff = mu[pos] - threshold
    z = diff / sigma[pos]
    ei[pos] = diff * norm.cdf(z) + sigma[pos] * norm.pdf(z)
    ei[~pos] = np.maximum(0.0, mu[~pos] - threshold)
    return ei


def cost_cooled(scores: np.ndarray, cost: np.ndarray | None, alpha: float = 1.0) -> np.ndarray:
    """Divide acquisition scores by ``cost ** alpha`` (Snoek et al. 2012).

    Non-finite or non-positive costs are replaced by the median of the valid ones so that such
    candidates are neither punished nor rewarded.  ``alpha=0`` disables the cooling.
    """
    scores = np.asarray(scores, dtype=float)
    if alpha == 0.0 or cost is None:
        return scores
    cost = np.asarray(cost, dtype=float)
    valid = np.isfinite(cost) & (cost > 0)
    if not valid.any():
        return scores
    safe = np.where(valid, cost, np.median(cost[valid]))
    return scores / safe**alpha
