"""arenaml: one-line AutoML on top of the TabArena leaderboard.

Every configuration on the TabArena leaderboard (across all model families) is a candidate and
the search runs cost-aware Bayesian optimisation over that discrete set on the user's dataset.
Instead of a surrogate over encoded hyperparameters, each configuration is represented by its
benchmark scores on the TabArena tasks and a Bayesian linear regression without intercept
learns one weight per benchmark task relating it to the target dataset.  The same construction
on the runtime matrix predicts the cost of every candidate, and expected improvement per unit
of predicted cost selects the next configuration to run.
"""

from __future__ import annotations

from arenaml.evaluate import CVStrategy, EvalResult, run_config
from arenaml.leaderboard import Leaderboard, load_leaderboard
from arenaml.search import ArenaSearch, search

__all__ = [
    "ArenaSearch",
    "CVStrategy",
    "EvalResult",
    "Leaderboard",
    "load_leaderboard",
    "run_config",
    "search",
]

__version__ = "0.1.0"
