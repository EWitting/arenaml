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

from arenaml.applicability import warm_up as _warm_up
from arenaml.evaluate import CVStrategy, EvalResult, run_config
from arenaml.leaderboard import Leaderboard, load_leaderboard
from arenaml.search import ArenaSearch, search

# ArenaSearch.fit() (via applicability()) does several first-touch imports on its own the
# first time it runs -- torch (DLL loading, custom-op registration, CUDA device detection)
# and tabarena's own model-constraints/registry lookups (which pull in tabarena's full
# dependency chain: seaborn, plotly, autorank, ...) -- together several seconds' worth of
# work that would otherwise silently come out of whichever search runs first's own
# time_budget. Paying it here instead, at import time, is exactly what warm_up() is for; see
# its docstring for the full breakdown. Safe to call unconditionally: every step it runs is
# cached and swallows its own import/driver errors.
_warm_up()

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
