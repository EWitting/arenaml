"""arenaml vs. plain AutoGluon on a real dataset outside TabArena's 51 benchmark tasks.

Dataset: UCI Adult / Census Income (OpenML ``adult``, version 2) -- ~48.8k rows, 14 mixed
categorical/numeric features with real missing values, binary target (income >$50K). It is
verified at runtime (not just by inspection) to not be one of TabArena's 51 tasks: the loaded
``Leaderboard``'s own ``task_names`` are asserted to exclude it.

Comparison design
------------------
Both systems see exactly the same held-out test split and are optimised for the same metric
(``roc_auc``). arenaml always runs ``ArenaSearch(..., models=ARENAML_MODELS,
cv=CVStrategy("holdout", holdout_frac=HOLDOUT_FRAC))`` -- single leaderboard-config fits, no
ensembling. ``ARENAML_MODELS`` (default ``"cpu"``) and ``AUTOGLUON_MODE`` (default
``"matched"``), both set at the top of this file, control which configs/models each system
may draw from and how AutoGluon validates.

* ``"matched"`` (default): ``TabularPredictor.fit(..., num_bag_folds=0,
  holdout_frac=HOLDOUT_FRAC, fit_weighted_ensemble=False)`` -- AutoGluon's own from-scratch
  CASH/HPO over its default model zoo, under the identical single-holdout, no-ensemble protocol
  arenaml uses (this literally reuses the same AutoGluon holdout-splitting code path that
  arenaml's per-config fits go through, see ``arenaml.evaluate._fit_kwargs``). This isolates
  "does a TabArena-leaderboard-informed search beat AutoGluon's own search", not "does
  ensembling help" or "do the two use different validation".
* ``"defaults"``: ``TabularPredictor.fit(train_data, time_limit=budget)`` with nothing else
  set -- AutoGluon's genuine out-of-the-box behaviour. This typically means weighted
  ensembling of whatever base models it fits and its own automatic holdout split (no bagging
  or stacking unless a preset requests it), which is *not* the same protocol arenaml uses.
  This answers a different, equally valid question -- "plain, default AutoGluon usage" vs.
  arenaml -- where ensembling and AutoGluon's own validation choices are real, expected
  advantages rather than confounds to control away.

What is deliberately NOT controlled in either mode: which model families each system
considers (arenaml draws only from TabArena's benchmarked configs; AutoGluon explores its own
default hyperparameter search space). That is the comparison's whole point.

Both systems are run once per budget in ``BUDGETS`` (a fresh instance each time, no incumbent
carried over between budgets -- the fair way to ask "given only this many seconds, what do you
get"), then scored on the common held-out test set with the same scorer. Regret is defined
relative to *one shared reference*, the best test error observed by either system at any budget
in this experiment (the true optimum on a fresh dataset is unknowable, so this is the usual
"best observed" proxy, exactly as ``compute_normalized_regret`` uses ``max(ground_truth)`` in
the reference code) -- every point in the regret plot subtracts that same number.

Outputs (under ``examples/output/``): ``adult_vs_autogluon.csv`` (one row per method x budget)
and ``adult_vs_autogluon.png`` (error and regret vs. budget).
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import arenaml
from arenaml import CVStrategy
from arenaml.leaderboard import load_leaderboard
from arenaml.normalization import dummy_error

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("adult_vs_autogluon")

OUT = Path(__file__).parent / "output"
RUNS = OUT / "adult_runs"
OUT.mkdir(exist_ok=True)

DATASET = "adult"
PROBLEM_TYPE = "binary"
EVAL_METRIC = "roc_auc"
HOLDOUT_FRAC = 0.2
TEST_FRAC = 0.2
BUDGETS = [15, 30, 60, 120, 240, 300]  # seconds; log-ish spacing, capped at 300s (both
# methods were already plateauing/overfitting past that on this dataset). A 15s floor is
# meaningful now that arenaml's own applicability()/registry setup is warmed at
# `import arenaml` time rather than paid out of the first search's own budget (see
# arenaml.applicability.warm_up) -- it used to cost several seconds by itself.
SEED = 0

# "matched": AutoGluon is forced onto arenaml's own protocol (single holdout, no ensembling)
#   -- isolates search-strategy quality, see the module docstring.
# "defaults": AutoGluon.fit(train_data, time_limit=budget) with nothing else set -- its own
#   out-of-the-box behaviour (weighted ensembling of whatever it fits; no bagging/stacking
#   unless a preset requests it, so its own automatic holdout split rather than HOLDOUT_FRAC).
#   This answers a different question ("plain AutoGluon usage" vs. arenaml) rather than
#   isolating the search strategy -- ensembling and a possibly different validation split are
#   then real, expected advantages of AutoGluon, not confounds to control away.
# "extreme": AutoGluon.fit(train_data, time_limit=budget, presets="extreme_quality") --
#   AutoGluon's strongest preset, multi-layer stacking plus its "zeroshot_2025_12_18_gpu"
#   portfolio (TabDPT, TabICL, Mitra, TabM, GBM, CatBoost, RealTabPFN-v2; confirmed via
#   autogluon.tabular.configs.presets_configs.tabular_presets_dict["extreme_quality"] ->
#   hyperparameters="zeroshot_2025_12_18_gpu"). The explicit time_limit overrides the
#   preset's own default (AutoGluon's own fit() docstring: "Any user-specified arguments in
#   fit() will override the values used by presets."). Needs optional extras this
#   environment does not have installed (tabicl, tabdpt, tabpfn, ...); AutoGluon skips a
#   model it cannot fit rather than failing the whole run, so without those extras this
#   silently degrades toward a smaller (GBM/CAT-only) stacked ensemble -- install the extras
#   first if you actually want the foundation models exercised.
AUTOGLUON_MODE = "defaults"  # "matched" | "defaults" | "extreme"

# Which TabArena leaderboard configs arenaml may draw from. "cpu" (default) matches what
# AutoGluon's own default recipe considers: checked empirically in this environment via
# ``autogluon.tabular.configs.hyperparameter_configs.get_hyperparameter_config("default")``,
# which resolves to exactly {NN_TORCH, GBM, CAT, XGB, FASTAI, RF, XT} -- no TabPFN, TabICL, or
# any other tabular foundation model, regardless of AUTOGLUON_MODE (those need optional extras
# that are not part of a plain AutoGluon install and are never pulled in by the default
# hyperparameter config). So "cpu" is not a handicap relative to AutoGluon's defaults; set
# "all" to deliberately give arenaml access to GPU-tier leaderboard configs AutoGluon's
# default recipe cannot reach either way.
ARENAML_MODELS = "cpu"  # "cpu" | "all"

INK, MUTED, GRID, SURFACE = "#1a1a19", "#6b6a63", "#e6e5df", "#fcfcfb"
COLOR_ARENAML, COLOR_AUTOGLUON = "#2a78d6", "#eb6834"  # validated categorical palette slots 1-2


def load_dataset(lb) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Fetch Adult/Census Income and split once into train/test (used identically by both
    systems). Asserts the dataset is not one of TabArena's own benchmark tasks.
    """
    assert DATASET not in lb.task_names, (
        f"{DATASET!r} is one of TabArena's own benchmark tasks ({lb.task_names}); "
        "pick a dataset outside that set."
    )
    X, y = fetch_openml(DATASET, version=2, return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, stratify=y, random_state=SEED
    )
    log.info(
        "%s: %d train / %d test rows, %d features, positive rate %.3f",
        DATASET,
        len(X_train),
        len(X_test),
        X.shape[1],
        (y == y.unique()[1]).mean(),
    )
    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def _test_error(predictor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """1 - AUC of ``predictor`` on the held-out test set, on the roc_auc scale used throughout."""
    proba = predictor.predict_proba(X_test, as_multiclass=False)
    y_bin = (y_test == predictor.positive_class).astype(int)
    return 1.0 - roc_auc_score(y_bin, proba)


def run_arenaml(budget: int, lb, X_train, y_train, X_test, y_test) -> dict:
    run_dir = RUNS / f"arenaml_{budget}s_{uuid.uuid4().hex[:6]}"
    t0 = time.perf_counter()
    s = arenaml.search(
        X_train,
        y_train,
        problem_type=PROBLEM_TYPE,
        eval_metric=EVAL_METRIC,
        time_budget=budget,
        models=ARENAML_MODELS,
        cv=CVStrategy("holdout", holdout_frac=HOLDOUT_FRAC),
        leaderboard=lb,  # pre-loaded: cache/download time is not charged against the budget
        output_dir=run_dir,
        verbosity=1,
    )
    wall_s = time.perf_counter() - t0
    test_error = _test_error(s.best_predictor_, X_test, y_test)
    row = {
        "method": "arenaml",
        "budget_s": budget,
        "wall_s": wall_s,
        "n_evals": len(s.history_),
        "config": s.best_config_,
        "val_error": s.best_error_,
        "test_error": test_error,
    }
    s.best_result_.cleanup()
    shutil.rmtree(run_dir, ignore_errors=True)
    return row


def run_autogluon(budget: int, X_train, y_train, X_test, y_test) -> dict:
    from autogluon.tabular import TabularPredictor

    run_dir = RUNS / f"autogluon_{budget}s_{uuid.uuid4().hex[:6]}"
    label = "__label__"
    train = X_train.copy()
    train[label] = y_train.values

    fit_kwargs: dict = {"train_data": train, "time_limit": budget}
    if AUTOGLUON_MODE == "matched":
        fit_kwargs.update(
            num_bag_folds=0,  # matches arenaml's CVStrategy("holdout") protocol, see module docstring
            holdout_frac=HOLDOUT_FRAC,
            fit_weighted_ensemble=False,  # arenaml never ensembles either; keep it single-model
            calibrate=False,
        )
    elif AUTOGLUON_MODE == "extreme":
        fit_kwargs["presets"] = "extreme_quality"  # time_limit above still wins, see module docstring
    elif AUTOGLUON_MODE != "defaults":
        raise ValueError(
            f"Unknown AUTOGLUON_MODE {AUTOGLUON_MODE!r}; expected 'matched', 'defaults' or 'extreme'"
        )
    # else "defaults": nothing added -- AutoGluon's own out-of-the-box behaviour.

    t0 = time.perf_counter()
    predictor = TabularPredictor(
        label=label, problem_type=PROBLEM_TYPE, eval_metric=EVAL_METRIC, path=str(run_dir), verbosity=0
    )
    predictor.fit(**fit_kwargs)
    wall_s = time.perf_counter() - t0
    board = predictor.leaderboard(score_format="error", set_refit_score_to_parent=True).set_index("model")
    val_error = float(board.loc[predictor.model_best, "metric_error_val"])
    test_error = _test_error(predictor, X_test, y_test)
    row = {
        "method": f"AutoGluon ({AUTOGLUON_MODE})",
        "budget_s": budget,
        "wall_s": wall_s,
        "n_evals": len(board),
        "config": predictor.model_best,
        "val_error": val_error,
        "test_error": test_error,
    }
    del predictor
    shutil.rmtree(run_dir, ignore_errors=True)
    return row


def plot(results: pd.DataFrame, dummy_err: float) -> None:
    df = results.sort_values("budget_s")
    optimal = df["test_error"].min()  # the single shared regret reference for both curves
    df = df.assign(test_regret=df["test_error"] - optimal)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8, which="both")
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.minorticks_off()  # avoid log-scale minor ticks (2x10^2, ...) colliding with the below
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlabel("time budget (s, log scale)", color=MUTED)

    colors = {"arenaml": COLOR_ARENAML, f"AutoGluon ({AUTOGLUON_MODE})": COLOR_AUTOGLUON}
    ax = axes[0]
    for method, sub in df.groupby("method"):
        ax.plot(
            sub["budget_s"], sub["test_error"], "o-", color=colors[method], label=method, linewidth=2, ms=6
        )
    ax.axhline(dummy_err, color=MUTED, linewidth=1, linestyle=":", label="dummy baseline")
    ax.set_ylabel("test error (1 - AUC)", color=MUTED)
    ax.set_title("Held-out test error vs. budget", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False, labelcolor=INK)

    ax = axes[1]
    for method, sub in df.groupby("method"):
        ax.plot(
            sub["budget_s"], sub["test_regret"], "o-", color=colors[method], label=method, linewidth=2, ms=6
        )
    ax.axhline(0.0, color=MUTED, linewidth=1, linestyle="--")
    # Anchored top-left (empty whitespace above the leftmost points), not on the axhline
    # itself, since a method's regret can sit right at/near 0 for most of the budget range
    # and would otherwise collide with this label.
    ax.text(
        0.02,
        0.95,
        "- - -  best test error observed in this experiment (shared reference)",
        color=MUTED,
        fontsize=8,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    ax.set_ylabel("test regret (test error - best observed)", color=MUTED)
    ax.set_title("Regret vs. budget (same reference for both curves)", color=INK, fontsize=11, loc="left")

    protocol = {
        "matched": f"AutoGluon matched to arenaml's protocol: holdout_frac={HOLDOUT_FRAC}, no ensembling",
        "defaults": "AutoGluon's own defaults: automatic validation split, ensembling enabled",
        "extreme": "AutoGluon extreme_quality preset: multi-layer stacking + zeroshot portfolio",
    }[AUTOGLUON_MODE]
    fig.suptitle(
        f"arenaml vs. AutoGluon on {DATASET} (not a TabArena task) -- {protocol}",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "adult_vs_autogluon.png", dpi=150)


if __name__ == "__main__":
    # load_leaderboard() only understands "all" or an explicit method list (unlike
    # ArenaSearch/applicability(), which also accept "cpu"/"gpu"); load everything once and
    # let ArenaSearch(models=ARENAML_MODELS, leaderboard=lb) filter configs at search time.
    lb = load_leaderboard(methods="all")
    X_train, y_train, X_test, y_test = load_dataset(lb)
    dummy_err = dummy_error(y_test, PROBLEM_TYPE, EVAL_METRIC)
    log.info("dummy baseline test error: %.4f", dummy_err)

    rows = []
    for budget in BUDGETS:
        log.info("=== budget %ds: arenaml ===", budget)
        rows.append(run_arenaml(budget, lb, X_train, y_train, X_test, y_test))
        log.info("=== budget %ds: AutoGluon ===", budget)
        rows.append(run_autogluon(budget, X_train, y_train, X_test, y_test))

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "adult_vs_autogluon.csv", index=False)
    plot(results, dummy_err)
    print(results.to_string(index=False))
    print("saved", OUT / "adult_vs_autogluon.csv", OUT / "adult_vs_autogluon.png")
