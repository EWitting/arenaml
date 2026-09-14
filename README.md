# arenaml

One-line AutoML on top of the [TabArena](https://github.com/autogluon/tabarena) leaderboard.

```python
import arenaml

search = arenaml.search(X_train, y_train, time_budget=600)
y_pred = search.predict(X_test)
```

Given a dataset, a time budget and a cross-validation strategy, `arenaml` picks the best
configuration among the thousands of model configurations benchmarked on TabArena (gradient
boosting, neural networks, tabular foundation models, ... with their tuned hyperparameter
settings), runs the most promising ones on your data through AutoGluon exactly as TabArena
ran them, and returns the fitted best one.

## How it works

The search is cost-aware Bayesian optimisation over a *discrete* candidate set: every
configuration on the leaderboard. What differs from standard HPO is the surrogate.

* **Performance representation instead of hyperparameter encoding.** Each candidate is
  represented by its benchmark scores on the 51 TabArena tasks. A Bayesian linear regression
  (scikit-learn `BayesianRidge`, no intercept) is fitted on the candidates already evaluated on
  the target dataset, learning one weight per benchmark task that relates it to the target.
  The model then predicts a score distribution for every remaining candidate.
* **Runtime prediction with the same trick.** The same matrix is built from the benchmark
  train + inference times and the same Bayesian linear regression (on log runtime) predicts
  how long every candidate will take on your machine and data.
* **Cold start with uniform weights.** Before any candidate has been run, all task weights
  are `1 / n_tasks`, so the estimated score of a candidate is its average over the benchmark
  tasks and the estimated runtime is its (geometric) average runtime.
* **Acquisition: expected improvement per second.** Candidates are ranked by expected
  improvement over the best score observed so far divided by the predicted runtime
  (`cost_alpha=1`), and candidates that do not fit in the remaining budget are skipped.
* **Normalisation without leakage.** Every benchmark error (`1 - AUC`, log loss, RMSE) is
  divided by the error of a constant dummy predictor on the same task and negated, so all
  tasks and task types are higher-is-better and on a comparable scale, and `-1` always means
  "dummy level". The same dummy error is computed on the target dataset from its labels, so
  observed scores land on the same scale. Scores worse than the dummy predictor are clipped to
  the dummy level (`clip_at_dummy=True`). TabArena's own leaderboard rescaling uses the best
  score per task, which is not available during a search, and is only used for reporting.
* **Applicability.** Models are excluded when they have no genuine benchmark results for the
  target's problem type (e.g. TabICL on regression; TabArena's imputed rows are ignored), when
  TabArena's dataset size constraints reject the target (TabPFNv2 above 10k rows), when they
  need a GPU that is not available, or when their optional dependencies are not installed.

This is the "Linear-Perf" method of the thesis *Simple Meta-Learning for HPO and CASH Using
Performance Representations*, packaged for real use: the whole TabArena leaderboard is the
meta-dataset and the models are actually trained on your data.

## Installation

TabArena is not on PyPI, so install it (and through it AutoGluon) first:

```bash
git clone https://github.com/autogluon/tabarena
uv pip install --prerelease=allow -e "./tabarena/packages/tabarena" -e "./tabarena/packages/bencheval"
uv pip install -e ./arenaml
```

Optional model families need extra packages (`tabpfn`, `tabicl`, `pytabkit` for RealMLP,
`tabdpt`, ...); see the extras of `tabarena/packages/tabarena/pyproject.toml`. Models whose
extras are missing are skipped automatically. GPU-benchmarked models are only considered when
CUDA is available (override with `include_gpu_models=True`).

The first search downloads the TabArena result tables into `~/.cache/tabarena` (set
`TABARENA_CACHE` to move it) and caches the aggregated matrices in `~/.cache/arenaml`
(`ARENAML_CACHE`).

## Usage

```python
import arenaml
from arenaml import CVStrategy

search = arenaml.search(
    X_train, y_train,
    time_budget=1800,                        # seconds, model fits included
    models=["LightGBM", "CatBoost", "XGBoost", "RealMLP_GPU", "TabPFNv2_GPU"],  # or "all" / "cpu" / "gpu"
    cv=CVStrategy("kfold", n_folds=8),       # TabArena's protocol (bagged, out-of-fold score)
    # cv=CVStrategy("holdout", holdout_frac=0.2),   # cheaper alternative
    eval_metric=None,                        # default: roc_auc / log_loss / rmse by problem type
    refit_full=False,                        # refit the winner on all rows at the end
)

search.best_config_          # e.g. "CatBoost_r13_BAG_L1"
search.best_error_           # validation metric error of the winner
search.best_predictor_       # the fitted AutoGluon TabularPredictor
search.history_              # one row per evaluated configuration
search.predictions()         # surrogate mean / std / runtime for every candidate
search.task_weights()        # learned weight per TabArena task
search.applicability_        # why each leaderboard configuration was in or out
search.predict(X_test); search.predict_proba(X_test)
```

`arenaml.search(...)` is `ArenaSearch(**kwargs).fit(X, y)`; see the `ArenaSearch` docstring
for all options (`cost_alpha`, `xi`, `source_tasks="same_type"`, `max_evals`,
`max_eval_time`, `num_cpus`, `num_gpus`, `output_dir`, ...).

Lower-level pieces are available too:

```python
from arenaml import load_leaderboard, run_config, CVStrategy

lb = load_leaderboard(methods="all")          # error / cost matrices + config specs + task info
spec = lb.configs.loc["LightGBM_r19_BAG_L1"]
res = run_config("LightGBM_r19_BAG_L1", spec["model_cls"], spec["hyperparameters"],
                 X, y, problem_type="binary", eval_metric="roc_auc", cv=CVStrategy("holdout"))
res.metric_error_val, res.seconds, res.predictor
```

## Notes and limitations

* Runtimes on the leaderboard were measured on TabArena's hardware with 8-fold bagging. The
  cold-start runtime estimate is rescaled by the number of fits of your CV strategy, and the
  runtime surrogate adapts to your machine after the first two evaluations.
* Benchmark scores use TabArena's `metric_error` on the test folds by default
  (`score_column="val"` switches to the out-of-fold validation error). If you pass a custom
  `eval_metric`, the target is optimised for it while the benchmark columns stay on
  TabArena's metric; the linear surrogate only needs the two to be monotonically related.
* The problem type is inferred with AutoGluon's rules when not given; pass
  `problem_type="multiclass"` explicitly for integer-coded classes to avoid a regression fit.
* Failed or timed-out fits are scored at the dummy level (`penalize_failures=True`) so the
  surrogate steers away from similar configurations.
* Ensembling several configurations is out of scope for now.
* GPU families need a CUDA build of torch (e.g. `uv pip install --index-url
  https://download.pytorch.org/whl/cu130 "torch==2.13.0+cu130"`). TabM_GPU was verified end-to-end
  on an RTX 5060; the other GPU families additionally need their extras (`tabpfn`, `tabicl`,
  `pytabkit`, ...) and were not run.

## Development

```bash
uv venv --python 3.12 && uv pip install --prerelease=allow -e ../tabarena/packages/tabarena -e ../tabarena/packages/bencheval -e ".[dev]"
pytest tests/test_core.py                  # pure unit tests
pytest tests/test_tabarena.py              # needs the TabArena artifacts (downloaded on demand)
pytest tests/test_live.py                  # fits real models, a few minutes on CPU
ruff check src tests && ruff format src tests
```
