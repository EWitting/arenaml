"""Decide which leaderboard configurations can run on the target dataset.

A configuration is excluded when

* its method was not requested,
* its model family has no genuine TabArena results for the target's problem type
  (e.g. TabICL or TabFlex on regression),
* TabArena's own dataset-size constraints for the model reject the target
  (e.g. TabPFNv2 above 10k training rows or 500 features),
* it needs a GPU and none is available,
* its optional Python dependencies are not installed,
* its AutoGluon model class cannot be resolved.

The same rules produce a table with one reason per configuration so the user can see why a
model was left out.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from arenaml.leaderboard import Leaderboard

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetProfile:
    """Shape of the target dataset as seen by one model fit."""

    problem_type: str
    n_samples: int
    n_features: int
    n_classes: int
    n_train_per_fold: int


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@cache
def _model_constraints() -> dict:
    """TabArena's per-model dataset constraints keyed by AutoGluon key (empty if unavailable)."""
    try:
        from tabarena.benchmark.experiment.bundle import TabArenaExperimentBundle

        return dict(TabArenaExperimentBundle.DEFAULT_MODEL_CONSTRAINTS)
    except Exception as exc:  # pragma: no cover
        log.warning("Could not load TabArena model constraints: %s", exc)
        return {}


@cache
def _registry_by_class_name() -> dict:
    try:
        from tabarena.models import get_model_registry

        return {info.model_cls.__name__: info for info in get_model_registry().values()}
    except Exception as exc:  # pragma: no cover
        log.warning("Could not load the TabArena model registry: %s", exc)
        return {}


_SPEC_SPLIT = re.compile(r"[\s@\[<>=!~;]")


def _import_name(spec: str) -> str:
    """Best-effort module name for a pip requirement spec."""
    spec = spec.strip()
    if spec.startswith("autogluon.tabular[interpret]"):
        return "interpret"
    name = _SPEC_SPLIT.split(spec, maxsplit=1)[0]
    return (
        name.replace("-", "_").replace(".", "_")
        if name.startswith("tabpfn_extensions")
        else name.replace("-", "_")
    )


@cache
def missing_dependencies(model_cls_name: str) -> tuple[str, ...]:
    """Optional dependencies of a model class that are not importable."""
    info = _registry_by_class_name().get(model_cls_name)
    if info is None:
        return ()
    missing = []
    for spec in info.pip_extra:
        module = _import_name(spec)
        try:
            found = importlib.util.find_spec(module) is not None
        except Exception:
            found = False
        if not found:
            missing.append(spec)
    return tuple(missing)


@cache
def resolve_model_cls(model_cls_name: str, ag_key: str | None = None):
    """Resolve the ``model_cls`` string from the TabArena artifacts to an AutoGluon model class.

    The artifacts store the class name used at benchmark time; when that name no longer exists
    (renamed wrapper), the AutoGluon registry key stored next to it is tried as a fallback.
    """
    from tabarena.benchmark.exec_models import registry

    # Pass the registry explicitly: the module builds it lazily via PEP 562 ``__getattr__``,
    # which does not cover the bare-name lookup inside ``infer_model_cls`` itself.
    model_register = registry.tabarena_model_registry
    try:
        return registry.infer_model_cls(model_cls_name, model_register=model_register)
    except AssertionError:
        if ag_key and ag_key in model_register.key_to_cls_map():
            return model_register.key_to_cls(key=ag_key)
        raise


def _size_ok(ag_key: str, target: TargetProfile) -> bool:
    constraints = _model_constraints().get(ag_key)
    if constraints is None:
        return True
    return constraints.applies(
        n_features=target.n_features,
        n_classes=target.n_classes,
        n_samples_train_per_fold=target.n_train_per_fold,
        problem_type=target.problem_type,
    )


def applicability(
    lb: Leaderboard,
    target: TargetProfile,
    methods: Iterable[str] | str = "all",
    include_gpu_models: bool | None = None,
    check_dependencies: bool = True,
    min_support: float = 0.5,
) -> pd.DataFrame:
    """Per-configuration applicability table with columns ``applicable`` and ``reason``.

    Args:
        lb: the leaderboard.
        target: shape of the target dataset.
        methods: ``"all"``, ``"cpu"``, ``"gpu"`` or an explicit list of TabArena method names.
        include_gpu_models: force GPU models in or out; ``None`` includes them only when CUDA is
            available.
        check_dependencies: drop models whose optional packages are not installed.
        min_support: a method must have genuine results on at least this fraction of the
            benchmark tasks of the target's problem type, otherwise it is considered unsupported.
    """
    configs = lb.configs
    reason = pd.Series("", index=configs.index, dtype=object)

    if isinstance(methods, str):
        if methods == "all":
            selected = pd.Series(True, index=configs.index)
        elif methods in ("cpu", "gpu"):
            selected = configs["compute"] == methods
        else:
            selected = configs["method"] == methods
    else:
        wanted = set(methods)
        unknown = wanted - set(lb.methods)
        if unknown:
            raise ValueError(f"Unknown methods {sorted(unknown)}; available: {lb.methods}")
        selected = configs["method"].isin(wanted)
    reason[~selected] = "method_not_selected"

    support = lb.method_support()
    if target.problem_type in support.columns:
        supported_methods = support.index[support[target.problem_type] >= min_support]
    else:
        supported_methods = support.index
    unsupported = ~configs["method"].isin(supported_methods)
    reason[(reason == "") & unsupported] = "problem_type_unsupported"

    size_ok = configs["ag_key"].map(lambda k: _size_ok(k, target)).astype(bool)
    reason[(reason == "") & ~size_ok] = "dataset_size_constraint"

    if include_gpu_models is None:
        include_gpu_models = gpu_available()
    if not include_gpu_models:
        reason[(reason == "") & (configs["compute"] == "gpu")] = "gpu_required"

    if check_dependencies:
        missing = configs["model_cls"].map(lambda c: bool(missing_dependencies(str(c))))
        reason[(reason == "") & missing] = "missing_dependency"
        for cls_name in configs.loc[(reason == "missing_dependency"), "model_cls"].unique():
            log.info("Excluding %s: missing %s", cls_name, ", ".join(missing_dependencies(str(cls_name))))

    resolvable = {}
    keys = configs["config_ag_key"] if "config_ag_key" in configs.columns else configs["ag_key"]
    pairs = pd.DataFrame({"cls": configs["model_cls"].astype(str), "key": keys.astype(str)})
    for cls_name, key in pairs.loc[reason == ""].drop_duplicates().itertuples(index=False):
        try:
            resolve_model_cls(cls_name, key)
            resolvable[cls_name] = True
        except Exception as exc:
            log.warning("Cannot resolve model class %s (key %s): %s", cls_name, key, exc)
            resolvable[cls_name] = False
    unresolvable = configs["model_cls"].astype(str).map(lambda c: not resolvable.get(c, True))
    reason[(reason == "") & unresolvable] = "model_cls_unresolvable"

    table = pd.DataFrame({"method": configs["method"], "applicable": reason == "", "reason": reason})
    table.loc[table["applicable"], "reason"] = np.nan
    return table
