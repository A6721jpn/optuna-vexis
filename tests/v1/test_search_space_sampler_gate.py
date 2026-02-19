from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


def _ensure_v1_package_loaded() -> None:
    if "v1" in sys.modules:
        return
    project_root = Path(__file__).resolve().parents[2]
    pkg_dir = project_root / "src" / "v1"
    spec = importlib.util.spec_from_file_location(
        "v1",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load v1 package for tests")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1"] = mod
    spec.loader.exec_module(mod)


_ensure_v1_package_loaded()

from v1.config import BoundsSpec  # noqa: E402
from v1.search_space import FeasibilityAwareSampler, build_fixed_search_space  # noqa: E402


class _DeterministicBaseSampler(BaseSampler):
    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> float:
        # Always returns the same value, so retries would be duplicates
        # unless the wrapper injects random fallback samples.
        return 0.1


class _AlwaysHighFallbackSampler:
    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> float:
        return 0.9


class _RelativeFailingBaseSampler(BaseSampler):
    def __init__(self) -> None:
        self.independent_calls = 0

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
        raise KeyError("TIP-DROP")

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> float:
        self.independent_calls += 1
        return 0.1


class _FailsWhenFixedInRelativeSpaceSampler(BaseSampler):
    def __init__(self) -> None:
        self.saw_fixed_dimension = False

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
        if "TIP-DROP" in search_space:
            self.saw_fixed_dimension = True
            raise KeyError("TIP-DROP")
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> float:
        return 0.3


def test_sampler_uses_random_candidate_when_base_retries_repeat_same_point() -> None:
    bounds = [BoundsSpec(name="x", min=0.0, max=1.0)]
    fixed_space = build_fixed_search_space(bounds)

    sampler = FeasibilityAwareSampler(
        base_sampler=_DeterministicBaseSampler(),
        predict_fn=lambda p: p["x"] >= 0.5,
        fixed_search_space=fixed_space,
        max_retries=2,
    )
    # Make fallback deterministic for test.
    sampler._fallback_random = _AlwaysHighFallbackSampler()  # type: ignore[assignment]

    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=1)

    trial = study.trials[0]
    assert trial.params["x"] >= 0.5
    stats = sampler.rejection_stats
    assert stats["accepted"] == 1
    assert stats["rejected"] == 1


def test_relative_failure_skips_base_independent_sampling() -> None:
    bounds = [BoundsSpec(name="x", min=0.0, max=1.0)]
    fixed_space = build_fixed_search_space(bounds)
    base = _RelativeFailingBaseSampler()

    sampler = FeasibilityAwareSampler(
        base_sampler=base,
        predict_fn=lambda p: p["x"] >= 0.5,
        fixed_search_space=fixed_space,
        max_retries=1,
    )
    sampler._fallback_random = _AlwaysHighFallbackSampler()  # type: ignore[assignment]

    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=1)

    assert study.trials[0].params["x"] >= 0.5
    assert base.independent_calls == 0


def test_fixed_dimensions_are_injected_and_excluded_from_relative_sampling() -> None:
    bounds = [
        BoundsSpec(name="x", min=0.0, max=1.0),
        BoundsSpec(name="TIP-DROP", min=1.0, max=1.0),
    ]
    fixed_space = build_fixed_search_space(bounds)
    base = _FailsWhenFixedInRelativeSpaceSampler()
    seen_candidates: list[dict[str, float]] = []

    def _predict(params: dict[str, float]) -> bool:
        seen_candidates.append(dict(params))
        return params["TIP-DROP"] == 1.0 and params["x"] >= 0.0

    sampler = FeasibilityAwareSampler(
        base_sampler=base,
        predict_fn=_predict,
        fixed_search_space=fixed_space,
        max_retries=1,
    )

    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=1)

    assert not base.saw_fixed_dimension
    assert "TIP-DROP" not in study.trials[0].params
    assert seen_candidates
    assert seen_candidates[0]["TIP-DROP"] == 1.0
