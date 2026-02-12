import optuna
import pytest
from optuna.samplers import RandomSampler, TPESampler, GPSampler

from proto4_codex.config import BoundsSpec, OptimizationSpec
import proto4_codex.search_space as ss


def test_auto_fallback_uses_tpe_without_startup(monkeypatch):
    monkeypatch.setattr(ss, "_HAS_AUTO", False)

    spec = OptimizationSpec(sampler="AUTO", seed=42, n_startup_trials=70)
    sampler = ss.create_sampler(
        spec,
        constraints_func=ss.make_constraints_func(),
        n_objectives=1,
    )

    assert isinstance(sampler, TPESampler)
    assert sampler._n_startup_trials == 0


def test_create_sampler_supports_gp():
    spec = OptimizationSpec(sampler="GP", seed=42, n_startup_trials=0)
    sampler = ss.create_sampler(
        spec,
        constraints_func=ss.make_constraints_func(),
    )
    assert isinstance(sampler, GPSampler)


def test_create_sampler_supports_nsgaiii():
    if not getattr(ss, "_HAS_NSGAIII", False):
        pytest.skip("NSGAIIISampler unavailable in this Optuna build")
    spec = OptimizationSpec(sampler="NSGAIII", seed=42, n_startup_trials=0)
    sampler = ss.create_sampler(
        spec,
        constraints_func=ss.make_constraints_func(),
        n_objectives=3,
    )
    assert "NSGAIII" in type(sampler).__name__.upper()


def test_create_sampler_supports_motpe_alias():
    spec = OptimizationSpec(sampler="MOTPE", seed=42, n_startup_trials=0)
    sampler = ss.create_sampler(
        spec,
        constraints_func=ss.make_constraints_func(),
        n_objectives=2,
    )
    assert isinstance(sampler, TPESampler)
    assert sampler._multivariate is True
    assert sampler._group is True
    assert sampler._n_startup_trials == 0


def test_motpe_with_feasibility_wrapper_runs_without_assertion():
    bounds = [
        BoundsSpec(name="x", min=0.0, max=1.0),
        BoundsSpec(name="y", min=0.0, max=1.0),
    ]
    fixed_space = ss.build_fixed_search_space(bounds)
    base = ss.create_sampler(
        OptimizationSpec(sampler="MOTPE", seed=42, n_startup_trials=0),
        constraints_func=ss.make_constraints_func(),
        n_objectives=2,
    )
    sampler = ss.FeasibilityAwareSampler(
        base_sampler=base,
        predict_fn=lambda _: True,
        fixed_search_space=fixed_space,
        expected_param_names=["x", "y"],
        max_retries=2,
    )
    study = optuna.create_study(sampler=sampler, directions=["maximize", "maximize"])

    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_float("y", 0.0, 1.0)
        return x, y

    study.optimize(objective, n_trials=3)
    assert len(study.trials) == 3


def test_feasibility_check_uses_complete_parameter_vector():
    bounds = [
        BoundsSpec(name="x", min=0.0, max=1.0),
        BoundsSpec(name="y", min=0.0, max=1.0),
    ]
    fixed_space = ss.build_fixed_search_space(bounds)

    seen_param_sizes: list[int] = []

    def predict(params: dict[str, float]) -> bool:
        seen_param_sizes.append(len(params))
        return params["x"] + params["y"] <= 1.2

    sampler = ss.FeasibilityAwareSampler(
        base_sampler=RandomSampler(seed=0),
        predict_fn=predict,
        fixed_search_space=fixed_space,
        expected_param_names=["x", "y"],
        max_retries=3,
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_float("y", 0.0, 1.0)
        return x + y

    study.optimize(objective, n_trials=5)

    assert seen_param_sizes
    assert all(size == 2 for size in seen_param_sizes)


def test_repair_path_returns_feasible_point_when_retries_exhausted():
    bounds = [
        BoundsSpec(name="x", min=0.0, max=1.0),
        BoundsSpec(name="y", min=0.0, max=1.0),
    ]
    fixed_space = ss.build_fixed_search_space(bounds)

    def predict(params: dict[str, float]) -> bool:
        return params["x"] == 0.0 and params["y"] == 0.0

    def repair(_: dict[str, float]) -> dict[str, float]:
        return {"x": 0.0, "y": 0.0}

    sampler = ss.FeasibilityAwareSampler(
        base_sampler=RandomSampler(seed=1),
        predict_fn=predict,
        fixed_search_space=fixed_space,
        repair_fn=repair,
        max_retries=1,
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        y = trial.suggest_float("y", 0.0, 1.0)
        return x + y

    study.optimize(objective, n_trials=4)

    for t in study.trials:
        assert t.params["x"] == pytest.approx(0.0)
        assert t.params["y"] == pytest.approx(0.0)

    stats = sampler.rejection_stats
    assert stats["repaired"] == 4
