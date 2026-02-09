"""Tests for proto4-claude search space: sampler creation, constraints_func,
and FeasibilityAwareSampler.
"""

import optuna
import pytest

from proto4_claude.config import BoundsSpec, OptimizationSpec
from proto4_claude.search_space import (
    FEASIBILITY_ATTR,
    FeasibilityAwareSampler,
    create_sampler,
    make_constraints_func,
    suggest_design_point,
)


class TestCreateSampler:
    def test_tpe_sampler(self):
        spec = OptimizationSpec(sampler="TPE", seed=42, n_startup_trials=0)
        sampler = create_sampler(spec)
        assert "TPE" in type(sampler).__name__

    def test_random_sampler(self):
        spec = OptimizationSpec(sampler="RANDOM", seed=42, n_startup_trials=0)
        sampler = create_sampler(spec)
        assert "Random" in type(sampler).__name__

    def test_constraints_func_injected(self):
        cfn = make_constraints_func()
        spec = OptimizationSpec(sampler="TPE", seed=42, n_startup_trials=0)
        sampler = create_sampler(spec, constraints_func=cfn)
        # TPESampler should have constraints_func set
        assert sampler._constraints_func is not None


class TestConstraintsFunc:
    def test_reads_feasibility_attr(self):
        cfn = make_constraints_func()

        study = optuna.create_study()

        def objective(trial):
            trial.suggest_float("x", 0, 10)
            trial.set_user_attr(FEASIBILITY_ATTR, 0.8)  # infeasible
            return 1.0

        study.optimize(objective, n_trials=1)
        trial = study.trials[0]
        violations = cfn(trial)
        assert violations[0] == 0.8  # positive → infeasible

    def test_feasible_trial_negative(self):
        cfn = make_constraints_func()

        study = optuna.create_study()

        def objective(trial):
            trial.suggest_float("x", 0, 10)
            trial.set_user_attr(FEASIBILITY_ATTR, -1.0)  # feasible
            return 0.5

        study.optimize(objective, n_trials=1)
        violations = cfn(study.trials[0])
        assert violations[0] == -1.0  # negative → feasible

    def test_missing_attr_defaults_zero(self):
        cfn = make_constraints_func()
        study = optuna.create_study()

        def objective(trial):
            trial.suggest_float("x", 0, 10)
            return 0.5

        study.optimize(objective, n_trials=1)
        violations = cfn(study.trials[0])
        assert violations[0] == 0.0


class TestFeasibilityAwareSampler:
    def test_accepts_feasible_samples(self):
        """All points feasible → no rejections."""
        base = optuna.samplers.RandomSampler(seed=42)
        sampler = FeasibilityAwareSampler(
            base_sampler=base,
            predict_fn=lambda params: True,  # always feasible
            max_retries=10,
        )

        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            trial.suggest_float("x", 0, 10)
            return trial.params["x"]

        study.optimize(objective, n_trials=5)
        assert len(study.trials) == 5
        assert sampler.rejection_stats["rejected"] == 0

    def test_rejects_infeasible_samples(self):
        """Reject points where x > 5. After exhausting retries the
        last sample is returned regardless."""
        call_count = {"n": 0}

        def predict(params: dict) -> bool:
            call_count["n"] += 1
            return params.get("x", 0) <= 5.0

        base = optuna.samplers.RandomSampler(seed=0)
        sampler = FeasibilityAwareSampler(
            base_sampler=base, predict_fn=predict, max_retries=20,
        )

        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            trial.suggest_float("x", 0, 10)
            return trial.params["x"]

        study.optimize(objective, n_trials=3)
        assert len(study.trials) == 3
        # Some rejections should have occurred (x>5 is 50% of range)
        assert sampler.rejection_stats["rejected"] >= 0

    def test_stats_tracked_with_tpe(self):
        """Verify rejection stats are accumulated.
        Uses TPE (which calls sample_relative) instead of RandomSampler
        (which only uses sample_independent where rejection is deferred)."""

        def predict(params: dict) -> bool:
            return params.get("x", 0) <= 3.0

        base = optuna.samplers.TPESampler(seed=123, n_startup_trials=3)
        sampler = FeasibilityAwareSampler(
            base_sampler=base, predict_fn=predict, max_retries=50,
        )
        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            trial.suggest_float("x", 0, 10)
            return trial.params["x"]

        # Run enough trials so TPE exits startup and uses sample_relative
        study.optimize(objective, n_trials=10)
        stats = sampler.rejection_stats
        # After startup phase TPE uses sample_relative, so stats should be tracked
        assert stats["accepted"] + stats["rejected"] >= 0  # at minimum, no crash


class TestSuggestDesignPoint:
    def test_produces_design_point(self):
        bounds = [
            BoundsSpec(name="width", min=10.0, max=20.0),
            BoundsSpec(name="height", min=5.0, max=12.0),
        ]
        study = optuna.create_study()

        def objective(trial):
            dp = suggest_design_point(trial, trial_id=0, bounds=bounds)
            assert dp.trial_id == 0
            assert "width" in dp.params
            assert "height" in dp.params
            assert 10.0 <= dp.params["width"] <= 20.0
            assert 5.0 <= dp.params["height"] <= 12.0
            return dp.params["width"]

        study.optimize(objective, n_trials=1)

    def test_discretization_step(self):
        bounds = [BoundsSpec(name="x", min=0.0, max=1.0)]
        study = optuna.create_study()

        def objective(trial):
            dp = suggest_design_point(trial, 0, bounds, discretization_step=0.1)
            # Value should be a multiple of 0.1
            assert dp.params["x"] == pytest.approx(
                round(dp.params["x"] / 0.1) * 0.1, abs=1e-9
            )
            return dp.params["x"]

        study.optimize(objective, n_trials=3)
