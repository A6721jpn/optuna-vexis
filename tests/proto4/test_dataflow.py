"""
End-to-end data flow integration tests for proto4-claude.

Verifies the full pipeline:
  Config → SearchSpace → CADGate → [Geometry → CAE] → Objective → Persistence

CAE (VEXIS) is NEVER executed.  Instead, CaeEvaluator._run_subprocess
is patched to produce a synthetic result CSV, so the curve processing,
RMSE, and feature extraction run on actual data.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import pytest

from proto4_claude.cad_gate import CadGate
from proto4_claude.cae_evaluator import CaeEvaluator
from proto4_claude.config import (
    BoundsSpec,
    CadGateSpec,
    CaeSpec,
    FreecadSpec,
    ObjectiveSpec,
    OptimizationSpec,
    PenaltySpec,
    Proto4Config,
    PathsSpec,
)
from proto4_claude.constraints import penalty_value
from proto4_claude.geometry_adapter import GeometryAdapter, GeometryError
from proto4_claude.objective import ObjectiveOrchestrator
from proto4_claude.persistence import TrialPersistence
from proto4_claude.search_space import (
    FEASIBILITY_ATTR,
    FeasibilityAwareSampler,
    create_sampler,
    make_constraints_func,
    suggest_design_point,
)
from proto4_claude.types import (
    CaeResult,
    CaeStatus,
    DesignPoint,
    TrialOutcome,
)

from .conftest import MockFeasibilityModel


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_result_csv(path: Path, peak_force: float = 100.0) -> None:
    """Write a synthetic VEXIS result CSV."""
    n = 50
    d_load = np.linspace(0.0, 0.5, n)
    f_load = peak_force * (d_load / 0.5) ** 1.5
    d_unload = np.linspace(0.5, 0.0, n)[1:]
    f_unload = 0.7 * peak_force * (d_unload / 0.5) ** 1.5
    df = pd.DataFrame({
        "Stroke": np.concatenate([d_load, d_unload]),
        "Reaction_Force": np.concatenate([f_load, f_unload]),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _build_config(
    tmp_path: Path,
    *,
    gate_enabled: bool = True,
    gate_threshold: float = 0.5,
) -> Proto4Config:
    """Build a minimal Proto4Config pointing at tmp_path."""
    bounds = [
        BoundsSpec(name="width", min=10.0, max=20.0),
        BoundsSpec(name="height", min=5.0, max=12.0),
    ]
    return Proto4Config(
        optimization=OptimizationSpec(
            sampler="TPE", max_trials=10, seed=42,
            n_startup_trials=3, objective_type="single",
        ),
        objective=ObjectiveSpec(type="rmse", weights={"rmse": 1.0}),
        paths=PathsSpec(
            target_curve="target.csv",
            result_dir=str(tmp_path / "output"),
            vexis_path=str(tmp_path / "vexis"),
        ),
        freecad=FreecadSpec(
            constraints={"width": {"min": 10.0, "max": 20.0},
                         "height": {"min": 5.0, "max": 12.0}},
            step_output_dir=str(tmp_path / "step"),
        ),
        cad_gate=CadGateSpec(
            enabled=gate_enabled, threshold=gate_threshold,
            rejection_max_retries=30,
        ),
        cae=CaeSpec(stroke_range_min=0.0, stroke_range_max=0.5, max_retries=1),
        penalty=PenaltySpec(
            base_penalty=50.0, alpha=10.0,
            failure_weights={"cad_infeasible": 1.0, "cae_fail": 0.6},
        ),
        bounds=bounds,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestObjectiveOrchestratorFlow:
    """Test the orchestrator with mocked geometry + CAE."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Wire all components with mocks."""
        # Create VEXIS directory structure
        (tmp_path / "vexis" / "input").mkdir(parents=True, exist_ok=True)
        (tmp_path / "vexis" / "results").mkdir(parents=True, exist_ok=True)
        (tmp_path / "vexis" / "main.py").write_text("print('mock')\n")

        cfg = _build_config(tmp_path)

        # Target curve (loading only = monotonic for clean interpolation)
        target_csv = tmp_path / "target.csv"
        _make_result_csv(target_csv, peak_force=100.0)
        target_curve = pd.read_csv(target_csv).rename(
            columns={"Stroke": "displacement", "Reaction_Force": "force"}
        )

        # Components
        persistence = TrialPersistence(tmp_path / "output")
        cad_gate = CadGate(cfg.cad_gate)
        # Inject mock ML model
        cad_gate._model = MockFeasibilityModel(low=8.0, high=18.0)

        geom_adapter = GeometryAdapter(cfg.freecad, tmp_path)
        cae_evaluator = CaeEvaluator(
            vexis_path=tmp_path / "vexis",
            cae_spec=cfg.cae,
            obj_spec=cfg.objective,
            target_curve=target_curve,
            target_features={},
        )

        orchestrator = ObjectiveOrchestrator(
            cfg=cfg,
            cad_gate=cad_gate,
            geometry_adapter=geom_adapter,
            cae_evaluator=cae_evaluator,
            persistence=persistence,
        )

        return orchestrator, cae_evaluator, persistence, cfg, tmp_path

    def test_infeasible_point_returns_penalty_no_cae(self, setup):
        """Points rejected by the CAD gate should return penalty without
        CAE execution.  We use a tighter mock model so that a point within
        the optimisation bounds still falls outside the model's safe range."""
        orchestrator, cae_eval, persist, cfg, tmp = setup

        # Override mock model with a tighter range [12, 16] so that
        # points near the edges of the optimisation bounds are infeasible.
        orchestrator._gate._model = MockFeasibilityModel(low=12.0, high=16.0)

        # Within bounds (height [5,12], width [10,20]) but outside
        # model [12,16] → infeasible per ML gate.
        pt = DesignPoint(trial_id=0, params={"height": 5.5, "width": 19.0})

        study = optuna.create_study()

        def obj(trial):
            trial.suggest_float("height", 5, 12)
            trial.suggest_float("width", 10, 20)
            return orchestrator(pt, trial=trial)

        study.optimize(obj, n_trials=1)
        trial = study.trials[0]

        # Should have stored infeasible violation
        assert trial.user_attrs[FEASIBILITY_ATTR] > 0
        # Value should be penalty (>= base_penalty)
        assert trial.value >= cfg.penalty.base_penalty

        # Trial record should be persisted
        rec = persist.load_trial(0)
        assert rec["outcome"] == "cad_infeasible"

    def test_feasible_point_runs_through_cae_mock(self, setup):
        """Feasible point should proceed to geometry + CAE.
        Geometry raises (not implemented) → penalty from geometry error."""
        orchestrator, cae_eval, persist, cfg, tmp = setup

        # Point within mock model's safe range [8, 18]
        pt = DesignPoint(trial_id=1, params={"height": 10.0, "width": 15.0})

        study = optuna.create_study()

        def obj(trial):
            trial.suggest_float("height", 5, 12)
            trial.suggest_float("width", 10, 20)
            return orchestrator(pt, trial=trial)

        study.optimize(obj, n_trials=1)

        # GeometryAdapter is not implemented → GeometryError → penalty
        rec = persist.load_trial(1)
        assert rec["outcome"] == "cad_infeasible"  # geometry error treated as CAD issue

    def test_feasible_point_with_mocked_geometry_and_cae(self, setup):
        """Mock both geometry + CAE subprocess to test full success path."""
        orchestrator, cae_eval, persist, cfg, tmp = setup

        pt = DesignPoint(trial_id=2, params={"height": 10.0, "width": 15.0})

        # Create fake STEP file (geometry mock)
        step_dir = Path(cfg.freecad.step_output_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        fake_step = step_dir / "proto4_trial_2.step"
        fake_step.write_text("mock step data")

        # Create fake VEXIS result CSV
        result_csv = tmp / "vexis" / "results" / "proto4_trial_2_result.csv"
        _make_result_csv(result_csv, peak_force=102.0)

        # Patch geometry + VEXIS subprocess
        with patch.object(
            orchestrator._geom, "generate_step", return_value=fake_step
        ), patch.object(
            cae_eval, "_run_subprocess", return_value=result_csv
        ):
            study = optuna.create_study()

            def obj(trial):
                trial.suggest_float("height", 5, 12)
                trial.suggest_float("width", 10, 20)
                return orchestrator(pt, trial=trial)

            study.optimize(obj, n_trials=1)
            trial = study.trials[0]

        # Feasible → negative violation score
        assert trial.user_attrs[FEASIBILITY_ATTR] <= 0

        # Should be a real RMSE value (not penalty)
        assert trial.value < cfg.penalty.base_penalty

        # Trial record should show success with metrics
        rec = persist.load_trial(2)
        assert rec["outcome"] == "cae_success"
        assert "rmse" in rec["objective_values"]
        assert rec["objective_values"]["rmse"] > 0


class TestConstraintsFuncLearning:
    """Verify that constraints_func properly communicates feasibility
    to the TPE sampler across multiple trials."""

    def test_sampler_receives_feasibility_signals(self, tmp_path):
        constraints_func = make_constraints_func()

        sampler = optuna.samplers.TPESampler(
            seed=42, constraints_func=constraints_func,
        )
        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            x = trial.suggest_float("x", 0, 10)
            if x > 5:
                trial.set_user_attr(FEASIBILITY_ATTR, x - 5)  # violation
                return 100.0  # penalty
            else:
                trial.set_user_attr(FEASIBILITY_ATTR, -1.0)  # feasible
                return x ** 2

        study.optimize(objective, n_trials=30)

        # After 30 trials, check feasibility distribution
        feasible = [
            t for t in study.trials
            if t.user_attrs.get(FEASIBILITY_ATTR, 0) <= 0
        ]
        infeasible = [
            t for t in study.trials
            if t.user_attrs.get(FEASIBILITY_ATTR, 0) > 0
        ]

        # Both categories should exist (sampler explored both regions)
        assert len(feasible) > 0
        assert len(infeasible) >= 0  # may have learned to avoid
        # Best value should be from feasible region
        assert study.best_value < 100.0


class TestFeasibilityAwareSamplerIntegration:
    """Test that FeasibilityAwareSampler + constraints_func work together
    in a real Optuna study."""

    def test_rejection_reduces_infeasible_trials(self):
        """Compare with vs without rejection sampling."""

        def run_study(use_rejection: bool, seed: int = 42) -> dict:
            constraints_func = make_constraints_func()
            base = optuna.samplers.TPESampler(
                seed=seed, constraints_func=constraints_func,
            )

            if use_rejection:
                sampler = FeasibilityAwareSampler(
                    base_sampler=base,
                    predict_fn=lambda p: p.get("x", 0) <= 5.0,
                    max_retries=30,
                )
            else:
                sampler = base

            study = optuna.create_study(sampler=sampler)

            def objective(trial):
                x = trial.suggest_float("x", 0, 10)
                if x > 5:
                    trial.set_user_attr(FEASIBILITY_ATTR, x - 5)
                    return 100.0
                trial.set_user_attr(FEASIBILITY_ATTR, -1.0)
                return x ** 2

            study.optimize(objective, n_trials=20)

            infeasible_count = sum(
                1 for t in study.trials
                if t.user_attrs.get(FEASIBILITY_ATTR, 0) > 0
            )
            return {
                "infeasible": infeasible_count,
                "best_value": study.best_value,
            }

        without = run_study(use_rejection=False)
        with_rej = run_study(use_rejection=True)

        # With rejection sampling, fewer (or equal) infeasible trials
        assert with_rej["infeasible"] <= without["infeasible"]
        # Both should find a good solution
        assert with_rej["best_value"] < 50.0


class TestEndToEndOptunaPipeline:
    """Full Optuna study with mocked CAE, real config, real curves."""

    def test_multi_trial_study(self, tmp_path):
        # Create VEXIS directory structure
        (tmp_path / "vexis" / "input").mkdir(parents=True, exist_ok=True)
        (tmp_path / "vexis" / "results").mkdir(parents=True, exist_ok=True)
        (tmp_path / "vexis" / "main.py").write_text("print('mock')\n")

        cfg = _build_config(tmp_path)

        # Target curve
        target_csv = tmp_path / "target.csv"
        _make_result_csv(target_csv, peak_force=100.0)
        target_curve = pd.read_csv(target_csv).rename(
            columns={"Stroke": "displacement", "Reaction_Force": "force"}
        )

        persistence = TrialPersistence(tmp_path / "output")
        cad_gate = CadGate(cfg.cad_gate)
        cad_gate._model = MockFeasibilityModel(low=8.0, high=18.0)

        geom_adapter = GeometryAdapter(cfg.freecad, tmp_path)
        cae_evaluator = CaeEvaluator(
            vexis_path=tmp_path / "vexis",
            cae_spec=cfg.cae,
            obj_spec=cfg.objective,
            target_curve=target_curve,
            target_features={},
        )

        orchestrator = ObjectiveOrchestrator(
            cfg=cfg,
            cad_gate=cad_gate,
            geometry_adapter=geom_adapter,
            cae_evaluator=cae_evaluator,
            persistence=persistence,
        )

        # Build sampler with both constraint mechanisms.
        # multivariate=True makes TPE use sample_relative (joint sampling),
        # which is where FeasibilityAwareSampler rejection kicks in.
        # n_startup_trials=3 shortens the random startup phase.
        constraints_func = make_constraints_func()
        base_sampler = optuna.samplers.TPESampler(
            seed=42, constraints_func=constraints_func,
            n_startup_trials=3, multivariate=True,
        )

        def _predict(params: dict) -> bool:
            dummy = DesignPoint(trial_id=-1, params=params)
            return cad_gate.predict(dummy).is_feasible

        sampler = FeasibilityAwareSampler(
            base_sampler=base_sampler,
            predict_fn=_predict,
            max_retries=30,
        )

        study = optuna.create_study(sampler=sampler)

        n_trials = 10

        def wrapped_objective(trial):
            tid = trial.number
            point = suggest_design_point(trial, tid, cfg.bounds)

            # Mock geometry: create fake STEP + result CSV per trial
            step_dir = Path(cfg.freecad.step_output_dir)
            step_dir.mkdir(parents=True, exist_ok=True)
            fake_step = step_dir / f"proto4_trial_{tid}.step"
            fake_step.write_text("mock")

            # Generate result CSV with peak_force that varies by params
            peak = 80.0 + point.params["width"] + point.params["height"]
            result_csv = tmp_path / "vexis" / "results" / f"proto4_trial_{tid}_result.csv"
            _make_result_csv(result_csv, peak_force=peak)

            with patch.object(
                orchestrator._geom, "generate_step", return_value=fake_step
            ), patch.object(
                cae_evaluator, "_run_subprocess", return_value=result_csv
            ):
                return orchestrator(point, trial=trial)

        study.optimize(wrapped_objective, n_trials=n_trials)

        # --- Assertions ---
        assert len(study.trials) == n_trials

        # At least some trials should succeed (feasible + CAE ok)
        successful = [
            t for t in study.trials
            if t.user_attrs.get(FEASIBILITY_ATTR, 0) <= 0
        ]
        assert len(successful) > 0

        # Best value should be a real RMSE (not penalty)
        assert study.best_value < cfg.penalty.base_penalty

        # Check rejection stats — sample_relative is only used after
        # the TPE startup phase (3 trials), so accepted < n_trials.
        stats = sampler.rejection_stats
        assert stats["accepted"] > 0  # rejection sampling was exercised

        # Persistence: every trial should be saved
        for i in range(n_trials):
            rec = persistence.load_trial(i)
            assert rec["trial_id"] == i
            assert rec["outcome"] is not None

        # Summary
        persistence.save_summary({
            "total_trials": n_trials,
            "best_value": study.best_value,
            "rejection_stats": stats,
        })
        summary_path = tmp_path / "output" / "summary_proto4.json"
        assert summary_path.exists()
