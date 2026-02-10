"""Regression tests for proto4-codex objective error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import optuna

from proto4_codex.cad_gate import CadGate
from proto4_codex.config import (
    BoundsSpec,
    CadGateSpec,
    CaeSpec,
    FreecadSpec,
    ObjectiveSpec,
    PenaltySpec,
    Proto4Config,
)
from proto4_codex.geometry_adapter import GeometryAdapter
from proto4_codex.objective import FAILURE_REASON_ATTR, FAILURE_STAGE_ATTR, ObjectiveOrchestrator
from proto4_codex.persistence import TrialPersistence
from proto4_codex.search_space import FEASIBILITY_ATTR
from proto4_codex.types import CaeResult, CaeStatus, DesignPoint


class _AlwaysFeasibleModel:
    def predict_proba(self, features):
        return [[0.01, 0.99] for _ in features]


class _FixedCaeEvaluator:
    def __init__(self, result: CaeResult) -> None:
        self._result = result

    def evaluate(self, step_path: Path, point: DesignPoint) -> CaeResult:
        return self._result


class _SequenceCaeEvaluator:
    def __init__(self) -> None:
        self._calls = 0

    def evaluate(self, step_path: Path, point: DesignPoint) -> CaeResult:
        self._calls += 1
        if self._calls == 1:
            return CaeResult(status=CaeStatus.FAIL, failure_reason="process_exit_1")
        return CaeResult(status=CaeStatus.SUCCESS, metrics={"rmse": 0.123})


def _build_cfg(tmp_path: Path) -> Proto4Config:
    return Proto4Config(
        objective=ObjectiveSpec(type="rmse", weights={"rmse": 1.0}, features={}),
        freecad=FreecadSpec(
            constraints={"x": {"min": 0.0, "max": 2.0}},
            step_output_dir=str(tmp_path / "step"),
        ),
        cad_gate=CadGateSpec(enabled=True, threshold=0.5),
        cae=CaeSpec(stroke_range_min=0.0, stroke_range_max=0.5, max_retries=1),
        penalty=PenaltySpec(
            base_penalty=50.0,
            alpha=10.0,
            failure_weights={"cae_fail": 1.0, "cad_infeasible": 1.0},
        ),
        bounds=[BoundsSpec(name="x", min=0.0, max=2.0)],
    )


def _build_orchestrator(
    tmp_path: Path,
    cfg: Proto4Config,
    cae_evaluator,
) -> tuple[ObjectiveOrchestrator, GeometryAdapter, TrialPersistence]:
    persistence = TrialPersistence(tmp_path / "output")
    cad_gate = CadGate(cfg.cad_gate)
    cad_gate._model = _AlwaysFeasibleModel()
    geometry_adapter = GeometryAdapter(cfg.freecad, tmp_path)
    orchestrator = ObjectiveOrchestrator(
        cfg=cfg,
        cad_gate=cad_gate,
        geometry_adapter=geometry_adapter,
        cae_evaluator=cae_evaluator,
        persistence=persistence,
    )
    return orchestrator, geometry_adapter, persistence


def test_cae_failure_is_marked_infeasible_for_sampler(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    cae_evaluator = _FixedCaeEvaluator(
        CaeResult(status=CaeStatus.FAIL, failure_reason="process_exit_1")
    )
    orchestrator, geometry_adapter, persistence = _build_orchestrator(
        tmp_path,
        cfg,
        cae_evaluator,
    )

    fake_step = tmp_path / "step" / "proto4_trial_0.step"
    fake_step.parent.mkdir(parents=True, exist_ok=True)
    fake_step.write_text("dummy", encoding="utf-8")

    study = optuna.create_study()
    point = DesignPoint(trial_id=0, params={"x": 1.0})

    with patch.object(geometry_adapter, "generate_step", return_value=fake_step):
        def objective(trial):
            trial.suggest_float("x", 0.0, 2.0)
            return orchestrator(point, trial=trial)

        study.optimize(objective, n_trials=1)

    frozen = study.trials[0]
    assert frozen.user_attrs[FEASIBILITY_ATTR] > 0
    assert frozen.user_attrs[FAILURE_STAGE_ATTR] == "cae_evaluation"
    assert "process_exit_1" in frozen.user_attrs[FAILURE_REASON_ATTR]
    assert frozen.value >= cfg.penalty.base_penalty

    rec = persistence.load_trial(0)
    assert rec["outcome"] == "cae_fail"
    assert rec["cae_result"]["failure_reason"] == "process_exit_1"


def test_optuna_continues_after_cae_failure(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path)
    cae_evaluator = _SequenceCaeEvaluator()
    orchestrator, geometry_adapter, persistence = _build_orchestrator(
        tmp_path,
        cfg,
        cae_evaluator,
    )

    def _fake_step(point: DesignPoint) -> Path:
        out = tmp_path / "step" / f"proto4_trial_{point.trial_id}.step"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("dummy", encoding="utf-8")
        return out

    study = optuna.create_study()
    with patch.object(geometry_adapter, "generate_step", side_effect=_fake_step):
        def objective(trial):
            x = trial.suggest_float("x", 0.0, 2.0)
            point = DesignPoint(trial_id=trial.number, params={"x": x})
            return orchestrator(point, trial=trial)

        study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert study.trials[0].user_attrs[FEASIBILITY_ATTR] > 0
    assert study.trials[1].user_attrs[FEASIBILITY_ATTR] <= 0
    assert study.best_value < cfg.penalty.base_penalty

    rec0 = persistence.load_trial(0)
    rec1 = persistence.load_trial(1)
    assert rec0["outcome"] == "cae_fail"
    assert rec1["outcome"] == "cae_success"
