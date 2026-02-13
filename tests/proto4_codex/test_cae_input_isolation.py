"""Tests for single-STEP input isolation in proto4-codex CAE evaluator."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from proto4_codex.cae_evaluator import CaeEvaluator
from proto4_codex.config import CaeSpec, ObjectiveSpec
from proto4_codex.types import CaeStatus, DesignPoint


def _build_evaluator(tmp_path: Path) -> tuple[CaeEvaluator, Path]:
    vexis_root = tmp_path / "vexis"
    (vexis_root / "input").mkdir(parents=True)
    (vexis_root / "results").mkdir(parents=True)
    (vexis_root / "main.py").write_text("print('mock')\n", encoding="utf-8")

    target_curve = pd.DataFrame({"displacement": [0.0, 0.5], "force": [0.0, 1.0]})
    cae = CaeSpec(
        stroke_range_min=0.0,
        stroke_range_max=0.5,
        solver_progress_stall_sec=60,
        max_retries=1,
        stream_stdout=False,
        stdout_log_dir=None,
        stdout_console_level="INFO",
    )
    obj = ObjectiveSpec(type="rmse", weights={"rmse": 1.0}, features={})
    evaluator = CaeEvaluator(
        vexis_path=vexis_root,
        cae_spec=cae,
        obj_spec=obj,
        target_curve=target_curve,
        target_features={},
    )
    return evaluator, vexis_root


def test_evaluate_uses_only_one_step_file_and_restores_previous_files(tmp_path: Path) -> None:
    evaluator, vexis_root = _build_evaluator(tmp_path)
    input_dir = vexis_root / "input"

    (input_dir / "keep_a.step").write_text("A", encoding="utf-8")
    (input_dir / "keep_b.step").write_text("B", encoding="utf-8")
    trial_step = tmp_path / "trial.step"
    trial_step.write_text("TRIAL", encoding="utf-8")

    result_csv = vexis_root / "results" / "proto4_trial_0_result.csv"

    seen_step_files: list[str] = []

    def _fake_run_subprocess(_job_name: str) -> Path:
        seen_step_files[:] = sorted(p.name for p in input_dir.glob("*.step"))
        pd.DataFrame({"Stroke": [0.0, 0.5], "Reaction_Force": [0.0, 1.0]}).to_csv(result_csv, index=False)
        return result_csv

    with patch.object(evaluator, "_run_subprocess", side_effect=_fake_run_subprocess):
        result = evaluator.evaluate(trial_step, DesignPoint(trial_id=0, params={"x": 1.0}))

    assert result.status == CaeStatus.SUCCESS
    assert seen_step_files == ["proto4_trial_0.step"]
    assert sorted(p.name for p in input_dir.glob("*.step")) == ["keep_a.step", "keep_b.step"]


def test_evaluate_recovers_orphaned_step_stash(tmp_path: Path) -> None:
    evaluator, vexis_root = _build_evaluator(tmp_path)
    input_dir = vexis_root / "input"

    orphan_dir = input_dir / ".optuna_step_stash" / "orphan_run"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "orphan.step").write_text("ORPHAN", encoding="utf-8")

    trial_step = tmp_path / "trial.step"
    trial_step.write_text("TRIAL", encoding="utf-8")

    result_csv = vexis_root / "results" / "proto4_trial_1_result.csv"

    def _fake_run_subprocess(_job_name: str) -> Path:
        pd.DataFrame({"Stroke": [0.0, 0.5], "Reaction_Force": [0.0, 1.0]}).to_csv(result_csv, index=False)
        return result_csv

    with patch.object(evaluator, "_run_subprocess", side_effect=_fake_run_subprocess):
        result = evaluator.evaluate(trial_step, DesignPoint(trial_id=1, params={"x": 1.0}))

    assert result.status == CaeStatus.SUCCESS
    assert (input_dir / "orphan.step").exists()
    assert not (input_dir / ".optuna_step_stash").exists()
