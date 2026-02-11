"""Tests for proto4-codex CAE stdout logging controls."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from proto4_codex.cae_evaluator import CaeEvaluator
from proto4_codex.config import CaeSpec, ObjectiveSpec
from proto4_codex.types import DesignPoint


def test_vexis_stdout_log_written(tmp_path: Path):
    vexis_root = tmp_path / "vexis"
    (vexis_root / "input").mkdir(parents=True)
    (vexis_root / "results").mkdir(parents=True)
    (vexis_root / "main.py").write_text("print('mock')\n")

    target_curve = pd.DataFrame({"displacement": [0.0, 0.5], "force": [0.0, 1.0]})

    cae = CaeSpec(
        stroke_range_min=0.0,
        stroke_range_max=0.5,
        solver_progress_stall_sec=60,
        max_retries=1,
        stream_stdout=True,
        stdout_log_dir=str(tmp_path / "logs"),
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

    step_path = tmp_path / "dummy.step"
    step_path.write_text("x")

    result_csv = vexis_root / "results" / "proto4_trial_0_result.csv"
    pd.DataFrame({"Stroke": [0.0, 0.5], "Reaction_Force": [0.0, 1.0]}).to_csv(result_csv, index=False)

    with patch.object(evaluator, "_run_subprocess", return_value=result_csv):
        evaluator.evaluate(step_path, DesignPoint(trial_id=0, params={"x": 1.0}))

    # ensure log dir exists even if subprocess is mocked
    assert (tmp_path / "logs").exists()
