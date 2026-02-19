"""Tests for proto4-claude persistence module."""

import json

from proto4_claude.persistence import TrialPersistence
from proto4_claude.types import (
    CadFeasibilityResult,
    CaeResult,
    CaeStatus,
    DesignPoint,
    TrialOutcome,
    TrialRecord,
)


class TestTrialPersistence:
    def test_save_and_load_trial(self, tmp_path):
        persist = TrialPersistence(tmp_path)
        rec = TrialRecord(
            trial_id=7,
            design_point=DesignPoint(trial_id=7, params={"w": 12.0}),
            feasibility=CadFeasibilityResult(is_feasible=True, confidence=0.9),
            cae_result=CaeResult(
                status=CaeStatus.SUCCESS, metrics={"rmse": 0.05}
            ),
            objective_values={"rmse": 0.05},
            outcome=TrialOutcome.CAE_SUCCESS,
            wall_clock_sec=15.3,
        )
        path = persist.save_trial(rec)
        assert path.exists()

        loaded = persist.load_trial(7)
        assert loaded["trial_id"] == 7
        assert loaded["outcome"] == "cae_success"
        assert loaded["feasibility"]["confidence"] == 0.9
        assert loaded["cae_result"]["metrics"]["rmse"] == 0.05

    def test_save_summary(self, tmp_path):
        persist = TrialPersistence(tmp_path)
        summary = {"total_trials": 20, "best_value": 0.01}
        path = persist.save_summary(summary)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["best_value"] == 0.01

    def test_save_run_config(self, tmp_path):
        persist = TrialPersistence(tmp_path)
        cfg = {"sampler": "TPE", "seed": 42}
        path = persist.save_run_config(cfg)
        assert path.exists()

    def test_load_nonexistent_trial_raises(self, tmp_path):
        persist = TrialPersistence(tmp_path)
        import pytest
        with pytest.raises(FileNotFoundError):
            persist.load_trial(999)
