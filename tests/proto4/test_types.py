"""Tests for proto4-claude types module."""

from proto4_claude.types import (
    CaeResult,
    CaeStatus,
    CadFeasibilityResult,
    DesignPoint,
    TrialOutcome,
    TrialRecord,
)


class TestDesignPoint:
    def test_creation(self):
        dp = DesignPoint(trial_id=0, params={"width": 15.0, "height": 8.0})
        assert dp.trial_id == 0
        assert dp.params["width"] == 15.0

    def test_to_dict(self):
        dp = DesignPoint(trial_id=1, params={"a": 1.0})
        d = dp.to_dict()
        assert d["trial_id"] == 1
        assert d["params"] == {"a": 1.0}


class TestCadFeasibilityResult:
    def test_feasible(self):
        r = CadFeasibilityResult(is_feasible=True, confidence=0.9)
        assert r.is_feasible
        assert r.confidence == 0.9

    def test_infeasible_with_reason(self):
        r = CadFeasibilityResult(
            is_feasible=False, confidence=0.3, reason_code="ml_infeasible"
        )
        d = r.to_dict()
        assert not d["is_feasible"]
        assert d["reason_code"] == "ml_infeasible"


class TestCaeResult:
    def test_success(self):
        r = CaeResult(
            status=CaeStatus.SUCCESS,
            metrics={"rmse": 0.05, "rmse_loading": 0.04},
            runtime_sec=120.5,
        )
        assert r.status == CaeStatus.SUCCESS
        assert r.metrics["rmse"] == 0.05

    def test_fail(self):
        r = CaeResult(status=CaeStatus.FAIL)
        d = r.to_dict()
        assert d["status"] == "FAIL"
        assert d["metrics"] == {}


class TestTrialRecord:
    def test_full_record(self):
        dp = DesignPoint(trial_id=5, params={"x": 1.0})
        feas = CadFeasibilityResult(is_feasible=True, confidence=0.95)
        cae = CaeResult(status=CaeStatus.SUCCESS, metrics={"rmse": 0.01})
        rec = TrialRecord(
            trial_id=5,
            design_point=dp,
            feasibility=feas,
            cae_result=cae,
            objective_values={"rmse": 0.01},
            outcome=TrialOutcome.CAE_SUCCESS,
            wall_clock_sec=30.0,
        )
        d = rec.to_dict()
        assert d["trial_id"] == 5
        assert d["outcome"] == "cae_success"
        assert "feasibility" in d
        assert "cae_result" in d
        assert d["objective_values"]["rmse"] == 0.01
