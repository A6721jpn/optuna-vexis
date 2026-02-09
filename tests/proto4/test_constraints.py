"""Tests for proto4-claude constraints module."""

from proto4_claude.config import BoundsSpec, PenaltySpec
from proto4_claude.constraints import (
    check_hard_constraints,
    distance_from_bounds,
    penalty_value,
)
from proto4_claude.types import DesignPoint, TrialOutcome

BOUNDS = [
    BoundsSpec(name="width", min=10.0, max=20.0),
    BoundsSpec(name="height", min=5.0, max=12.0),
]
PENALTY = PenaltySpec(
    base_penalty=50.0,
    alpha=10.0,
    failure_weights={
        "cad_infeasible": 1.0,
        "cae_fail": 0.6,
    },
)


class TestDistanceFromBounds:
    def test_inside_bounds(self):
        pt = DesignPoint(trial_id=0, params={"width": 15.0, "height": 8.0})
        assert distance_from_bounds(pt, BOUNDS) == 0.0

    def test_on_boundary(self):
        pt = DesignPoint(trial_id=0, params={"width": 10.0, "height": 12.0})
        assert distance_from_bounds(pt, BOUNDS) == 0.0

    def test_outside_one_param(self):
        pt = DesignPoint(trial_id=0, params={"width": 25.0, "height": 8.0})
        assert distance_from_bounds(pt, BOUNDS) == 5.0  # 25 - 20

    def test_outside_both_params(self):
        pt = DesignPoint(trial_id=0, params={"width": 25.0, "height": 3.0})
        # (25-20) + (5-3) = 7
        assert distance_from_bounds(pt, BOUNDS) == 7.0

    def test_below_min(self):
        pt = DesignPoint(trial_id=0, params={"width": 7.0, "height": 8.0})
        assert distance_from_bounds(pt, BOUNDS) == 3.0  # 10 - 7


class TestHardConstraints:
    def test_feasible(self):
        pt = DesignPoint(trial_id=0, params={"width": 15.0, "height": 8.0})
        assert check_hard_constraints(pt, BOUNDS) is None

    def test_violation_detected(self):
        pt = DesignPoint(trial_id=0, params={"width": 25.0, "height": 8.0})
        reason = check_hard_constraints(pt, BOUNDS)
        assert reason is not None
        assert "width" in reason


class TestPenaltyValue:
    def test_inside_bounds_base_penalty_only(self):
        pt = DesignPoint(trial_id=0, params={"width": 15.0, "height": 8.0})
        p = penalty_value(pt, BOUNDS, PENALTY, TrialOutcome.CAD_INFEASIBLE)
        # weight=1.0, distance=0 → 1.0 * (50 + 10*0) = 50
        assert p == 50.0

    def test_outside_bounds_adds_distance(self):
        pt = DesignPoint(trial_id=0, params={"width": 25.0, "height": 8.0})
        p = penalty_value(pt, BOUNDS, PENALTY, TrialOutcome.CAD_INFEASIBLE)
        # weight=1.0, distance=5 → 1.0 * (50 + 10*5) = 100
        assert p == 100.0

    def test_failure_weight_applied(self):
        pt = DesignPoint(trial_id=0, params={"width": 15.0, "height": 8.0})
        p = penalty_value(pt, BOUNDS, PENALTY, TrialOutcome.CAE_FAIL)
        # weight=0.6, distance=0 → 0.6 * 50 = 30
        assert p == 30.0

    def test_outside_worse_than_inside(self):
        inside = DesignPoint(trial_id=0, params={"width": 15.0, "height": 8.0})
        outside = DesignPoint(trial_id=1, params={"width": 25.0, "height": 3.0})
        p_in = penalty_value(inside, BOUNDS, PENALTY, TrialOutcome.CAD_INFEASIBLE)
        p_out = penalty_value(outside, BOUNDS, PENALTY, TrialOutcome.CAD_INFEASIBLE)
        assert p_out > p_in
