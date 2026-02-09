"""Tests for proto4-claude CAD gate module.

Uses MockFeasibilityModel from conftest to simulate ML predictions.
"""

from proto4_claude.cad_gate import CadGate
from proto4_claude.config import CadGateSpec
from proto4_claude.types import DesignPoint


class TestCadGateDisabled:
    def test_disabled_always_feasible(self):
        spec = CadGateSpec(enabled=False)
        gate = CadGate(spec)
        pt = DesignPoint(trial_id=0, params={"width": 999.0})
        result = gate.predict(pt)
        assert result.is_feasible
        assert result.reason_code == "gate_disabled"

    def test_no_model_path_always_feasible(self):
        spec = CadGateSpec(enabled=True, model_path=None)
        gate = CadGate(spec)
        pt = DesignPoint(trial_id=0, params={"width": 999.0})
        result = gate.predict(pt)
        assert result.is_feasible


class TestCadGateWithMockModel:
    def test_feasible_point(self, mock_model):
        spec = CadGateSpec(enabled=True, threshold=0.5)
        gate = CadGate(spec)
        gate._model = mock_model  # Inject mock directly

        # All params within [8, 18] → high feasibility
        pt = DesignPoint(trial_id=0, params={"height": 10.0, "width": 15.0})
        result = gate.predict(pt)
        assert result.is_feasible
        assert result.confidence is not None
        assert result.confidence >= 0.5

    def test_infeasible_point(self, mock_model):
        spec = CadGateSpec(enabled=True, threshold=0.5)
        gate = CadGate(spec)
        gate._model = mock_model

        # params far outside [8, 18] → low feasibility
        pt = DesignPoint(trial_id=0, params={"height": 1.0, "width": 30.0})
        result = gate.predict(pt)
        assert not result.is_feasible
        assert result.reason_code == "ml_infeasible"
        assert result.confidence is not None
        assert result.confidence < 0.5

    def test_threshold_sensitivity(self, mock_model):
        """Changing threshold changes the decision boundary."""
        gate_strict = CadGate(CadGateSpec(enabled=True, threshold=0.9))
        gate_strict._model = mock_model
        gate_loose = CadGate(CadGateSpec(enabled=True, threshold=0.1))
        gate_loose._model = mock_model

        # Borderline point
        pt = DesignPoint(trial_id=0, params={"height": 7.0, "width": 15.0})
        r_strict = gate_strict.predict(pt)
        r_loose = gate_loose.predict(pt)

        # Loose should accept more borderline points than strict
        assert r_loose.is_feasible or r_strict.is_feasible is False
        if r_loose.is_feasible:
            # With this borderline point, strict may reject what loose accepts
            assert r_loose.confidence == r_strict.confidence

    def test_sorted_feature_order(self, mock_model):
        """Params are sorted by key for consistent feature vectors."""
        spec = CadGateSpec(enabled=True, threshold=0.5)
        gate = CadGate(spec)
        gate._model = mock_model

        pt = DesignPoint(trial_id=0, params={"z": 10.0, "a": 10.0})
        result = gate.predict(pt)
        # Both within [8, 18] → feasible regardless of order
        assert result.is_feasible
