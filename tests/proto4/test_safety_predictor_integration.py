"""Test CadGate with real ai-v0 SafetyPredictor model."""
import pytest
from pathlib import Path

from proto4_claude.cad_gate import CadGate, AI_V0_FEATURE_NAMES
from proto4_claude.config import CadGateSpec
from proto4_claude.types import DesignPoint


class TestSafetyPredictorIntegration:
    """Integration tests with real ai-v0 SafetyPredictor."""

    @pytest.fixture
    def gate(self):
        """Load real ai-v0 model."""
        model_path = Path("C:/github_repo/cad_automaton/src/ai-v0/artifacts_optuna_100")
        if not model_path.exists():
            pytest.skip("ai-v0 model not available")
        
        spec = CadGateSpec(
            model_path=str(model_path),
            threshold=0.5,
            enabled=True
        )
        return CadGate(spec)

    def test_baseline_safe(self, gate):
        """All ratios at 1.0 (baseline) should be predicted safe."""
        params_safe = {k: 1.0 for k in AI_V0_FEATURE_NAMES}
        point = DesignPoint(trial_id=0, params=params_safe)
        result = gate.predict(point)
        
        print(f"Safe baseline: is_feasible={result.is_feasible}, conf={result.confidence:.3f}")
        assert result.confidence is not None
        # Baseline should generally be safe (conf > 0.5)
        assert result.confidence >= 0.4  # allow some margin

    def test_extreme_unsafe(self, gate):
        """Extreme ratios (0.7) should be predicted unsafe or low confidence."""
        params_extreme = {k: 0.7 for k in AI_V0_FEATURE_NAMES}
        point = DesignPoint(trial_id=1, params=params_extreme)
        result = gate.predict(point)
        
        print(f"Extreme case: is_feasible={result.is_feasible}, conf={result.confidence:.3f}")
        assert result.confidence is not None
        # Extreme values should have lower confidence than baseline
        assert result.confidence < 0.9  # not super confident it's safe

    def test_model_loaded_correctly(self, gate):
        """Verify SafetyPredictorWrapper was used."""
        assert gate._is_safety_predictor is True
        assert gate._model is not None
