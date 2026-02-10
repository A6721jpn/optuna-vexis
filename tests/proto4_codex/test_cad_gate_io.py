"""Tests for proto4-codex CAD gate IO."""

from proto4_codex.cad_gate import CadGate
from proto4_codex.config import CadGateSpec
from proto4_codex.types import DesignPoint


def test_missing_model_path_disables_gate(tmp_path):
    spec = CadGateSpec(enabled=True, model_path=str(tmp_path / "missing"))
    gate = CadGate(spec)
    pt = DesignPoint(trial_id=1, params={"a": 1.0})
    result = gate.predict(pt)
    assert result.is_feasible
    assert result.reason_code == "gate_disabled"


def test_model_dir_requires_both_files(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.joblib").write_text("x")
    # missing scaler.joblib
    spec = CadGateSpec(enabled=True, model_path=str(model_dir))
    gate = CadGate(spec)
    pt = DesignPoint(trial_id=1, params={"a": 1.0})
    result = gate.predict(pt)
    assert result.is_feasible
    assert result.reason_code == "gate_disabled"
