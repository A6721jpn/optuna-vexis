"""
Proto4 CAD Feasibility Gate

Adapter to an ML model that predicts whether a given DesignPoint will
produce a valid CAD geometry or break during STEP generation.

If no model is loaded the gate defaults to *always feasible* so the
pipeline can run without a trained predictor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .config import CadGateSpec
from .types import CadFeasibilityResult, DesignPoint

logger = logging.getLogger(__name__)


class CadGate:
    """Predict CAD feasibility for a DesignPoint."""

    def __init__(self, spec: CadGateSpec) -> None:
        self._spec = spec
        self._model: Any = None

        if not spec.enabled:
            logger.info("CAD gate disabled; all points will be accepted")
            return

        if spec.model_path:
            self._model = self._load_model(Path(spec.model_path))
        else:
            logger.info("No CAD gate model path; gate passes all points")

    def _load_model(self, path: Path) -> Any:
        """Load a serialised ML model (joblib / pickle / ONNX).

        Returns *None* and logs a warning when the file is missing so
        the pipeline can proceed without a gate.
        """
        if not path.exists():
            logger.warning("CAD gate model not found: %s — gate disabled", path)
            return None

        try:
            import joblib
            model = joblib.load(path)
            logger.info("CAD gate model loaded: %s", path)
            return model
        except Exception as exc:
            logger.warning("Failed to load CAD gate model: %s — gate disabled", exc)
            return None

    def predict(self, point: DesignPoint) -> CadFeasibilityResult:
        """Return feasibility prediction for *point*."""
        if not self._spec.enabled or self._model is None:
            return CadFeasibilityResult(
                is_feasible=True,
                confidence=None,
                reason_code="gate_disabled",
            )

        try:
            feature_vec = self._design_point_to_features(point)
            proba = float(self._model.predict_proba([feature_vec])[0][1])
            is_feasible = proba >= self._spec.threshold

            return CadFeasibilityResult(
                is_feasible=is_feasible,
                confidence=proba,
                reason_code=None if is_feasible else "ml_infeasible",
                metadata={"threshold": self._spec.threshold},
            )
        except Exception as exc:
            logger.warning("CAD gate prediction error: %s — defaulting to feasible", exc)
            return CadFeasibilityResult(
                is_feasible=True,
                confidence=None,
                reason_code="prediction_error",
                metadata={"error": str(exc)},
            )

    def _design_point_to_features(self, point: DesignPoint) -> list[float]:
        """Convert DesignPoint params dict to an ordered feature vector."""
        return [point.params[k] for k in sorted(point.params)]
