"""
v1.0 CAD Feasibility Gate

Adapter to an ML model that predicts whether a given DesignPoint will
produce a valid CAD geometry or break during STEP generation.

Supports two loading modes:
  1. Directory path → load ai-v0 SafetyPredictor (model.joblib + scaler.joblib)
  2. File path → load raw joblib model (legacy)

If no model is loaded the gate defaults to *always feasible* so the
pipeline can run without a trained predictor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from .config import CadGateSpec
from .types import CadFeasibilityResult, DesignPoint

logger = logging.getLogger(__name__)


# ai-v0 SafetyPredictor feature order (20 dimensions)
AI_V0_FEATURE_NAMES = [
    "CROWN-D-L", "CROWN-D-H", "CROWN-W", "PUSHER-D-H", "PUSHER-D-L",
    "TIP-D", "STROKE-OUT", "STROKE-CENTER", "FOOT-W", "FOOT-MID",
    "SHOULDER-ANGLE-OUT", "SHOULDER-ANGLE-IN", "TOP-T", "TOP-DROP",
    "FOOT-IN", "DIAMETER", "HEIGHT", "TIP-DROP", "SHOUDER-T", "FOOT-OUT"
]


class SafetyPredictorWrapper:
    """Wrapper for ai-v0 SafetyPredictor model directory."""

    def __init__(self, model_dir: Path) -> None:
        import joblib
        import numpy as np
        
        self._np = np
        self._model = joblib.load(model_dir / "model.joblib")
        self._scaler = joblib.load(model_dir / "scaler.joblib")
        self._feature_names = AI_V0_FEATURE_NAMES
        logger.info("SafetyPredictor loaded from %s", model_dir)

    def predict_proba(self, features: List[List[float]]) -> Any:
        """Return (N, 2) array of [unsafe_prob, safe_prob]."""
        import numpy as np
        arr = np.atleast_2d(np.asarray(features, dtype=np.float64))
        scaled = self._scaler.transform(arr)
        return self._model.predict_proba(scaled)


class CadGate:
    """Predict CAD feasibility for a DesignPoint."""

    def __init__(self, spec: CadGateSpec) -> None:
        self._spec = spec
        self._model: Any = None
        self._is_safety_predictor = False
        self._load_error: str | None = None

        if not spec.enabled:
            logger.info("CAD gate disabled; all points will be accepted")
            return

        if spec.model_path:
            self._model = self._load_model(Path(spec.model_path))
        else:
            logger.info("No CAD gate model path; gate passes all points")

    def _load_model(self, path: Path) -> Any:
        """Load ML model.
        
        If path is a directory containing model.joblib + scaler.joblib,
        load as ai-v0 SafetyPredictor.
        Otherwise, load as raw joblib model.
        """
        if not path.exists():
            logger.warning("CAD gate model not found: %s — gate disabled", path)
            return None

        try:
            # Check if it's a SafetyPredictor directory
            if path.is_dir():
                model_file = path / "model.joblib"
                scaler_file = path / "scaler.joblib"
                if model_file.exists() and scaler_file.exists():
                    self._is_safety_predictor = True
                    return SafetyPredictorWrapper(path)
                else:
                    logger.warning(
                        "Directory %s missing model.joblib or scaler.joblib — gate disabled",
                        path
                    )
                    return None
            
            # Legacy: load single joblib file
            import joblib
            model = joblib.load(path)
            logger.info("CAD gate model loaded: %s", path)
            return model
        except ModuleNotFoundError as exc:
            self._load_error = f"missing_dependency:{exc.name or 'unknown'}"
            logger.warning(
                "Failed to load CAD gate model due to missing dependency (%s) — gate disabled",
                exc.name or str(exc),
            )
            return None
        except Exception as exc:
            self._load_error = f"load_error:{exc.__class__.__name__}: {exc}"
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
        """Convert DesignPoint params dict to an ordered feature vector.
        
        For SafetyPredictor: expects 20-dim ratio vector in AI_V0_FEATURE_NAMES order.
        For legacy models: sorted param keys.
        """
        if self._is_safety_predictor:
            # ai-v0 expects features in specific order
            return [point.params.get(k, 1.0) for k in AI_V0_FEATURE_NAMES]
        return [point.params[k] for k in sorted(point.params)]
