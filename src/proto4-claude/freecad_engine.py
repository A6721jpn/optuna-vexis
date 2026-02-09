"""
Proto4 FreeCAD Engine

Core FreeCAD operations extracted from proto3-hybrid/hybrid_solver.py.
Provides headless FreeCAD execution for constraint updates and STEP export.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


# Lazy FreeCAD import with environment setup
_FreeCAD = None


def _get_freecad():
    """Lazy-load FreeCAD module with environment fallback."""
    global _FreeCAD
    if _FreeCAD is not None:
        return _FreeCAD

    try:
        import FreeCAD  # type: ignore
        _FreeCAD = FreeCAD
        return _FreeCAD
    except ImportError:
        pass

    # Try to find FreeCAD in conda environments
    conda_prefix = os.environ.get(
        "CONDA_PREFIX",
        r"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad",
    )
    env_candidates = []
    if conda_prefix:
        env_candidates.append(Path(conda_prefix))
    for name in ("fcad", "fcad-codex", "b123d"):
        env_candidates.append(Path(r"C:\Users\aokuni\AppData\Local\miniforge3\envs") / name)

    for env_path in env_candidates:
        freecad_bin = env_path / "Library" / "bin"
        freecad_lib = env_path / "Library" / "lib"
        if not freecad_bin.exists():
            continue
        os.environ["PATH"] = str(freecad_bin) + os.pathsep + os.environ.get("PATH", "")
        sys.path.insert(0, str(freecad_bin))
        if freecad_lib.exists():
            sys.path.insert(0, str(freecad_lib))
        try:
            import FreeCAD  # type: ignore
            _FreeCAD = FreeCAD
            logger.info("FreeCAD loaded from %s", env_path)
            return _FreeCAD
        except ImportError:
            continue

    raise ImportError("FreeCAD not found in any conda environment")


@dataclass
class ConstraintSpec:
    """Specification for a sketch constraint."""
    index: int
    name: str
    ctype: str  # "Distance", "DistanceX", "DistanceY", "Angle"
    base_value: float
    sketch: Optional[Any] = None
    angle_unit: Optional[str] = None  # "rad" or "deg"


class FreecadEngine:
    """Execute FreeCAD operations for constraint updates and STEP export."""

    def __init__(
        self,
        fcstd_path: Path,
        sketch_name: Optional[str] = None,
        surface_name: Optional[str] = None,
        surface_label: Optional[str] = None,
    ) -> None:
        self._fcstd_path = Path(fcstd_path)
        self._sketch_name = sketch_name
        self._surface_name = surface_name or "Face"
        self._surface_label = surface_label or "SURFACE"
        self._doc = None
        self._sketch = None
        self._surface = None
        self._specs: List[ConstraintSpec] = []

    def open(self) -> None:
        """Open FreeCAD document and find sketch/surface."""
        FreeCAD = _get_freecad()

        if not self._fcstd_path.exists():
            raise FileNotFoundError(f"FCStd not found: {self._fcstd_path}")

        self._doc = FreeCAD.openDocument(str(self._fcstd_path))
        self._sketch = self._find_sketch(self._doc, self._sketch_name)
        self._surface = self._find_surface(self._doc, self._surface_name, self._surface_label)
        logger.info("Opened FreeCAD document: %s", self._fcstd_path)

    def close(self) -> None:
        """Close FreeCAD document."""
        FreeCAD = _get_freecad()
        if self._doc is not None:
            try:
                FreeCAD.closeDocument(self._doc.Name)
            except Exception:
                pass
            self._doc = None
            self._sketch = None
            self._surface = None

    def set_constraints(self, specs: List[ConstraintSpec]) -> None:
        """Set the constraint specifications to use."""
        self._specs = specs
        for spec in specs:
            spec.sketch = self._sketch

    def apply_ratios(self, ratios: dict[str, float]) -> bool:
        """Apply ratio values to constraints and recompute.
        
        Args:
            ratios: {constraint_name: ratio_value} - ratio to base_value
            
        Returns:
            True if geometry is valid, False otherwise.
        """
        if self._doc is None or self._sketch is None:
            raise RuntimeError("Document not opened. Call open() first.")

        # Apply each constraint
        for spec in self._specs:
            ratio = ratios.get(spec.name, 1.0)
            candidate = self._clamp_candidate(spec.ctype, spec.base_value * ratio, spec.angle_unit)
            if candidate is None:
                logger.debug("Constraint %s: invalid candidate value", spec.name)
                return False

            sketch_value = self._convert_to_sketch_value(spec.ctype, candidate, spec.angle_unit)
            if sketch_value is None:
                return False

            try:
                self._set_constraint_value(spec.sketch, spec.index, sketch_value, spec.ctype, spec.angle_unit)
            except Exception as exc:
                logger.debug("Failed to set constraint %s: %s", spec.name, exc)
                return False

        # Recompute and validate
        self._doc.recompute()
        if not self._check_recompute(self._doc):
            logger.debug("Recompute failed")
            return False
        if not self._check_surface(self._surface):
            logger.debug("Surface validation failed")
            return False

        return True

    def export_step(self, output_path: Path) -> Path:
        """Export current geometry to STEP file.
        
        Args:
            output_path: Target STEP file path.
            
        Returns:
            Path to exported STEP file.
        """
        if self._surface is None:
            raise RuntimeError("No surface to export")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import Part  # type: ignore
        Part.export([self._surface], str(output_path))
        logger.info("Exported STEP: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private helpers (from proto3-hybrid)
    # ------------------------------------------------------------------

    def _find_sketch(self, doc, sketch_hint: Optional[str] = None):
        """Find sketch object in document."""
        sketches = [obj for obj in doc.Objects if obj.TypeId == "Sketcher::SketchObject"]
        if not sketches:
            raise ValueError("No Sketcher::SketchObject found in document")
        if sketch_hint:
            for obj in sketches:
                if obj.Name == sketch_hint or obj.Label == sketch_hint:
                    return obj
            hint_lower = sketch_hint.lower()
            for obj in sketches:
                if obj.Name.lower() == hint_lower or obj.Label.lower() == hint_lower:
                    return obj
            raise ValueError(f"Sketch not found: {sketch_hint}")
        return sketches[0]

    def _find_surface(self, doc, name: Optional[str], label: Optional[str]):
        """Find surface object in document."""
        if name:
            obj = doc.getObject(name)
            if obj:
                return obj
        if label:
            for obj in doc.Objects:
                if obj.Label == label:
                    return obj
        return None

    def _check_surface(self, obj) -> bool:
        """Validate surface geometry."""
        if obj is None:
            return False
        shape = getattr(obj, "Shape", None)
        if shape is None:
            return False
        if hasattr(shape, "isNull") and shape.isNull():
            return False
        try:
            if hasattr(shape, "isValid") and not shape.isValid():
                return False
        except Exception:
            return False
        try:
            if hasattr(shape, "check"):
                problems = shape.check(True)
                if problems:
                    return False
        except Exception:
            return False
        try:
            area = getattr(shape, "Area", None)
            if area is not None and area <= 0:
                return False
        except Exception:
            return False
        return True

    def _check_recompute(self, doc) -> bool:
        """Check if recompute was successful."""
        try:
            objects = list(getattr(doc, "Objects", []))
        except Exception:
            objects = []
        for obj in objects:
            try:
                state = getattr(obj, "State", None)
                if state and any(flag in state for flag in ("Invalid", "RecomputeError")):
                    return False
            except Exception:
                continue
        return True

    def _set_constraint_value(
        self,
        sketch,
        index: int,
        value: float,
        ctype: str,
        angle_unit: Optional[str],
    ) -> None:
        """Set constraint value in sketch."""
        FreeCAD = _get_freecad()
        if ctype == "Angle":
            if angle_unit == "rad":
                value = math.degrees(value)
            quantity = FreeCAD.Units.Quantity(f"{value} deg")
            sketch.setDatum(index, quantity)
            return
        if ctype in {"Distance", "DistanceX", "DistanceY"}:
            sketch.setDatum(index, float(value))
            return
        sketch.setDatum(index, value)

    def _clamp_candidate(self, ctype: str, value: float, angle_unit: Optional[str]) -> Optional[float]:
        """Validate and clamp candidate value."""
        if not math.isfinite(value):
            return None
        if ctype in {"Distance", "DistanceX", "DistanceY"}:
            if value <= 0:
                return None
            return value
        if ctype == "Angle":
            if angle_unit == "rad":
                if value <= 0 or value >= math.pi:
                    return None
                return value
            if value <= 0 or value >= 180:
                return None
            return value
        return value

    def _convert_to_sketch_value(self, ctype: str, value: float, angle_unit: Optional[str]) -> Optional[float]:
        """Convert value to sketch-compatible format."""
        if not math.isfinite(value):
            return None
        if ctype in {"Distance", "DistanceX", "DistanceY"}:
            if value <= 0:
                return None
            return value
        if ctype == "Angle":
            if angle_unit == "rad":
                if value <= 0 or value >= math.pi:
                    return None
                return value
            if value <= 0 or value >= 180:
                return None
            return value
        return value
