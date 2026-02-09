"""
Proto4 Geometry Adapter

Converts a feasible DesignPoint into a STEP file that VEXIS can consume.
Wraps FreeCAD / cad-automaton headless execution (same role as proto3 CadEditor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .config import FreecadSpec
from .types import DesignPoint

logger = logging.getLogger(__name__)


class GeometryError(RuntimeError):
    """Raised when geometry generation fails."""


class GeometryAdapter:
    """Generate STEP geometry from a DesignPoint via FreeCAD."""

    def __init__(self, spec: FreecadSpec, project_root: Path) -> None:
        self._spec = spec
        self._project_root = project_root
        self._fcstd_path = project_root / spec.fcstd_path
        self._step_output_dir = project_root / spec.step_output_dir
        self._engine: Optional[Any] = None

    def _get_engine(self):
        """Lazy-load FreecadEngine."""
        if self._engine is not None:
            return self._engine

        from .freecad_engine import FreecadEngine, ConstraintSpec

        # Build constraint specs from config
        specs = []
        for idx, (name, constraint) in enumerate(self._spec.constraints.items()):
            spec = ConstraintSpec(
                index=idx,
                name=name,
                ctype="Distance",  # Default to Distance; real impl needs mapping
                base_value=constraint.get("base_value", 1.0),
            )
            specs.append(spec)

        engine = FreecadEngine(
            fcstd_path=self._fcstd_path,
            sketch_name=self._spec.sketch_name,
            surface_name=getattr(self._spec, "surface_name", "Face"),
            surface_label=getattr(self._spec, "surface_label", "SURFACE"),
        )
        engine.open()
        engine.set_constraints(specs)
        self._engine = engine
        return engine

    def generate_step(self, point: DesignPoint) -> Path:
        """Update FreeCAD constraints and export STEP.

        Args:
            point: Design point whose params map to sketch constraint names.
                   Params are ratio values (1.0 = baseline).

        Returns:
            Path to the exported STEP file.

        Raises:
            GeometryError: on any FreeCAD / export failure.
        """
        step_name = self._spec.step_filename_template.format(trial_id=point.trial_id)
        step_path = self._step_output_dir / step_name

        if not self._fcstd_path.exists():
            raise GeometryError(f"FCStd not found: {self._fcstd_path}")

        step_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            engine = self._get_engine()

            # Apply ratio values from DesignPoint
            if not engine.apply_ratios(point.params):
                raise GeometryError(
                    f"FreeCAD recompute failed for trial {point.trial_id}"
                )

            # Export STEP
            engine.export_step(step_path)

            # Validate exported file
            if not step_path.exists() or step_path.stat().st_size == 0:
                raise GeometryError(f"STEP export failed: empty or missing {step_path}")

            logger.info("Generated STEP for trial %d: %s", point.trial_id, step_path)
            return step_path

        except GeometryError:
            raise
        except Exception as exc:
            raise GeometryError(f"FreeCAD error for trial {point.trial_id}: {exc}") from exc

    def cleanup(self, point: DesignPoint) -> None:
        """Remove temporary STEP file for the given trial."""
        step_name = self._spec.step_filename_template.format(trial_id=point.trial_id)
        step_path = self._step_output_dir / step_name
        if step_path.exists():
            step_path.unlink()
            logger.debug("Cleaned up STEP: %s", step_path)

    def close(self) -> None:
        """Close FreeCAD engine if open."""
        if self._engine is not None:
            self._engine.close()
            self._engine = None

