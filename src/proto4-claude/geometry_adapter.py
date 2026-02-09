"""
Proto4 Geometry Adapter

Converts a feasible DesignPoint into a STEP file that VEXIS can consume.
Wraps FreeCAD / cad-automaton headless execution (same role as proto3 CadEditor).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

    def generate_step(self, point: DesignPoint) -> Path:
        """Update FreeCAD constraints and export STEP.

        Args:
            point: Design point whose params map to sketch constraint names.

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

        # TODO: integrate cad-automaton / FreeCAD headless
        #   1. Open FCStd
        #   2. Update sketch constraints from point.params
        #   3. Export STEP to step_path
        #   4. Validate exported file
        raise GeometryError(
            "GeometryAdapter is not implemented yet — "
            "wire cad-automaton here"
        )

    def cleanup(self, point: DesignPoint) -> None:
        """Remove temporary STEP file for the given trial."""
        step_name = self._spec.step_filename_template.format(trial_id=point.trial_id)
        step_path = self._step_output_dir / step_name
        if step_path.exists():
            step_path.unlink()
            logger.debug("Cleaned up STEP: %s", step_path)
