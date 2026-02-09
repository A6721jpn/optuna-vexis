"""
Proto3 CAD Editor (skeleton)

Wraps cad-automaton / FreeCAD headless execution to update sketch constraints
and export STEP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CadError(RuntimeError):
    pass


class CadEditor:
    def __init__(self) -> None:
        # Placeholder for future wiring to cad-automaton
        pass

    def update_constraints_and_export_step(
        self,
        fcstd_path: str | Path,
        sketch_name: str,
        constraints: dict[str, Any],
        step_path: str | Path,
        timeout_sec: int = 300,
    ) -> Path:
        """
        Update constraints in a FreeCAD sketch and export STEP.

        Args:
            fcstd_path: Path to .FCStd file
            sketch_name: Sketch object name
            constraints: {constraint_name: value}
            step_path: Output STEP path
            timeout_sec: Timeout seconds for FreeCAD execution

        Returns:
            Path to the exported STEP file
        """
        fcstd_path = Path(fcstd_path)
        step_path = Path(step_path)

        if not fcstd_path.exists():
            raise CadError(f"FCStd not found: {fcstd_path}")

        step_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: integrate cad-automaton script/CLI
        # - call FreeCAD headless
        # - update constraints
        # - export STEP
        # - validate STEP exists
        raise CadError("CadEditor is not implemented yet")

