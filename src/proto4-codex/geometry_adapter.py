"""
Proto4 Geometry Adapter

Converts a feasible DesignPoint into a STEP file that VEXIS can consume.
FreeCAD work is executed in a dedicated subprocess so ABI mismatches
between the main Python and FreeCAD's Python do not crash optimization.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .config import FreecadSpec
from .types import DesignPoint

logger = logging.getLogger(__name__)


class GeometryError(RuntimeError):
    """Raised when geometry generation fails."""


class GeometryAdapter:
    """Generate STEP geometry from a DesignPoint via FreeCAD subprocess."""

    def __init__(self, spec: FreecadSpec, project_root: Path) -> None:
        self._spec = spec
        self._project_root = project_root
        self._fcstd_path = project_root / spec.fcstd_path
        self._step_output_dir = project_root / spec.step_output_dir
        self._engine = None  # compatibility no-op

    def _worker_script_path(self) -> Path:
        return self._project_root / "src" / "proto4-codex" / "freecad_worker.py"

    @staticmethod
    def _python_from_bin_dir(bin_dir: Path) -> Optional[Path]:
        if os.name == "nt":
            py = bin_dir / "python.exe"
            return py if py.exists() else None
        py = bin_dir / "python"
        return py if py.exists() else None

    def _resolve_freecad_python(self) -> Path:
        # Highest priority: explicit full interpreter path
        explicit = os.environ.get("FREECAD_PYTHON")
        if explicit:
            p = Path(explicit)
            if p.exists():
                return p

        # Next: directory that contains FreeCAD's python.exe
        freecad_bin = os.environ.get("FREECAD_BIN")
        if freecad_bin:
            p = self._python_from_bin_dir(Path(freecad_bin))
            if p is not None:
                return p

        # Default Windows install location
        default_bin = Path(r"C:\Program Files\FreeCAD 1.0\bin")
        p = self._python_from_bin_dir(default_bin)
        if p is not None:
            return p

        raise GeometryError(
            "FreeCAD Python interpreter not found. "
            "Set FREECAD_PYTHON or FREECAD_BIN."
        )

    def _run_worker(self, point: DesignPoint, step_path: Path) -> None:
        worker = self._worker_script_path()
        if not worker.exists():
            raise GeometryError(f"FreeCAD worker script not found: {worker}")

        freecad_python = self._resolve_freecad_python()
        timeout_sec = self._spec.timeout_sec if self._spec.timeout_sec > 0 else None

        with tempfile.TemporaryDirectory(prefix=f"proto4_trial_{point.trial_id}_") as tmpdir:
            tmp = Path(tmpdir)
            constraints_json = tmp / "constraints.json"
            params_json = tmp / "params.json"

            constraints_json.write_text(
                json.dumps(self._spec.constraints, ensure_ascii=False),
                encoding="utf-8",
            )
            params_json.write_text(
                json.dumps(point.params, ensure_ascii=False),
                encoding="utf-8",
            )

            cmd = [
                str(freecad_python),
                str(worker),
                "--fcstd-path",
                str(self._fcstd_path),
                "--sketch-name",
                str(self._spec.sketch_name),
                "--surface-name",
                str(getattr(self._spec, "surface_name", "Face")),
                "--surface-label",
                str(getattr(self._spec, "surface_label", "SURFACE")),
                "--constraints-json",
                str(constraints_json),
                "--params-json",
                str(params_json),
                "--step-path",
                str(step_path),
            ]

            logger.info("FreeCAD worker run: %s", " ".join(cmd))
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self._project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise GeometryError(
                    f"FreeCAD worker timeout after {timeout_sec}s for trial {point.trial_id}"
                ) from exc

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                detail = stderr or stdout or "no output"
                raise GeometryError(
                    f"FreeCAD worker failed for trial {point.trial_id} "
                    f"(exit={proc.returncode}): {detail}"
                )

    def generate_step(self, point: DesignPoint) -> Path:
        """Update FreeCAD constraints and export STEP."""
        step_name = self._spec.step_filename_template.format(trial_id=point.trial_id)
        step_path = self._step_output_dir / step_name

        if not self._fcstd_path.exists():
            raise GeometryError(f"FCStd not found: {self._fcstd_path}")

        step_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_worker(point, step_path)
        except GeometryError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise GeometryError(f"FreeCAD error for trial {point.trial_id}: {exc}") from exc

        if not step_path.exists() or step_path.stat().st_size == 0:
            raise GeometryError(f"STEP export failed: empty or missing {step_path}")

        logger.info("Generated STEP for trial %d: %s", point.trial_id, step_path)
        return step_path

    def cleanup(self, point: DesignPoint) -> None:
        """Remove temporary STEP file for the given trial."""
        step_name = self._spec.step_filename_template.format(trial_id=point.trial_id)
        step_path = self._step_output_dir / step_name
        if step_path.exists():
            step_path.unlink()
            logger.debug("Cleaned up STEP: %s", step_path)

    def close(self) -> None:
        """Compatibility no-op; worker subprocess owns FreeCAD lifecycle."""
        self._engine = None

