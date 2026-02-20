"""
v2 Geometry Adapter

Converts a feasible DesignPoint into a STEP file that VEXIS can consume.
FreeCAD work is executed in a dedicated subprocess so ABI mismatches
between the main Python and FreeCAD's Python do not crash optimization.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .config import FreecadSpec, OptimizationSpec
from .types import DesignPoint

logger = logging.getLogger(__name__)


class GeometryError(RuntimeError):
    """Raised when geometry generation fails."""


class GeometryAdapter:
    """Generate STEP geometry from a DesignPoint via FreeCAD subprocess."""

    def __init__(
        self,
        spec: FreecadSpec,
        project_root: Path,
        optimization: Optional[OptimizationSpec] = None,
    ) -> None:
        self._spec = spec
        self._project_root = project_root
        self._optimization = optimization
        self._fcstd_path = project_root / spec.fcstd_path
        self._step_output_dir = project_root / spec.step_output_dir
        self._engine = None  # compatibility no-op

    def _worker_script_path(self) -> Path:
        return self._project_root / "src" / "v2" / "freecad_worker.py"

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

        with tempfile.TemporaryDirectory(prefix=f"v2_trial_{point.trial_id}_") as tmpdir:
            tmp = Path(tmpdir)
            constraints_json = tmp / "constraints.json"
            params_json = tmp / "params.json"
            relative_constraints_json = tmp / "relative_constraints.json"
            relative_repair_json = tmp / "relative_repair.json"

            constraints_json.write_text(
                json.dumps(self._spec.constraints, ensure_ascii=False),
                encoding="utf-8",
            )
            params_json.write_text(
                json.dumps(point.params, ensure_ascii=False),
                encoding="utf-8",
            )
            relative_rules = []
            for rule in getattr(self._spec, "relative_constraints", []):
                relative_rules.append(
                    {
                        "id": str(rule.id),
                        "lhs": str(rule.lhs),
                        "op": str(rule.op),
                        "rhs": str(rule.rhs),
                        "tolerance": float(rule.tolerance),
                        "weight": float(rule.weight),
                        "on_violation": str(rule.on_violation),
                        "repair_drivers": [str(x) for x in rule.repair_drivers],
                    }
                )
            relative_constraints_json.write_text(
                json.dumps(relative_rules, ensure_ascii=False),
                encoding="utf-8",
            )
            repair_spec = getattr(self._spec, "relative_constraint_repair", None)
            relative_repair_json.write_text(
                json.dumps(
                    {
                        "enabled": bool(getattr(repair_spec, "enabled", True)),
                        "max_iters": int(getattr(repair_spec, "max_iters", 20)),
                        "max_evals": int(getattr(repair_spec, "max_evals", 80)),
                        "step_decay": float(getattr(repair_spec, "step_decay", 0.5)),
                        "initial_step_scale": float(
                            getattr(repair_spec, "initial_step_scale", 6.0)
                        ),
                        "min_step_ratio": float(
                            getattr(repair_spec, "min_step_ratio", 1.0e-4)
                        ),
                        "regularization_lambda": float(
                            getattr(repair_spec, "regularization_lambda", 1.0e-2)
                        ),
                    },
                    ensure_ascii=False,
                ),
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
                "--constraints-domain",
                str(getattr(self._spec, "constraints_domain", "ratio")),
                "--relative-constraints-json",
                str(relative_constraints_json),
                "--relative-repair-json",
                str(relative_repair_json),
                "--step-path",
                str(step_path),
                "--enable-dimension-discretization",
                str(
                    bool(self._optimization.enable_dimension_discretization)
                    if self._optimization is not None
                    else False
                ).lower(),
                "--non-angle-step",
                str(self._optimization.non_angle_step if self._optimization is not None else 0.01),
                "--angle-step",
                str(self._optimization.angle_step if self._optimization is not None else 0.001),
                "--angle-name-token",
                str(self._optimization.angle_name_token if self._optimization is not None else "ANGLE"),
            ]
            if self._optimization is not None and self._optimization.discretization_step is not None:
                cmd.extend([
                    "--discretization-step",
                    str(self._optimization.discretization_step),
                ])

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

    def probe_base_values(self, constraint_names: list[str]) -> dict[str, float]:
        """Read current sketch constraint values (physical domain) from FCStd."""
        if not self._fcstd_path.exists():
            raise GeometryError(f"FCStd not found: {self._fcstd_path}")

        worker = self._worker_script_path()
        if not worker.exists():
            raise GeometryError(f"FreeCAD worker script not found: {worker}")

        freecad_python = self._resolve_freecad_python()
        timeout_sec = self._spec.timeout_sec if self._spec.timeout_sec > 0 else None

        with tempfile.TemporaryDirectory(prefix="v2_probe_base_") as tmpdir:
            tmp = Path(tmpdir)
            constraints_json = tmp / "constraints.json"
            params_json = tmp / "params.json"
            step_path = tmp / "probe_unused.step"
            out_json = tmp / "base_values.json"

            constraints_payload = {
                name: self._spec.constraints.get(name, {})
                for name in constraint_names
            }
            constraints_json.write_text(
                json.dumps(constraints_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            params_json.write_text("{}", encoding="utf-8")

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
                "--dump-base-values-json",
                str(out_json),
            ]

            logger.info("FreeCAD base-value probe run: %s", " ".join(cmd))
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
                    f"FreeCAD base-value probe timeout after {timeout_sec}s"
                ) from exc

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                detail = stderr or stdout or "no output"
                raise GeometryError(
                    f"FreeCAD base-value probe failed (exit={proc.returncode}): {detail}"
                )

            if not out_json.exists():
                raise GeometryError(f"Base-value dump not found: {out_json}")

            raw = json.loads(out_json.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise GeometryError("Invalid base-value dump format")

            out: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(v):
                    continue
                out[str(key)] = v
            return out

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
