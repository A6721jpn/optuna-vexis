"""
FreeCAD worker subprocess for v2-claude geometry generation.

This script is executed by a FreeCAD-compatible Python interpreter.
It updates sketch constraints from ratio parameters and exports a STEP.

Supports --check-only mode for lightweight feasibility validation
without STEP export (~0.5s vs ~3600s for full CAE pipeline).
"""

from __future__ import annotations

import os
import sys

# Prevent local `types.py` from shadowing stdlib `types` in FreeCAD Python.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
while _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

import argparse
import importlib.util
import json
import logging
from pathlib import Path


def _load_freecad_engine_module():
    module_path = Path(__file__).resolve().parent / "freecad_engine.py"
    spec = importlib.util.spec_from_file_location("v2_freecad_engine_worker", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load freecad_engine module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v2-claude FreeCAD worker")
    p.add_argument("--fcstd-path", required=True)
    p.add_argument("--sketch-name", required=True)
    p.add_argument("--surface-name", required=True)
    p.add_argument("--surface-label", required=True)
    p.add_argument("--constraints-json", required=True)
    p.add_argument("--params-json", required=True)
    p.add_argument("--step-path", required=False, default=None)
    p.add_argument("--check-only", action="store_true", default=False,
                    help="Lightweight feasibility check: recompute + surface validation only, no STEP export")
    p.add_argument("--dump-base-values-json", required=False, default=None)
    p.add_argument("--enable-dimension-discretization", required=False, default="false")
    p.add_argument("--non-angle-step", required=False, type=float, default=0.01)
    p.add_argument("--angle-step", required=False, type=float, default=0.001)
    p.add_argument("--angle-name-token", required=False, default="ANGLE")
    p.add_argument("--discretization-step", required=False, type=float, default=None)
    return p


def _str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _physical_step_for_constraint(
    *,
    name: str,
    enable_dimension_discretization: bool,
    non_angle_step: float,
    angle_step: float,
    angle_name_token: str,
    discretization_step: float | None,
) -> float | None:
    if enable_dimension_discretization:
        token = angle_name_token.upper()
        is_angle = bool(token) and token in name.upper()
        return angle_step if is_angle else non_angle_step
    return discretization_step


def _build_constraint_specs(
    engine,
    constraint_names: list[str],
    ConstraintSpec,
    *,
    enable_dimension_discretization: bool,
    non_angle_step: float,
    angle_step: float,
    angle_name_token: str,
    discretization_step: float | None,
):
    sketch = engine._sketch
    if sketch is None:
        raise RuntimeError("Sketch not loaded")

    sketch_map: dict[str, tuple[int, str]] = {}
    for i in range(sketch.ConstraintCount):
        c = sketch.Constraints[i]
        if c.Name:
            sketch_map[c.Name] = (i, c.Type)

    ctype_map = {
        "Distance": "Distance",
        "DistanceX": "DistanceX",
        "DistanceY": "DistanceY",
        "Angle": "Angle",
    }

    specs = []
    for name in constraint_names:
        if name not in sketch_map:
            logging.warning("Constraint '%s' not found in sketch; skipping", name)
            continue
        real_idx, real_type = sketch_map[name]
        ctype = ctype_map.get(real_type, "Distance")
        current_val = sketch.Constraints[real_idx].Value
        spec = ConstraintSpec(
            index=real_idx,
            name=name,
            ctype=ctype,
            base_value=current_val,
            angle_unit="rad" if ctype == "Angle" else None,
            physical_step=_physical_step_for_constraint(
                name=name,
                enable_dimension_discretization=enable_dimension_discretization,
                non_angle_step=non_angle_step,
                angle_step=angle_step,
                angle_name_token=angle_name_token,
                discretization_step=discretization_step,
            ),
        )
        specs.append(spec)
    return specs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[freecad-worker] %(message)s")
    args = _build_parser().parse_args()
    enable_dimension_discretization = _str_to_bool(args.enable_dimension_discretization)

    # Validate: --step-path is required unless --check-only or --dump-base-values-json
    if not args.check_only and not args.dump_base_values_json and not args.step_path:
        raise ValueError("--step-path is required when not using --check-only or --dump-base-values-json")

    mod = _load_freecad_engine_module()
    FreecadEngine = mod.FreecadEngine
    ConstraintSpec = mod.ConstraintSpec

    fcstd_path = Path(args.fcstd_path)
    constraints_json = Path(args.constraints_json)
    params_json = Path(args.params_json)

    if not fcstd_path.exists():
        raise FileNotFoundError(f"FCStd not found: {fcstd_path}")

    constraints_raw = json.loads(constraints_json.read_text(encoding="utf-8"))
    params = json.loads(params_json.read_text(encoding="utf-8"))
    constraint_names = list(constraints_raw.keys())

    engine = FreecadEngine(
        fcstd_path=fcstd_path,
        sketch_name=args.sketch_name,
        surface_name=args.surface_name,
        surface_label=args.surface_label,
    )
    try:
        engine.open()
        specs = _build_constraint_specs(
            engine,
            constraint_names,
            ConstraintSpec,
            enable_dimension_discretization=enable_dimension_discretization,
            non_angle_step=float(args.non_angle_step),
            angle_step=float(args.angle_step),
            angle_name_token=str(args.angle_name_token),
            discretization_step=args.discretization_step,
        )
        if args.dump_base_values_json:
            payload = {spec.name: float(spec.base_value) for spec in specs}
            out_path = Path(args.dump_base_values_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(out_path)
            return 0

        engine.set_constraints(specs)
        ok = engine.apply_ratios(params)

        if args.check_only:
            # Lightweight feasibility check: output JSON result, no STEP export
            result = {"feasible": ok}
            print(json.dumps(result))
            return 0

        if not ok:
            raise RuntimeError("FreeCAD recompute/validation failed")

        step_path = Path(args.step_path)
        engine.export_step(step_path)
    finally:
        engine.close()

    if not step_path.exists() or step_path.stat().st_size == 0:
        raise RuntimeError(f"STEP export failed: empty or missing {step_path}")

    print(step_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.exception("Worker failed: %s", exc)
        raise
