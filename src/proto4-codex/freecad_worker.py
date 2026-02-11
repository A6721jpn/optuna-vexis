"""
FreeCAD worker subprocess for proto4-codex geometry generation.

This script is executed by a FreeCAD-compatible Python interpreter.
It updates sketch constraints from ratio parameters and exports a STEP.
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
    spec = importlib.util.spec_from_file_location("proto4_freecad_engine_worker", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load freecad_engine module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Proto4 FreeCAD worker")
    p.add_argument("--fcstd-path", required=True)
    p.add_argument("--sketch-name", required=True)
    p.add_argument("--surface-name", required=True)
    p.add_argument("--surface-label", required=True)
    p.add_argument("--constraints-json", required=True)
    p.add_argument("--params-json", required=True)
    p.add_argument("--step-path", required=True)
    return p


def _build_constraint_specs(engine, constraint_names: list[str], ConstraintSpec):
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
        )
        specs.append(spec)
    return specs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[freecad-worker] %(message)s")
    args = _build_parser().parse_args()

    mod = _load_freecad_engine_module()
    FreecadEngine = mod.FreecadEngine
    ConstraintSpec = mod.ConstraintSpec

    fcstd_path = Path(args.fcstd_path)
    step_path = Path(args.step_path)
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
        specs = _build_constraint_specs(engine, constraint_names, ConstraintSpec)
        engine.set_constraints(specs)
        if not engine.apply_ratios(params):
            raise RuntimeError("FreeCAD recompute/validation failed")
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
