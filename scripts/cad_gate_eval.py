"""
Evaluate CAD gate vs CAE outcome on sampled design points.

Runs FreeCAD + VEXIS for N sampled points, records CAD gate confidence,
CAE success/failure, and computes false positive rate (CAE fail | gate OK).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import importlib.util
import sys


def _bootstrap_proto4_codex(project_root: Path) -> None:
    pkg_dir = project_root / "src" / "proto4-codex"
    sys.path.insert(0, str(pkg_dir))
    spec = importlib.util.spec_from_file_location(
        "proto4_codex",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to bootstrap proto4_codex package")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["proto4_codex"] = mod
    spec.loader.exec_module(mod)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CAD gate vs CAE on sampled points",
    )
    parser.add_argument("--config", default="config/optimizer_config.yaml")
    parser.add_argument("--limits", default="config/proto4_limitations.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cae-timeout", type=int, default=None)
    parser.add_argument("--freecad-bin", default=None)
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=None,
        help="Override gate acceptance threshold for CAE execution",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sample_point(constraints: dict[str, dict[str, float]], rng: random.Random) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, spec in constraints.items():
        lo = float(spec.get("min", 1.0))
        hi = float(spec.get("max", 1.0))
        if lo > hi:
            lo, hi = hi, lo
        params[name] = rng.uniform(lo, hi)
    return params


def _summarize(results: list[dict[str, Any]], threshold: float, success_value: str) -> dict[str, Any]:
    accepted = [r for r in results if r["gate_accept"]]
    fp = [r for r in accepted if r["cae_status"] != success_value]
    fpr = (len(fp) / len(accepted)) if accepted else None
    return {
        "accepted": len(accepted),
        "rejected": len(results) - len(accepted),
        "fp": len(fp),
        "fpr": fpr,
        "threshold": threshold,
    }


def _find_best_threshold(
    results: list[dict[str, Any]],
    current: float,
    success_value: str,
) -> dict[str, Any]:
    confidences = sorted({r["confidence"] for r in results if r["confidence"] is not None})
    if not confidences:
        return {"threshold": current, "summary": _summarize(results, current, success_value)}

    best = None
    for t in confidences:
        for r in results:
            r["gate_accept_tmp"] = r["confidence"] is not None and r["confidence"] >= t
        accepted = [r for r in results if r["gate_accept_tmp"]]
        if not accepted:
            continue
        fp = [r for r in accepted if r["cae_status"] != success_value]
        fpr = len(fp) / len(accepted)
        cand = {"threshold": float(t), "fpr": fpr, "accepted": len(accepted)}
        if best is None:
            best = cand
        else:
            if fpr < best["fpr"]:
                best = cand
            elif fpr == best["fpr"] and t > best["threshold"]:
                best = cand

    for r in results:
        r.pop("gate_accept_tmp", None)

    if best is None:
        return {"threshold": current, "summary": _summarize(results, current, success_value)}

    return {
        "threshold": best["threshold"],
        "summary": _summarize(results, best["threshold"], success_value),
    }


def _isolate_vexis_input(vexis_root: Path, keep_step: Path) -> list[tuple[Path, Path]]:
    input_dir = vexis_root / "input"
    moved: list[tuple[Path, Path]] = []
    backup_dir = vexis_root / "temp" / "cad_gate_eval_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("*.stp", "*.step"):
        for f in input_dir.glob(ext):
            if f.resolve() == keep_step.resolve():
                continue
            dest = backup_dir / f.name
            f.rename(dest)
            moved.append((dest, f))
    return moved


def _restore_vexis_input(moved: list[tuple[Path, Path]]) -> None:
    for src, dest in moved:
        if dest.exists():
            dest.unlink()
        src.rename(dest)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    _bootstrap_proto4_codex(project_root)

    if args.freecad_bin:
        os.environ["FREECAD_BIN"] = args.freecad_bin

    from proto4_codex.cad_gate import CadGate
    from proto4_codex.cae_evaluator import CaeEvaluator, extract_range, extract_features, load_curve
    from proto4_codex.config import load_config
    from proto4_codex.geometry_adapter import GeometryAdapter, GeometryError
    from proto4_codex.types import CaeStatus, DesignPoint

    cfg = load_config(project_root / args.config, project_root / args.limits)
    rng = random.Random(args.seed)

    # Ensure non-interactive VEXIS for batch runs
    os.environ.setdefault("VEXIS_NONINTERACTIVE", "1")
    os.environ.setdefault("PYTHONPATH", str(project_root / "vexis"))

    # Target curve + features
    target_path = project_root / cfg.paths.target_curve
    target_curve_raw = load_curve(target_path)
    target_curve = extract_range(
        target_curve_raw,
        cfg.cae.stroke_range_min,
        cfg.cae.stroke_range_max,
    )
    feat_cfg = cfg.objective.features if cfg.optimization.objective_type == "multi" else {}
    target_features = extract_features(target_curve, feat_cfg) if feat_cfg else {}

    # CAE config override (avoid infinite hangs if timeout=0)
    if args.cae_timeout is not None:
        cfg.cae.timeout_sec = args.cae_timeout
    elif cfg.cae.timeout_sec <= 0:
        cfg.cae.timeout_sec = 300

    cad_gate = CadGate(cfg.cad_gate)
    geometry_adapter = GeometryAdapter(cfg.freecad, project_root)
    cae_evaluator = CaeEvaluator(
        vexis_path=project_root / cfg.paths.vexis_path,
        cae_spec=cfg.cae,
        obj_spec=cfg.objective,
        target_curve=target_curve,
        target_features=target_features,
    )

    effective_threshold = (
        args.accept_threshold
        if args.accept_threshold is not None
        else cfg.cad_gate.threshold
    )

    results: list[dict[str, Any]] = []
    for idx in range(args.samples):
        params = _sample_point(cfg.freecad.constraints, rng)
        point = DesignPoint(trial_id=idx, params=params)

        gate = cad_gate.predict(point)
        if args.accept_threshold is None:
            gate_accept = gate.is_feasible
        else:
            gate_accept = gate.confidence is not None and gate.confidence >= args.accept_threshold

        cae_status = CaeStatus.FAIL
        cae_error = None
        moved_inputs: list[tuple[Path, Path]] = []
        try:
            step_path = geometry_adapter.generate_step(point)
            moved_inputs = _isolate_vexis_input(project_root / cfg.paths.vexis_path, step_path)
            if gate_accept:
                cae_result = cae_evaluator.evaluate(step_path, point)
                cae_status = cae_result.status
            else:
                cae_status = CaeStatus.FAIL
                cae_error = "skipped: gate_reject"
        except GeometryError as exc:
            cae_error = f"geometry_error: {exc}"
        except Exception as exc:
            cae_error = f"cae_error: {exc}"
        finally:
            try:
                _restore_vexis_input(moved_inputs)
            except Exception:
                pass
            try:
                geometry_adapter.cleanup(point)
            except Exception:
                pass

        results.append({
            "trial_id": idx,
            "params": dict(params),
            "confidence": gate.confidence,
            "gate_accept": gate_accept,
            "threshold": effective_threshold,
            "cae_status": cae_status.value,
            "cae_error": cae_error,
        })

    success_value = CaeStatus.SUCCESS.value
    summary_current = _summarize(results, effective_threshold, success_value)
    best = _find_best_threshold(results, cfg.cad_gate.threshold, success_value)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "samples": args.samples,
        "seed": args.seed,
        "current_threshold": cfg.cad_gate.threshold,
        "accept_threshold": effective_threshold,
        "summary_current": summary_current,
        "recommended_threshold": best["threshold"],
        "summary_recommended": best["summary"],
        "results": results,
    }

    out_dir = project_root / "output" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"cad_gate_eval_{_now_tag()}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
