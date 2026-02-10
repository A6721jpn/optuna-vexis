"""
CAD gate false-positive study using FreeCAD outcome only.

Compares:
  - model prediction (gate OK/NG)
  - FreeCAD geometry generation success/failure

No CAE is executed in this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


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
        description="Study CAD gate false positives against FreeCAD only",
    )
    parser.add_argument("--config", default="config/optimizer_config.yaml")
    parser.add_argument("--limits", default="config/proto4_limitations.yaml")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freecad-bin", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override CAD gate threshold for this study",
    )
    parser.add_argument(
        "--sampler",
        choices=("lhs", "uniform"),
        default="lhs",
        help="Sampling strategy for parameter space",
    )
    parser.add_argument(
        "--expand-factor",
        type=float,
        default=1.0,
        help="Multiply each parameter half-range around center",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _wilson_interval(k: int, n: int, z: float = 1.96) -> dict[str, float | None]:
    if n <= 0:
        return {"low": None, "high": None}
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _make_uniform_points(
    bounds: dict[str, tuple[float, float]],
    n: int,
    rng: random.Random,
) -> list[dict[str, float]]:
    keys = list(bounds.keys())
    points: list[dict[str, float]] = []
    for _ in range(n):
        row: dict[str, float] = {}
        for key in keys:
            lo, hi = bounds[key]
            row[key] = rng.uniform(lo, hi)
        points.append(row)
    return points


def _make_lhs_points(
    bounds: dict[str, tuple[float, float]],
    n: int,
    rng: random.Random,
) -> list[dict[str, float]]:
    keys = list(bounds.keys())
    unit_cols: dict[str, list[float]] = {}
    for key in keys:
        slots = list(range(n))
        rng.shuffle(slots)
        values = []
        for slot in slots:
            u = (slot + rng.random()) / n
            values.append(u)
        unit_cols[key] = values

    points: list[dict[str, float]] = []
    for index in range(n):
        row: dict[str, float] = {}
        for key in keys:
            lo, hi = bounds[key]
            u = unit_cols[key][index]
            row[key] = lo + (hi - lo) * u
        points.append(row)
    return points


def _sweep_thresholds(
    rows: list[dict[str, Any]],
    minimum_accept: int = 10,
) -> dict[str, Any]:
    confidences = sorted({r["confidence"] for r in rows if r["confidence"] is not None})
    best: dict[str, Any] | None = None
    for threshold in confidences:
        accepted = [r for r in rows if r["confidence"] is not None and r["confidence"] >= threshold]
        if len(accepted) < minimum_accept:
            continue
        fp = sum(1 for r in accepted if not r["cad_success"])
        fpr = fp / len(accepted)
        candidate = {
            "threshold": float(threshold),
            "accepted": len(accepted),
            "fp": fp,
            "fpr": fpr,
        }
        if best is None:
            best = candidate
            continue
        if fpr < best["fpr"]:
            best = candidate
            continue
        if fpr == best["fpr"] and threshold > best["threshold"]:
            best = candidate
    return best or {}


def _probe_thresholds(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted_rows = [
            r for r in rows if r["confidence"] is not None and r["confidence"] >= threshold
        ]
        accepted = len(accepted_rows)
        fp = sum(1 for r in accepted_rows if not r["cad_success"])
        fpr = (fp / accepted) if accepted else None
        out.append(
            {
                "threshold": threshold,
                "accepted": accepted,
                "fp": fp,
                "fpr": fpr,
                "fpr_wilson95": _wilson_interval(fp, accepted),
            }
        )
    return out


def _feature_ranges(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(r["params"][key]) for r in rows]
        stats[key] = {"min": min(values), "max": max(values), "mean": mean(values)}
    return stats


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    _bootstrap_proto4_codex(project_root)

    if args.freecad_bin:
        os.environ["FREECAD_BIN"] = args.freecad_bin

    from proto4_codex.cad_gate import CadGate
    from proto4_codex.config import load_config
    from proto4_codex.geometry_adapter import GeometryAdapter, GeometryError
    from proto4_codex.types import DesignPoint

    cfg = load_config(project_root / args.config, project_root / args.limits)
    threshold = cfg.cad_gate.threshold if args.threshold is None else args.threshold
    rng = random.Random(args.seed)

    bounds: dict[str, tuple[float, float]] = {}
    for key, spec in cfg.freecad.constraints.items():
        lo = float(spec.get("min", 1.0))
        hi = float(spec.get("max", 1.0))
        if lo > hi:
            lo, hi = hi, lo
        if args.expand_factor != 1.0:
            center = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            half = half * args.expand_factor
            lo = max(1e-6, center - half)
            hi = center + half
        bounds[key] = (lo, hi)

    if args.sampler == "lhs":
        sample_points = _make_lhs_points(bounds, args.samples, rng)
    else:
        sample_points = _make_uniform_points(bounds, args.samples, rng)

    cad_gate = CadGate(cfg.cad_gate)
    geometry_adapter = GeometryAdapter(cfg.freecad, project_root)

    rows: list[dict[str, Any]] = []
    for index, params in enumerate(sample_points):
        point = DesignPoint(trial_id=index, params=params)
        gate = cad_gate.predict(point)
        confidence = float(gate.confidence) if gate.confidence is not None else None
        model_ok = confidence is not None and confidence >= threshold

        cad_success = False
        cad_error: str | None = None
        try:
            step_path = geometry_adapter.generate_step(point)
            cad_success = step_path.exists() and step_path.stat().st_size > 0
        except GeometryError as exc:
            cad_error = str(exc)
            cad_success = False
        except Exception as exc:  # pragma: no cover - defensive fallback
            cad_error = str(exc)
            cad_success = False
        finally:
            try:
                geometry_adapter.cleanup(point)
            except Exception:
                pass

        rows.append(
            {
                "trial_id": index,
                "params": dict(params),
                "confidence": confidence,
                "model_ok": model_ok,
                "cad_success": cad_success,
                "cad_error": cad_error,
            }
        )

    tp = sum(1 for r in rows if r["model_ok"] and r["cad_success"])
    fp = sum(1 for r in rows if r["model_ok"] and not r["cad_success"])
    fn = sum(1 for r in rows if (not r["model_ok"]) and r["cad_success"])
    tn = sum(1 for r in rows if (not r["model_ok"]) and (not r["cad_success"]))

    accepted = tp + fp
    fpr = (fp / accepted) if accepted else None
    fpr_ci = _wilson_interval(fp, accepted)

    threshold_sweep = _sweep_thresholds(rows, minimum_accept=max(10, args.samples // 20))
    feature_stats = _feature_ranges(rows, list(bounds.keys()))
    threshold_probe = _probe_thresholds(
        rows,
        thresholds=[0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
    )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "samples": args.samples,
        "seed": args.seed,
        "sampler": args.sampler,
        "expand_factor": args.expand_factor,
        "threshold_eval": threshold,
        "threshold_config": cfg.cad_gate.threshold,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accepted": accepted,
        "false_positive_rate": fpr,
        "false_positive_rate_wilson95": fpr_ci,
        "cad_success_rate": (tp + fn) / args.samples,
        "threshold_sweep_best": threshold_sweep,
        "threshold_probe": threshold_probe,
        "feature_coverage": feature_stats,
        "rows": rows,
    }

    out_dir = project_root / "output" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"cad_gate_freecad_fp_{_now_tag()}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
