"""Generate CAD-gate training dataset using v2 geometry checks.

This script samples points in the v2 search space and labels each point as:
  - 1: CAD feasible (STEP generation succeeded)
  - 0: CAD infeasible (geometry generation failed)

Labels are produced by the active v2 FreeCAD flow, including relative-constraint
validation/repair configured in `freecad.relative_constraints`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class _SamplingSpec:
    name: str
    low: float
    high: float
    step: Optional[float]


_G_GEOMETRY_ADAPTER = None
_G_DESIGN_POINT = None
_G_GEOMETRY_ERROR = None


def _bootstrap_v2(project_root: Path) -> None:
    pkg_dir = project_root / "src" / "v2"
    spec = importlib.util.spec_from_file_location(
        "v2",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to bootstrap v2 package")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v2"] = mod
    spec.loader.exec_module(mod)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CAD-gate dataset with v2 geometry labels")
    p.add_argument("--config", default="config/optimizer_config.yaml")
    p.add_argument("--limits", default="config/v2_limitations.yaml")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", choices=("uniform", "lhs"), default="lhs")
    p.add_argument("--output", default="src/ml-prep/data/cad_gate_dataset.csv")
    p.add_argument("--summary", default=None)
    p.add_argument("--keep-step", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for FreeCAD labeling. 1 disables parallelism.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Progress print interval.",
    )
    return p.parse_args()


def _convert_physical_bounds_to_ratio(cfg) -> None:
    if getattr(cfg.freecad, "constraints_domain", "ratio") != "physical":
        return

    for b in cfg.bounds:
        try:
            base = float(b.base_value)
            raw_min = float(b.min)
            raw_max = float(b.max)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(base) or base == 0.0:
            continue

        ratio_min = raw_min / base
        ratio_max = raw_max / base
        if ratio_min <= ratio_max:
            b.min = ratio_min
            b.max = ratio_max
        else:
            b.min = ratio_max
            b.max = ratio_min


def _build_sampling_specs(cfg, sampling_spec_for_bound) -> list[_SamplingSpec]:
    specs: list[_SamplingSpec] = []
    for b in cfg.bounds:
        low, high, step, _ = sampling_spec_for_bound(
            b,
            optimization=cfg.optimization,
            discretization_step=cfg.optimization.discretization_step,
        )
        specs.append(_SamplingSpec(name=b.name, low=float(low), high=float(high), step=step))
    return specs


def _sample_from_spec(spec: _SamplingSpec, rng: random.Random, u: Optional[float] = None) -> float:
    if spec.low == spec.high:
        return spec.low

    if u is None:
        u = rng.random()
    v = spec.low + ((spec.high - spec.low) * u)

    if spec.step is None or spec.step <= 0:
        return float(v)

    n_steps = max(0, int(round((spec.high - spec.low) / spec.step)))
    idx = int(round((v - spec.low) / spec.step))
    if idx < 0:
        idx = 0
    if idx > n_steps:
        idx = n_steps
    snapped = spec.low + (idx * spec.step)
    return float(snapped)


def _sample_points_uniform(specs: list[_SamplingSpec], n: int, rng: random.Random) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for _ in range(n):
        row: dict[str, float] = {}
        for spec in specs:
            row[spec.name] = _sample_from_spec(spec, rng)
        out.append(row)
    return out


def _sample_points_lhs(specs: list[_SamplingSpec], n: int, rng: random.Random) -> list[dict[str, float]]:
    if n <= 0:
        return []

    columns: dict[str, list[float]] = {}
    slots = list(range(n))
    for spec in specs:
        shuffled = list(slots)
        rng.shuffle(shuffled)
        col: list[float] = []
        for slot in shuffled:
            u = (slot + rng.random()) / n
            col.append(_sample_from_spec(spec, rng, u=u))
        columns[spec.name] = col

    out: list[dict[str, float]] = []
    for i in range(n):
        row = {spec.name: columns[spec.name][i] for spec in specs}
        out.append(row)
    return out


def _label_point_serial(geometry_adapter, point, keep_step: bool) -> tuple[int, str, float]:
    from v2.geometry_adapter import GeometryError

    label = 0
    reason = ""
    step_path: Optional[Path] = None
    t0 = time.perf_counter()
    try:
        step_path = geometry_adapter.generate_step(point)
        label = 1
    except GeometryError as exc:
        label = 0
        reason = str(exc)
    except Exception as exc:
        label = 0
        reason = f"unexpected:{exc.__class__.__name__}:{exc}"
    finally:
        elapsed_sec = time.perf_counter() - t0
        if not keep_step:
            try:
                geometry_adapter.cleanup(point)
            except Exception:
                pass
        elif step_path is None:
            pass
    return label, reason, elapsed_sec


def _parallel_worker_init(project_root_str: str, config_rel: str, limits_rel: str) -> None:
    global _G_GEOMETRY_ADAPTER, _G_DESIGN_POINT, _G_GEOMETRY_ERROR

    project_root = Path(project_root_str)
    _bootstrap_v2(project_root)
    from v2.config import load_config
    from v2.geometry_adapter import GeometryAdapter, GeometryError
    from v2.types import DesignPoint

    cfg = load_config(project_root / config_rel, project_root / limits_rel)
    _G_GEOMETRY_ADAPTER = GeometryAdapter(cfg.freecad, project_root, cfg.optimization)
    _G_DESIGN_POINT = DesignPoint
    _G_GEOMETRY_ERROR = GeometryError


def _parallel_label_one(task: tuple[int, dict[str, float], bool]) -> tuple[int, int, str, float]:
    global _G_GEOMETRY_ADAPTER, _G_DESIGN_POINT, _G_GEOMETRY_ERROR

    if _G_GEOMETRY_ADAPTER is None or _G_DESIGN_POINT is None or _G_GEOMETRY_ERROR is None:
        raise RuntimeError("Worker context is not initialized")

    trial_id, params, keep_step = task
    point = _G_DESIGN_POINT(trial_id=trial_id, params=params)
    label = 0
    reason = ""
    step_path: Optional[Path] = None
    t0 = time.perf_counter()
    try:
        step_path = _G_GEOMETRY_ADAPTER.generate_step(point)
        label = 1
    except _G_GEOMETRY_ERROR as exc:
        label = 0
        reason = str(exc)
    except Exception as exc:
        label = 0
        reason = f"unexpected:{exc.__class__.__name__}:{exc}"
    finally:
        elapsed_sec = time.perf_counter() - t0
        if not keep_step:
            try:
                _G_GEOMETRY_ADAPTER.cleanup(point)
            except Exception:
                pass
        elif step_path is None:
            pass
    return trial_id, label, reason, elapsed_sec


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[3]
    _bootstrap_v2(project_root)

    from v2.config import load_config
    from v2.geometry_adapter import GeometryAdapter, GeometryError
    from v2.search_space import normalize_bounds_to_sampling_grid, sampling_spec_for_bound
    from v2.types import DesignPoint

    cfg = load_config(project_root / args.config, project_root / args.limits)
    rng = random.Random(args.seed)

    geometry_adapter = GeometryAdapter(cfg.freecad, project_root, cfg.optimization)

    # Align base values with actual FCStd dimensions before ratio conversion.
    try:
        base_values = geometry_adapter.probe_base_values([b.name for b in cfg.bounds])
        for b in cfg.bounds:
            if b.name in base_values:
                b.base_value = float(base_values[b.name])
    except Exception:
        pass

    _convert_physical_bounds_to_ratio(cfg)
    normalize_bounds_to_sampling_grid(
        cfg.bounds,
        optimization=cfg.optimization,
        discretization_step=cfg.optimization.discretization_step,
    )

    specs = _build_sampling_specs(cfg, sampling_spec_for_bound)
    if args.sampler == "lhs":
        sampled = _sample_points_lhs(specs, args.samples, rng)
    else:
        sampled = _sample_points_uniform(specs, args.samples, rng)

    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    success_count = 0
    workers = max(1, int(args.workers))
    progress_every = max(1, int(args.progress_every))

    if workers <= 1:
        for i, params in enumerate(sampled):
            point = DesignPoint(trial_id=i, params=params)
            label, reason, elapsed_sec = _label_point_serial(
                geometry_adapter=geometry_adapter,
                point=point,
                keep_step=bool(args.keep_step),
            )
            if label == 1:
                success_count += 1

            row: dict[str, object] = {
                "trial_id": i,
                "label": label,
                "reason": reason,
                "elapsed_sec": elapsed_sec,
            }
            for spec in specs:
                row[spec.name] = float(params.get(spec.name, 1.0))
            rows.append(row)

            if (i + 1) % progress_every == 0:
                print(f"[{i+1}/{args.samples}] feasible={success_count}")
    else:
        cpu_count = max(1, int(os.cpu_count() or 1))
        max_workers = min(workers, cpu_count, args.samples if args.samples > 0 else workers)
        print(f"[parallel] dataset_workers={max_workers} cpu={cpu_count}")
        results_by_trial: dict[int, tuple[int, str, float]] = {}
        tasks = [(i, sampled[i], bool(args.keep_step)) for i in range(len(sampled))]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_parallel_worker_init,
            initargs=(str(project_root), str(args.config), str(args.limits)),
        ) as pool:
            futures = [pool.submit(_parallel_label_one, task) for task in tasks]
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                trial_id, label, reason, elapsed_sec = fut.result()
                results_by_trial[trial_id] = (label, reason, elapsed_sec)
                if label == 1:
                    success_count += 1
                completed += 1
                if completed % progress_every == 0:
                    print(f"[{completed}/{args.samples}] feasible={success_count}")

        for i, params in enumerate(sampled):
            label, reason, elapsed_sec = results_by_trial.get(
                i, (0, "unexpected:missing_result", 0.0)
            )
            row = {
                "trial_id": i,
                "label": label,
                "reason": reason,
                "elapsed_sec": elapsed_sec,
            }
            for spec in specs:
                row[spec.name] = float(params.get(spec.name, 1.0))
            rows.append(row)

    feature_names = [spec.name for spec in specs]
    fieldnames = ["trial_id", "label", "reason", "elapsed_sec", *feature_names]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    infeasible_count = len(rows) - success_count
    summary = {
        "timestamp": datetime.now().isoformat(),
        "samples": len(rows),
        "feasible": success_count,
        "infeasible": infeasible_count,
        "feasible_rate": (success_count / len(rows)) if rows else 0.0,
        "config": args.config,
        "limits": args.limits,
        "sampler": args.sampler,
        "seed": args.seed,
        "workers": workers,
        "output": str(out_path),
        "relative_constraints_enabled": bool(getattr(cfg.freecad, "relative_constraints", [])),
    }

    summary_path = project_root / args.summary if args.summary else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
