"""Run one active-learning cycle for CAD-gate dataset growth.

Per execution:
1) Load current model and existing dataset.
2) Sample an unlabeled candidate pool in v2 bounds.
3) Select most uncertain candidates around a target threshold.
4) Label only selected points via FreeCAD.
5) Append labeled rows to dataset (+N samples, default 2000).
6) Optionally retrain model.
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
import subprocess
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
    p = argparse.ArgumentParser(description="Append +N active-learning samples to CAD-gate dataset")
    p.add_argument("--config", default="config/optimizer_config.yaml")
    p.add_argument("--limits", default="config/v2_limitations.yaml")
    p.add_argument("--dataset", default="src/ml-prep/data/cad_gate_dataset.csv")
    p.add_argument("--output-dataset", default=None)
    p.add_argument("--summary", default=None)
    p.add_argument("--model-dir", default="src/ml-prep/models/cad_gate_model")
    p.add_argument("--add-samples", type=int, default=2000)
    p.add_argument("--pool-size", type=int, default=20000)
    p.add_argument("--max-pool-rounds", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", choices=("uniform", "lhs"), default="lhs")
    p.add_argument(
        "--uncertainty-center",
        choices=("recommended", "fixed_0_5", "cad_gate_config"),
        default="recommended",
    )
    p.add_argument(
        "--center-threshold",
        type=float,
        default=None,
        help="If set, overrides uncertainty center threshold.",
    )
    p.add_argument("--dedupe-decimals", type=int, default=10)
    p.add_argument("--keep-step", action="store_true")
    p.add_argument(
        "--dataset-workers",
        type=int,
        default=1,
        help="Parallel worker processes for FreeCAD labeling.",
    )
    p.add_argument(
        "--dataset-progress-every",
        type=int,
        default=50,
        help="Progress print interval during FreeCAD labeling.",
    )
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--timeout-sec", type=int, default=1800)
    p.add_argument("--cv-n-jobs", type=int, default=0)
    p.add_argument("--optuna-n-jobs", type=int, default=0)
    p.add_argument("--tree-n-jobs", type=int, default=1)
    p.add_argument("--blas-threads", type=int, default=1)
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


def _load_existing_dataset(
    dataset_path: Path,
    feature_order: list[str],
) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    max_trial_id = -1

    if not dataset_path.exists():
        return rows, max_trial_id

    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, object] = {}
            try:
                tid = int(float(raw.get("trial_id", "-1")))
                max_trial_id = max(max_trial_id, tid)
                row["trial_id"] = tid
            except Exception:
                row["trial_id"] = max_trial_id + 1
                max_trial_id += 1

            try:
                row["label"] = int(float(raw.get("label", "0")))
            except Exception:
                row["label"] = 0
            row["reason"] = str(raw.get("reason", ""))
            try:
                row["elapsed_sec"] = float(raw.get("elapsed_sec", "0"))
            except Exception:
                row["elapsed_sec"] = 0.0

            valid = True
            for name in feature_order:
                try:
                    value = float(raw.get(name, "nan"))
                except Exception:
                    valid = False
                    break
                if not math.isfinite(value):
                    valid = False
                    break
                row[name] = value
            if valid:
                rows.append(row)

    return rows, max_trial_id


def _key_from_params(params: dict[str, float], feature_order: list[str], decimals: int) -> tuple[float, ...]:
    return tuple(round(float(params[name]), decimals) for name in feature_order)


def _load_center_threshold(args: argparse.Namespace, cfg, model_meta: dict) -> float:
    if args.center_threshold is not None:
        return float(args.center_threshold)
    if args.uncertainty_center == "fixed_0_5":
        return 0.5
    if args.uncertainty_center == "cad_gate_config":
        return float(getattr(cfg.cad_gate, "threshold", 0.5))
    return float((model_meta.get("metrics") or {}).get("recommended_threshold", 0.5))


def _select_candidates(
    *,
    specs: list[_SamplingSpec],
    feature_order: list[str],
    rng: random.Random,
    sampler: str,
    add_samples: int,
    pool_size: int,
    max_pool_rounds: int,
    existing_keys: set[tuple[float, ...]],
    dedupe_decimals: int,
    center_threshold: float,
    model,
    scaler,
) -> tuple[list[dict[str, float]], list[float], int]:
    import numpy as np

    selected: list[dict[str, float]] = []
    selected_uncertainty: list[float] = []
    new_keys: set[tuple[float, ...]] = set()
    rounds = 0

    while len(selected) < add_samples and rounds < max_pool_rounds:
        rounds += 1
        if sampler == "lhs":
            pool = _sample_points_lhs(specs, pool_size, rng)
        else:
            pool = _sample_points_uniform(specs, pool_size, rng)

        x_pool = np.asarray(
            [[float(p[name]) for name in feature_order] for p in pool],
            dtype="float64",
        )
        proba = model.predict_proba(scaler.transform(x_pool))[:, 1]
        uncertainty = np.abs(proba - float(center_threshold))
        order = np.argsort(uncertainty)

        before = len(selected)
        for idx in order.tolist():
            params = pool[idx]
            key = _key_from_params(params, feature_order, dedupe_decimals)
            if key in existing_keys or key in new_keys:
                continue
            selected.append(params)
            selected_uncertainty.append(float(uncertainty[idx]))
            new_keys.add(key)
            if len(selected) >= add_samples:
                break

        print(
            f"[select] round={rounds} picked_total={len(selected)} "
            f"picked_in_round={len(selected)-before}"
        )

    return selected, selected_uncertainty, rounds


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
    t0 = time.perf_counter()
    try:
        _G_GEOMETRY_ADAPTER.generate_step(point)
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
    return trial_id, label, reason, elapsed_sec


def _label_selected(
    *,
    project_root: Path,
    args: argparse.Namespace,
    selected: list[dict[str, float]],
    trial_id_start: int,
    feature_order: list[str],
) -> list[dict[str, object]]:
    from v2.config import load_config
    from v2.geometry_adapter import GeometryAdapter
    from v2.types import DesignPoint

    workers = max(1, int(args.dataset_workers))
    progress_every = max(1, int(args.dataset_progress_every))
    out_rows: list[dict[str, object]] = []

    if workers <= 1:
        cfg = load_config(project_root / args.config, project_root / args.limits)
        geometry_adapter = GeometryAdapter(cfg.freecad, project_root, cfg.optimization)
        for i, params in enumerate(selected):
            trial_id = trial_id_start + i
            point = DesignPoint(trial_id=trial_id, params=params)
            label, reason, elapsed = _label_point_serial(
                geometry_adapter=geometry_adapter,
                point=point,
                keep_step=bool(args.keep_step),
            )
            row: dict[str, object] = {
                "trial_id": trial_id,
                "label": label,
                "reason": reason,
                "elapsed_sec": elapsed,
            }
            for name in feature_order:
                row[name] = float(params[name])
            out_rows.append(row)
            if (i + 1) % progress_every == 0:
                feasible = sum(int(r["label"]) for r in out_rows)
                print(f"[label {i+1}/{len(selected)}] feasible={feasible}")
        return out_rows

    cpu_count = max(1, int(os.cpu_count() or 1))
    max_workers = min(workers, cpu_count, len(selected))
    print(f"[parallel] dataset_workers={max_workers} cpu={cpu_count}")
    tasks = [
        (trial_id_start + i, selected[i], bool(args.keep_step))
        for i in range(len(selected))
    ]
    by_trial: dict[int, tuple[int, str, float]] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_parallel_worker_init,
        initargs=(str(project_root), str(args.config), str(args.limits)),
    ) as pool:
        futures = [pool.submit(_parallel_label_one, t) for t in tasks]
        completed = 0
        feasible = 0
        for fut in concurrent.futures.as_completed(futures):
            trial_id, label, reason, elapsed = fut.result()
            by_trial[trial_id] = (label, reason, elapsed)
            feasible += int(label)
            completed += 1
            if completed % progress_every == 0:
                print(f"[label {completed}/{len(selected)}] feasible={feasible}")

    for i, params in enumerate(selected):
        trial_id = trial_id_start + i
        label, reason, elapsed = by_trial.get(
            trial_id, (0, "unexpected:missing_result", 0.0)
        )
        row = {
            "trial_id": trial_id,
            "label": int(label),
            "reason": str(reason),
            "elapsed_sec": float(elapsed),
        }
        for name in feature_order:
            row[name] = float(params[name])
        out_rows.append(row)
    return out_rows


def _write_dataset(path: Path, all_rows: list[dict[str, object]], feature_order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial_id", "label", "reason", "elapsed_sec", *feature_order]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            row = {
                "trial_id": int(r["trial_id"]),
                "label": int(r["label"]),
                "reason": str(r.get("reason", "")),
                "elapsed_sec": float(r.get("elapsed_sec", 0.0)),
            }
            for name in feature_order:
                row[name] = float(r[name])
            w.writerow(row)


def _run_retrain(project_root: Path, args: argparse.Namespace, dataset_rel: str) -> None:
    cmd = [
        sys.executable,
        str(project_root / "src" / "ml-prep" / "scripts" / "train_cad_gate_automl.py"),
        "--dataset",
        dataset_rel,
        "--model-dir",
        args.model_dir,
        "--trials",
        str(args.trials),
        "--timeout-sec",
        str(args.timeout_sec),
        "--seed",
        str(args.seed),
        "--cv-n-jobs",
        str(args.cv_n_jobs),
        "--optuna-n-jobs",
        str(args.optuna_n_jobs),
        "--tree-n-jobs",
        str(args.tree_n_jobs),
        "--blas-threads",
        str(args.blas_threads),
    ]
    print("[retrain] start")
    subprocess.run(cmd, cwd=str(project_root), check=True)


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[3]
    _bootstrap_v2(project_root)

    try:
        import joblib
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency. Install: numpy joblib scikit-learn") from exc

    from v2.config import load_config
    from v2.geometry_adapter import GeometryAdapter
    from v2.search_space import normalize_bounds_to_sampling_grid, sampling_spec_for_bound

    cfg = load_config(project_root / args.config, project_root / args.limits)
    rng = random.Random(args.seed)

    dataset_path = project_root / args.dataset
    output_dataset_path = (
        project_root / args.output_dataset if args.output_dataset else dataset_path
    )
    model_dir = project_root / args.model_dir
    meta_path = model_dir / "metadata.json"
    model_path = model_dir / "model.joblib"
    scaler_path = model_dir / "scaler.joblib"

    if not meta_path.exists() or not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(
            f"Model artifacts missing in {model_dir}. Require metadata.json/model.joblib/scaler.joblib"
        )

    model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_order = [str(x) for x in (model_meta.get("feature_order") or [])]
    if not feature_order:
        raise RuntimeError("feature_order missing in metadata.json")

    center_threshold = _load_center_threshold(args, cfg, model_meta)
    if not (0.0 <= center_threshold <= 1.0):
        raise RuntimeError(f"center threshold must be in [0,1], got {center_threshold}")
    print(f"[active] center_threshold={center_threshold:.6f}")

    existing_rows, max_trial_id = _load_existing_dataset(dataset_path, feature_order)
    existing_keys = {
        _key_from_params({k: float(r[k]) for k in feature_order}, feature_order, args.dedupe_decimals)
        for r in existing_rows
    }
    print(f"[active] existing_rows={len(existing_rows)}")

    geometry_adapter = GeometryAdapter(cfg.freecad, project_root, cfg.optimization)
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

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    selected, uncertainty_vals, rounds = _select_candidates(
        specs=specs,
        feature_order=feature_order,
        rng=rng,
        sampler=args.sampler,
        add_samples=int(args.add_samples),
        pool_size=int(args.pool_size),
        max_pool_rounds=int(args.max_pool_rounds),
        existing_keys=existing_keys,
        dedupe_decimals=int(args.dedupe_decimals),
        center_threshold=float(center_threshold),
        model=model,
        scaler=scaler,
    )
    if len(selected) < int(args.add_samples):
        raise RuntimeError(
            f"Not enough unique candidates selected: requested={args.add_samples}, selected={len(selected)}"
        )

    trial_id_start = max_trial_id + 1
    print(f"[active] labeling selected={len(selected)} starting_trial_id={trial_id_start}")
    new_rows = _label_selected(
        project_root=project_root,
        args=args,
        selected=selected,
        trial_id_start=trial_id_start,
        feature_order=feature_order,
    )

    all_rows = [*existing_rows, *new_rows]
    _write_dataset(output_dataset_path, all_rows, feature_order)
    print(output_dataset_path)

    new_feasible = sum(int(r["label"]) for r in new_rows)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "limits": args.limits,
        "dataset_before": str(dataset_path),
        "dataset_after": str(output_dataset_path),
        "model_dir": str(model_dir),
        "existing_rows": len(existing_rows),
        "added_rows": len(new_rows),
        "total_rows": len(all_rows),
        "new_feasible": new_feasible,
        "new_infeasible": len(new_rows) - new_feasible,
        "center_threshold": float(center_threshold),
        "selection_rounds": int(rounds),
        "pool_size": int(args.pool_size),
        "sampler": args.sampler,
        "seed": int(args.seed),
        "workers": int(args.dataset_workers),
        "uncertainty_mean": float(sum(uncertainty_vals) / len(uncertainty_vals))
        if uncertainty_vals
        else None,
        "uncertainty_min": float(min(uncertainty_vals)) if uncertainty_vals else None,
        "uncertainty_max": float(max(uncertainty_vals)) if uncertainty_vals else None,
        "retrain": bool(args.retrain),
    }
    summary_path = (
        project_root / args.summary
        if args.summary
        else output_dataset_path.with_suffix(".active_learn.summary.json")
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)

    if args.retrain:
        rel_for_train = str(output_dataset_path.relative_to(project_root))
        _run_retrain(project_root, args, rel_for_train)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
