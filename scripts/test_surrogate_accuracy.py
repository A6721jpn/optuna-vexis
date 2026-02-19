"""Validate surrogate accuracy against FreeCAD ground truth.

Generates test points, gets surrogate predictions, then verifies
with FreeCAD --check-only to measure precision/recall.
"""

import importlib.util
import sys
import os
import logging
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(ROOT, "src", "v2-claude")
MOD_NAME = "v2_claude"

spec = importlib.util.spec_from_file_location(
    MOD_NAME,
    os.path.join(PKG_DIR, "__init__.py"),
    submodule_search_locations=[PKG_DIR],
)
mod = importlib.util.module_from_spec(spec)
sys.modules[MOD_NAME] = mod
spec.loader.exec_module(mod)

for py_file in sorted(os.listdir(PKG_DIR)):
    if py_file.endswith(".py") and py_file != "__init__.py":
        sub_name = f"{MOD_NAME}.{py_file[:-3]}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name, os.path.join(PKG_DIR, py_file),
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[sub_name] = sub_mod
            sub_spec.loader.exec_module(sub_mod)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import numpy as np
from v2_claude.config import load_config
from v2_claude.geometry_adapter import GeometryAdapter
from v2_claude.feasibility_oracle import FeasibilityOracle


def main():
    cfg = load_config(
        Path(ROOT) / "config" / "optimizer_config.yaml",
        Path(ROOT) / "config" / "v2_claude_limitations.yaml",
    )
    geometry_adapter = GeometryAdapter(cfg.freecad, project_root=Path(ROOT))

    oracle = FeasibilityOracle(
        geometry_adapter=geometry_adapter,
        bounds=cfg.bounds,
        feasibility_spec=cfg.feasibility,
    )

    print(f"DB records: {len(oracle.db)} (feasible={oracle.db.feasible_count}, infeasible={oracle.db.infeasible_count})")
    print(f"Surrogate trained: {oracle.surrogate.is_trained}")

    if not oracle.surrogate.is_trained:
        print("ERROR: Surrogate not trained, cannot validate")
        return

    # Generate 20 test points with progressive perturbation
    rng = np.random.default_rng(seed=99)
    param_names = [b.name for b in cfg.bounds]
    n_dims = len(param_names)

    test_points = []
    # 5 points with 1-2 dims perturbed
    for _ in range(5):
        k = rng.integers(1, 3)
        dims = rng.choice(n_dims, size=k, replace=False)
        params = {name: 1.0 for name in param_names}
        for d in dims:
            b = cfg.bounds[d]
            lo = b.min / b.base_value
            hi = b.max / b.base_value
            params[param_names[d]] = float(rng.uniform(lo, hi))
        test_points.append(params)

    # 5 points with 5-8 dims perturbed
    for _ in range(5):
        k = rng.integers(5, 9)
        dims = rng.choice(n_dims, size=k, replace=False)
        params = {name: 1.0 for name in param_names}
        for d in dims:
            b = cfg.bounds[d]
            lo = b.min / b.base_value
            hi = b.max / b.base_value
            params[param_names[d]] = float(rng.uniform(lo, hi))
        test_points.append(params)

    # 5 points with all dims perturbed (within bounds, no margin)
    for _ in range(5):
        params = {}
        for i, name in enumerate(param_names):
            b = cfg.bounds[i]
            lo = b.min / b.base_value
            hi = b.max / b.base_value
            params[name] = float(rng.uniform(lo, hi))
        test_points.append(params)

    # 5 points at baseline neighborhood (±2%)
    for _ in range(5):
        params = {}
        for name in param_names:
            params[name] = float(rng.uniform(0.98, 1.02))
        test_points.append(params)

    print(f"\nTesting {len(test_points)} points...")
    print(f"{'#':>3} {'dims_pert':>9} {'surr_score':>10} {'tier_pred':>14} {'freecad':>8} {'correct':>8}")
    print("-" * 65)

    tp = fp = tn = fn = 0
    for i, params in enumerate(test_points):
        n_pert = sum(1 for v in params.values() if abs(v - 1.0) > 0.001)

        # Surrogate prediction
        score = oracle.predict_surrogate(params)
        if score >= cfg.feasibility.tier_high_confidence:
            tier = "high_conf"
        elif score >= cfg.feasibility.tier_uncertain:
            tier = "uncertain"
        else:
            tier = "rejected"

        surr_pred = score >= cfg.feasibility.tier_uncertain

        # FreeCAD ground truth
        record = oracle.check_freecad(params, source="validation")
        gt = record.feasible

        correct = surr_pred == gt
        if surr_pred and gt:
            tp += 1
        elif surr_pred and not gt:
            fp += 1
        elif not surr_pred and not gt:
            tn += 1
        else:
            fn += 1

        mark = "OK" if correct else "MISS"
        print(f"{i:3d} {n_pert:9d} {score:10.4f} {tier:>14} {str(gt):>8} {mark:>8}")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print(f"\n{'='*65}")
    print(f"Confusion Matrix (threshold=tier_uncertain={cfg.feasibility.tier_uncertain})")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")


if __name__ == "__main__":
    main()
