"""Test progressive-perturbation discovery strategy.

Run Phase 0 with 30 points using the new 3-tier sampling:
  Tier A (30%): 1-3 dims perturbed
  Tier B (40%): 5-10 dims perturbed
  Tier C (30%): all dims perturbed
"""

import importlib.util
import sys
import os
import logging

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(ROOT, "src", "v2-claude")
MOD_NAME = "v2_claude"

# Register v2-claude as v2_claude module
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

from v2_claude.config import load_config
from v2_claude.geometry_adapter import GeometryAdapter
from v2_claude.feasibility_oracle import FeasibilityOracle


def main():
    from pathlib import Path
    cfg = load_config(
        Path(ROOT) / "config" / "optimizer_config.yaml",
        Path(ROOT) / "config" / "v2_claude_limitations.yaml",
    )

    from pathlib import Path as _P
    geometry_adapter = GeometryAdapter(cfg.freecad, project_root=_P(ROOT))

    # Override margin to 0.05 for this test
    cfg.feasibility.discovery_bounds_margin = 0.05
    cfg.feasibility.db_path = "output/feasibility_db_test.json"

    oracle = FeasibilityOracle(
        geometry_adapter=geometry_adapter,
        bounds=cfg.bounds,
        feasibility_spec=cfg.feasibility,
    )

    # Run discovery with 30 points
    summary = oracle.run_discovery(n_points=30)

    print("\n" + "=" * 60)
    print("Phase 0 Discovery Summary (progressive perturbation)")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Analyze feasibility by # dims perturbed
    records = oracle.db.records
    print(f"\nTotal records: {len(records)}")
    print(f"Feasible: {oracle.db.feasible_count}")
    print(f"Infeasible: {oracle.db.infeasible_count}")

    baseline = 1.0
    tier_a = []  # 1-3 dims
    tier_b = []  # 5-10 dims
    tier_c = []  # >10 dims
    for r in records:
        n_perturbed = sum(1 for v in r.params.values() if abs(v - baseline) > 0.001)
        tag = "A" if n_perturbed <= 3 else ("B" if n_perturbed <= 10 else "C")
        bucket = tier_a if tag == "A" else (tier_b if tag == "B" else tier_c)
        bucket.append(r.feasible)

    def _rate(lst):
        if not lst:
            return "N/A"
        return f"{sum(lst)}/{len(lst)} ({sum(lst)/len(lst)*100:.0f}%)"

    print(f"\nTier A (1-3 dims): {_rate(tier_a)}")
    print(f"Tier B (5-10 dims): {_rate(tier_b)}")
    print(f"Tier C (>10 dims): {_rate(tier_c)}")

    print(f"\nSurrogate trained: {oracle.surrogate.is_trained}")
    print(f"Surrogate train count: {oracle.surrogate.train_count}")


if __name__ == "__main__":
    main()
