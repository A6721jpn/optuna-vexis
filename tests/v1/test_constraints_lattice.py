from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _ensure_v1_package_loaded() -> None:
    if "v1" in sys.modules:
        return
    project_root = Path(__file__).resolve().parents[2]
    pkg_dir = project_root / "src" / "v1"
    spec = importlib.util.spec_from_file_location(
        "v1",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load v1 package for tests")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1"] = mod
    spec.loader.exec_module(mod)


_ensure_v1_package_loaded()

from v1.config import BoundsSpec, OptimizationSpec  # noqa: E402
from v1.constraints import check_hard_constraints  # noqa: E402
from v1.types import DesignPoint  # noqa: E402


def _optimization_spec() -> OptimizationSpec:
    return OptimizationSpec(
        enable_dimension_discretization=True,
        non_angle_step=0.01,
        angle_step=0.001,
        angle_name_token="ANGLE",
    )


def test_hard_constraint_accepts_boundary_point_on_discrete_lattice() -> None:
    bounds = [
        BoundsSpec(
            name="SHOUDER-T",
            min=0.9000000000000001,
            max=1.1,
            base_value=0.3,
        ),
    ]
    point = DesignPoint(trial_id=0, params={"SHOUDER-T": 0.9})

    assert check_hard_constraints(point, bounds, optimization=_optimization_spec()) is None


def test_hard_constraint_rejects_off_lattice_point() -> None:
    bounds = [
        BoundsSpec(
            name="SHOUDER-T",
            min=0.9,
            max=1.1,
            base_value=0.3,
        ),
    ]
    point = DesignPoint(trial_id=0, params={"SHOUDER-T": 0.901})

    reason = check_hard_constraints(point, bounds, optimization=_optimization_spec())
    assert reason is not None
    assert "off lattice" in reason
