"""
Shared pytest fixtures for proto4-claude tests.

Provides:
  - Module aliasing for hyphenated directory (proto4-claude → proto4_claude)
  - Temporary directories and config files
  - Test CSV curves (target + result)
  - Mock ML model for CAD gate
  - Proto4Config instances
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

# ------------------------------------------------------------------
# Module aliasing: map 'proto4_claude' to 'src/proto4-claude'
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
_SRC_DIR = PROJECT_ROOT / "src"
_PKG_DIR = _SRC_DIR / "proto4-claude"

if "proto4_claude" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "proto4_claude",
        str(_PKG_DIR / "__init__.py"),
        submodule_search_locations=[str(_PKG_DIR)],
    )
    _mod = importlib.util.module_from_spec(spec)
    sys.modules["proto4_claude"] = _mod
    spec.loader.exec_module(_mod)

    # Register sub-modules so `from proto4_claude.X import Y` works.
    # NOTE: Do NOT pass submodule_search_locations — leaving it as None
    # marks these as regular modules (not packages).  If set to [],
    # Python treats each submodule as a package and relative imports
    # inside them resolve to new child modules (e.g.
    # proto4_claude.objective.geometry_adapter) instead of the already-
    # registered proto4_claude.geometry_adapter, breaking isinstance().
    for py_file in _PKG_DIR.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"proto4_claude.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name, str(py_file),
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sys.modules[sub_name] = sub_mod
            sub_spec.loader.exec_module(sub_mod)


# ------------------------------------------------------------------
# Mock ML model (sklearn-compatible predict_proba API)
# ------------------------------------------------------------------

class MockFeasibilityModel:
    """Mimics sklearn classifier with predict_proba().

    Feasibility rule: all params must be within [low, high].
    Points near the boundary have lower feasibility confidence.
    """

    def __init__(self, low: float = 8.0, high: float = 18.0) -> None:
        self.low = low
        self.high = high

    def predict_proba(self, X: list[list[float]]) -> list[list[float]]:
        results = []
        for row in X:
            violations = sum(
                max(0, self.low - v) + max(0, v - self.high)
                for v in row
            )
            feasible_prob = max(0.0, min(1.0, 1.0 - violations / 10.0))
            results.append([1.0 - feasible_prob, feasible_prob])
        return results


# ------------------------------------------------------------------
# CSV curve generation
# ------------------------------------------------------------------

def _make_curve(
    disp_max: float = 0.5,
    n_points: int = 50,
    peak_force: float = 100.0,
    noise_std: float = 0.0,
    include_unloading: bool = True,
) -> pd.DataFrame:
    """Generate a synthetic force-displacement curve."""
    rng = np.random.default_rng(42)

    d_load = np.linspace(0.0, disp_max, n_points)
    f_load = peak_force * (d_load / disp_max) ** 1.5

    if include_unloading:
        d_unload = np.linspace(disp_max, 0.0, n_points)[1:]
        f_unload = 0.7 * peak_force * (d_unload / disp_max) ** 1.5
        d_all = np.concatenate([d_load, d_unload])
        f_all = np.concatenate([f_load, f_unload])
    else:
        d_all = d_load
        f_all = f_load

    if noise_std > 0:
        f_all = f_all + rng.normal(0, noise_std, len(f_all))

    return pd.DataFrame({"displacement": d_all, "force": f_all})


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Temporary project layout that mimics the real one."""
    (tmp_path / "config").mkdir()
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "step").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "trials").mkdir()
    (tmp_path / "vexis" / "input").mkdir(parents=True)
    (tmp_path / "vexis" / "results").mkdir(parents=True)
    (tmp_path / "vexis" / "main.py").write_text("print('mock vexis')\n")
    return tmp_path


@pytest.fixture
def target_curve() -> pd.DataFrame:
    return _make_curve(peak_force=100.0, include_unloading=True)


@pytest.fixture
def target_curve_loading_only() -> pd.DataFrame:
    return _make_curve(peak_force=100.0, include_unloading=False)


@pytest.fixture
def result_curve_good() -> pd.DataFrame:
    return _make_curve(peak_force=102.0, noise_std=0.5, include_unloading=True)


@pytest.fixture
def result_curve_bad() -> pd.DataFrame:
    return _make_curve(peak_force=200.0, noise_std=2.0, include_unloading=True)


@pytest.fixture
def target_csv(tmp_path: Path, target_curve: pd.DataFrame) -> Path:
    p = tmp_path / "target_curve.csv"
    target_curve.to_csv(p, index=False)
    return p


@pytest.fixture
def result_csv_good(tmp_path: Path, result_curve_good: pd.DataFrame) -> Path:
    p = tmp_path / "result_good.csv"
    result_curve_good.to_csv(p, index=False)
    return p


@pytest.fixture
def result_csv_bad(tmp_path: Path, result_curve_bad: pd.DataFrame) -> Path:
    p = tmp_path / "result_bad.csv"
    result_curve_bad.to_csv(p, index=False)
    return p


@pytest.fixture
def result_csv_vexis_format(tmp_path: Path) -> Path:
    """Result CSV with VEXIS-style column names."""
    df = _make_curve(peak_force=95.0, include_unloading=True)
    df_renamed = df.rename(columns={
        "displacement": "Stroke",
        "force": "Reaction_Force",
    })
    p = tmp_path / "vexis_result.csv"
    df_renamed.to_csv(p, index=False)
    return p


@pytest.fixture
def mock_model() -> MockFeasibilityModel:
    return MockFeasibilityModel(low=8.0, high=18.0)


@pytest.fixture
def optimizer_config_yaml(tmp_path: Path) -> Path:
    cfg = {
        "optimization": {
            "sampler": "TPE",
            "max_trials": 20,
            "convergence_threshold": 0.001,
            "patience": 10,
            "seed": 42,
            "n_startup_trials": 5,
            "discretization_step": None,
            "objective_type": "single",
        },
        "objective": {
            "type": "rmse",
            "weights": {"rmse": 1.0},
            "features": {},
        },
        "paths": {
            "target_curve": "input/target_curve.csv",
            "input_dir": "input",
            "result_dir": "output",
            "vexis_path": "vexis",
        },
        "logging": {"level": "WARNING"},
    }
    p = tmp_path / "config" / "optimizer_config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    return p


@pytest.fixture
def limits_yaml(tmp_path: Path) -> Path:
    lim = {
        "freecad": {
            "fcstd_path": "input/model.FCStd",
            "sketch_name": "Sketch001",
            "constraints": {
                "width": {"min": 10.0, "max": 20.0},
                "height": {"min": 5.0, "max": 12.0},
            },
            "step_output_dir": "input/step",
            "step_filename_template": "proto4_trial_{trial_id}.step",
            "timeout_sec": 60,
        },
        "cad_gate": {
            "model_path": None,
            "threshold": 0.5,
            "enabled": True,
            "rejection_max_retries": 30,
        },
        "cae": {
            "stroke_range": {"min": 0.0, "max": 0.5},
            "timeout_sec": 60,
            "max_retries": 1,
        },
        "penalty": {
            "base_penalty": 50.0,
            "alpha": 10.0,
            "failure_weights": {
                "cad_infeasible": 1.0,
                "constraint_violation": 1.0,
                "cae_fail": 0.6,
            },
        },
    }
    p = tmp_path / "config" / "proto4_limitations.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(lim, f)
    return p


@pytest.fixture
def proto4_config(optimizer_config_yaml: Path, limits_yaml: Path):
    from proto4_claude.config import load_config
    return load_config(optimizer_config_yaml, limits_yaml)
