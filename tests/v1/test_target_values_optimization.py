from __future__ import annotations

import importlib.util
import textwrap
import sys
from pathlib import Path

import optuna
import pytest


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

from v1.cae_evaluator import feature_error  # noqa: E402
from v1.config import (  # noqa: E402
    BoundsSpec,
    ObjectiveSpec,
    OptimizationSpec,
    V1Config,
    load_config,
)
from v1.objective import ObjectiveOrchestrator  # noqa: E402
from v1.reporting import _objective_labels  # noqa: E402
from v1.runner import _adjust_multi_directions_for_targets  # noqa: E402


def _make_cfg_for_targets() -> V1Config:
    return V1Config(
        optimization=OptimizationSpec(
            objective_type="multi",
            directions=["maximize", "maximize"],
        ),
        objective=ObjectiveSpec(
            include_rmse_in_multi=False,
            multi_objectives=["click_ratio", "peak_force"],
            multi_objectives_use_error=False,
            target_values={"click_ratio": 0.0, "peak_force": 0.2},
            features={
                "click_ratio": {"type": "click_ratio", "column": "force"},
                "peak_force": {"type": "peak_force", "column": "force"},
            },
        ),
        bounds=[BoundsSpec(name="x", min=0.0, max=1.0)],
    )


def test_feature_error_uses_absolute_error_when_target_is_zero() -> None:
    assert feature_error(0.15, 0.0) == 0.15
    assert feature_error(None, 0.0) == 1.0
    assert feature_error(0.22, 0.2) == pytest.approx(0.1)


def test_objective_metric_key_switches_to_error_when_target_values_exist() -> None:
    cfg = _make_cfg_for_targets()
    orchestrator = ObjectiveOrchestrator(
        cfg=cfg,
        cad_gate=object(),
        geometry_adapter=object(),
        cae_evaluator=object(),
        persistence=object(),
    )
    assert orchestrator._metric_key_for_objective("click_ratio") == "click_ratio_error"
    assert orchestrator._metric_key_for_objective("peak_force") == "peak_force_error"


def test_runner_adjusts_directions_to_minimize_target_errors() -> None:
    cfg = _make_cfg_for_targets()
    adjusted = _adjust_multi_directions_for_targets(
        cfg,
        directions=["maximize", "maximize"],
        objective_names=["click_ratio", "peak_force"],
    )
    assert adjusted == ["minimize", "minimize"]


def test_reporting_uses_error_labels_when_target_values_are_set() -> None:
    cfg = _make_cfg_for_targets()
    study = optuna.create_study(directions=["minimize", "minimize"])
    labels = _objective_labels(cfg, study)
    assert labels == ["click_ratio_error", "peak_force_error"]


def test_load_config_reads_target_values_from_optimizer_yaml(tmp_path: Path) -> None:
    optimizer_yaml = tmp_path / "optimizer.yaml"
    limits_yaml = tmp_path / "limits.yaml"

    optimizer_yaml.write_text(
        textwrap.dedent(
            """
            optimization:
              sampler: RANDOM
              max_trials: 3
              objective_type: multi
              directions: ["maximize", "maximize"]
            objective:
              type: multi
              include_rmse_in_multi: false
              multi_objectives_use_error: false
              multi_objectives: ["click_ratio", "peak_force"]
              target_values:
                click_ratio: 0.0
                peak_force: 0.20
              features:
                click_ratio:
                  type: click_ratio
                  column: force
                peak_force:
                  type: peak_force
                  column: force
            paths:
              target_curve: input/target_curve_generated.csv
              input_dir: input
              result_dir: output/test
              vexis_path: vexis
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    limits_yaml.write_text(
        textwrap.dedent(
            """
            freecad:
              constraints:
                X:
                  min: 0.95
                  max: 1.05
                  base_value: 1.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(optimizer_yaml, limits_yaml)
    assert cfg.objective.target_values["click_ratio"] == 0.0
    assert cfg.objective.target_values["peak_force"] == 0.2
