from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import optuna


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

from v1.config import BoundsSpec, ObjectiveSpec, OptimizationSpec, V1Config  # noqa: E402
from v1.freecad_engine import ConstraintSpec, FreecadEngine  # noqa: E402
from v1.reporting import generate_markdown_report  # noqa: E402
from v1.search_space import suggest_design_point  # noqa: E402


def _is_aligned(value: float, step: float, tol: float = 1e-9) -> bool:
    if step <= 0:
        return False
    q = value / step
    return abs(q - round(q)) <= tol


def test_suggest_design_point_quantizes_physical_params_to_steps() -> None:
    spec = OptimizationSpec(
        enable_dimension_discretization=True,
        non_angle_step=0.01,
        angle_step=0.001,
        angle_name_token="ANGLE",
    )
    bounds = [
        BoundsSpec(name="HEIGHT", min=0.95, max=1.05, base_value=2.65),
        BoundsSpec(name="SHOULDER-ANGLE-OUT", min=0.95, max=1.05, base_value=2.186753),
    ]

    seen: dict[str, object] = {}
    sampler = optuna.samplers.RandomSampler(seed=1234)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        point = suggest_design_point(
            trial=trial,
            trial_id=trial.number,
            bounds=bounds,
            optimization=spec,
        )
        seen["point"] = point
        return 0.0

    study.optimize(objective, n_trials=1)
    point = seen["point"]
    assert point is not None
    assert _is_aligned(point.physical_params["HEIGHT"], 0.01)
    assert _is_aligned(point.physical_params["SHOULDER-ANGLE-OUT"], 0.001)


def test_apply_ratios_quantizes_constraint_values_before_sketch_set() -> None:
    engine = FreecadEngine(fcstd_path=Path("dummy.FCStd"))
    engine._sketch = object()

    class _DummyDoc:
        def recompute(self) -> None:
            return None

    engine._doc = _DummyDoc()
    engine._surface = object()
    engine._check_recompute = lambda _doc: True
    engine._check_surface = lambda _surface: True

    captured: dict[str, float] = {}

    def _fake_set_constraint_value(
        _sketch,
        _index: int,
        value: float,
        _ctype: str,
        _angle_unit: str | None,
        constraint_name: str | None = None,
    ) -> None:
        if constraint_name is not None:
            captured[constraint_name] = value

    engine._set_constraint_value = _fake_set_constraint_value
    engine._specs = [
        ConstraintSpec(
            index=0,
            name="HEIGHT",
            ctype="Distance",
            base_value=2.65,
            sketch=object(),
            angle_unit=None,
            physical_step=0.01,
        ),
        ConstraintSpec(
            index=1,
            name="SHOULDER-ANGLE-OUT",
            ctype="Angle",
            base_value=2.186753,
            sketch=object(),
            angle_unit="rad",
            physical_step=0.001,
        ),
    ]
    params = {
        "HEIGHT": 0.95123,
        "SHOULDER-ANGLE-OUT": 0.99057,
    }

    ok = engine.apply_ratios(params)
    assert ok is True
    expected_height = round((2.65 * 0.95123) / 0.01) * 0.01
    expected_angle = round((2.186753 * 0.99057) / 0.001) * 0.001
    assert captured["HEIGHT"] == expected_height
    assert captured["SHOULDER-ANGLE-OUT"] == expected_angle
    assert _is_aligned(captured["HEIGHT"], 0.01)
    assert _is_aligned(captured["SHOULDER-ANGLE-OUT"], 0.001)


def test_report_table_shows_quantized_physical_values(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "trials" / "trial_0").mkdir(parents=True, exist_ok=True)

    cfg = V1Config(
        optimization=OptimizationSpec(
            objective_type="multi",
            directions=["maximize", "maximize"],
            enable_dimension_discretization=True,
            non_angle_step=0.01,
            angle_step=0.001,
            angle_name_token="ANGLE",
        ),
        objective=ObjectiveSpec(
            type="multi",
            include_rmse_in_multi=False,
            multi_objectives_use_error=False,
            multi_objectives=["click_ratio", "peak_force"],
            features={
                "click_ratio": {"type": "click_ratio", "column": "force"},
                "peak_force": {"type": "peak_force", "column": "force"},
            },
        ),
        bounds=[
            BoundsSpec(name="HEIGHT", min=0.95, max=1.05, base_value=2.65),
            BoundsSpec(name="SHOULDER-ANGLE-OUT", min=0.95, max=1.05, base_value=2.186753),
        ],
    )

    study = optuna.create_study(directions=["maximize", "maximize"])

    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        trial.suggest_float("HEIGHT", 0.95, 1.05)
        trial.suggest_float("SHOULDER-ANGLE-OUT", 0.95, 1.05)
        return 0.0, 0.2

    study.optimize(objective, n_trials=1)

    trial_payload = {
        "trial_id": 0,
        "design_point": {
            "trial_id": 0,
            "params": dict(study.trials[0].params),
            "physical_params": {
                "HEIGHT": 2.5969989624838874,
                "SHOULDER-ANGLE-OUT": 2.206540013463214,
            },
        },
        "outcome": "cae_success",
        "wall_clock_sec": 10.0,
        "objective_values": {
            "click_ratio": 0.0,
            "peak_force": 0.2,
        },
    }
    (result_dir / "trials" / "trial_0" / "trial_info.json").write_text(
        json.dumps(trial_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = generate_markdown_report(
        result_dir=result_dir,
        study=study,
        cfg=cfg,
        optimizer_config_path="config/optimizer_config.yaml",
        limits_config_path="config/v1_0_limitations.yaml",
        start_time=datetime(2026, 2, 14, 0, 0, 0),
        end_time=datetime(2026, 2, 14, 0, 1, 0),
        actual_sampler_name=type(study.sampler).__name__,
    )
    content = report_path.read_text(encoding="utf-8")

    assert "| 0 | cae_success |" in content
    assert "2.6" in content
    assert "2.207" in content
    assert "2.5969989624838874" not in content
    assert "2.206540013463214" not in content

