"""Tests for proto4-codex markdown report generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import optuna

from proto4_codex.config import BoundsSpec, ObjectiveSpec, OptimizationSpec, Proto4Config
from proto4_codex.reporting import generate_markdown_report


def _write_trial_info(
    result_dir: Path,
    trial_id: int,
    *,
    outcome: str,
    params: dict[str, float],
    metrics: dict[str, float],
) -> None:
    trial_dir = result_dir / "trials" / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_id": trial_id,
        "design_point": {"trial_id": trial_id, "params": params},
        "outcome": outcome,
        "wall_clock_sec": 1.23,
        "objective_values": metrics,
    }
    (trial_dir / "trial_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_generate_markdown_report_single_objective(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = Proto4Config(
        optimization=OptimizationSpec(sampler="TPE", objective_type="single", seed=7),
        objective=ObjectiveSpec(type="rmse", weights={"rmse": 1.0}, features={}),
        bounds=[BoundsSpec(name="x", min=0.0, max=1.0)],
    )

    study = optuna.create_study(direction="minimize")

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return (x - 0.3) ** 2

    study.optimize(objective, n_trials=2)
    _write_trial_info(
        result_dir, 0, outcome="cae_success",
        params={"x": study.trials[0].params["x"]},
        metrics={"rmse": study.trials[0].value},
    )
    _write_trial_info(
        result_dir, 1, outcome="cae_success",
        params={"x": study.trials[1].params["x"]},
        metrics={"rmse": study.trials[1].value},
    )

    report_path = generate_markdown_report(
        result_dir=result_dir,
        study=study,
        cfg=cfg,
        optimizer_config_path="config/optimizer_config.yaml",
        limits_config_path="config/proto4_limitations.yaml",
        start_time=datetime(2026, 2, 11, 1, 0, 0),
        end_time=datetime(2026, 2, 11, 1, 1, 0),
        actual_sampler_name=type(study.sampler).__name__,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "CAE成功（完遂）数 / イテレーション総数" in content
    assert "## 2. Optuna設定" in content
    assert "## 3. 各イテレーションの特徴量と評価関数結果" in content
    assert "## 4. 目標関数の最適化時系列進捗グラフ" in content
    assert "| trial_id | outcome | objective_1 |" in content


def test_generate_markdown_report_multi_objective_shows_pareto_section(tmp_path: Path) -> None:
    result_dir = tmp_path / "result_multi"
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = Proto4Config(
        optimization=OptimizationSpec(sampler="NSGAII", objective_type="multi", seed=11),
        objective=ObjectiveSpec(
            type="multi",
            weights={"rmse": 1.0, "peak_force": 1.0},
            features={"peak_force": {"type": "max", "column": "force"}},
        ),
        bounds=[BoundsSpec(name="x", min=0.0, max=1.0)],
    )

    study = optuna.create_study(directions=["minimize", "minimize"])

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return (x - 0.2) ** 2, (x - 0.8) ** 2

    study.optimize(objective, n_trials=3)
    for t in study.trials:
        _write_trial_info(
            result_dir,
            t.number,
            outcome="cae_success",
            params={"x": t.params["x"]},
            metrics={"rmse": t.values[0], "peak_force_error": t.values[1]},
        )

    report_path = generate_markdown_report(
        result_dir=result_dir,
        study=study,
        cfg=cfg,
        optimizer_config_path="config/optimizer_config.yaml",
        limits_config_path="config/proto4_limitations.yaml",
        start_time=datetime(2026, 2, 11, 2, 0, 0),
        end_time=datetime(2026, 2, 11, 2, 1, 0),
        actual_sampler_name=type(study.sampler).__name__,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "## 5. パレート図（2目的時のみ）" in content
    assert "対象外です" not in content
