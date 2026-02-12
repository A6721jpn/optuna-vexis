"""Tests for production discretization behavior in search space sampling."""

from __future__ import annotations

import optuna

from proto4_codex.config import BoundsSpec, OptimizationSpec
from proto4_codex.search_space import suggest_design_point


def test_dimension_discretization_uses_angle_and_non_angle_steps() -> None:
    spec = OptimizationSpec(
        enable_dimension_discretization=True,
        non_angle_step=0.01,
        angle_step=0.001,
        angle_name_token="ANGLE",
    )
    bounds = [
        BoundsSpec(name="HEIGHT", min=0.95, max=1.05),
        BoundsSpec(name="SHOULDER-ANGLE-IN", min=0.95, max=1.05),
    ]

    seen = {}
    study = optuna.create_study(direction="minimize")

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
    trial = study.trials[0]
    assert trial.distributions["HEIGHT"].step == 0.01
    assert trial.distributions["SHOULDER-ANGLE-IN"].step == 0.001


def test_global_discretization_step_is_used_when_dimension_mode_disabled() -> None:
    spec = OptimizationSpec(enable_dimension_discretization=False)
    bounds = [
        BoundsSpec(name="HEIGHT", min=0.95, max=1.05),
        BoundsSpec(name="SHOULDER-ANGLE-IN", min=0.95, max=1.05),
    ]

    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.trial.Trial) -> float:
        suggest_design_point(
            trial=trial,
            trial_id=trial.number,
            bounds=bounds,
            discretization_step=0.02,
            optimization=spec,
        )
        return 0.0

    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    assert trial.distributions["HEIGHT"].step == 0.02
    assert trial.distributions["SHOULDER-ANGLE-IN"].step == 0.02

