"""Tests for runner convergence callback behavior."""

from __future__ import annotations

import optuna

from proto4_codex.runner import ConvergenceCallback


def test_convergence_callback_ignores_multi_objective_trials() -> None:
    study = optuna.create_study(directions=["maximize", "maximize"])

    def objective(trial: optuna.trial.Trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return x, 1.0 - x

    study.optimize(objective, n_trials=1)
    cb = ConvergenceCallback(threshold=0.9, patience=1)
    cb(study, study.trials[0])  # must not raise

