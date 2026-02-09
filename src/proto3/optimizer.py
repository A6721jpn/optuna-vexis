"""
Proto3 Optimizer Module

Generic Optuna wrapper for scalar parameter bounds.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
from optuna.samplers import TPESampler, RandomSampler, QMCSampler, CmaEsSampler
from optuna.study import Study
from optuna.trial import Trial

try:
    from optuna.samplers import GPSampler
    HAS_GP_SAMPLER = True
except ImportError:
    HAS_GP_SAMPLER = False

try:
    from optuna.samplers import NSGAIISampler
    HAS_NSGAII = True
except ImportError:
    HAS_NSGAII = False

try:
    import optunahub
    _auto_sampler_module = optunahub.load_module(package="samplers/auto_sampler")
    AutoSampler = _auto_sampler_module.AutoSampler
    HAS_AUTO_SAMPLER = True
except Exception:
    HAS_AUTO_SAMPLER = False


logger = logging.getLogger(__name__)


class Optimizer:
    """
    Optuna wrapper for Proto3.

    Bounds format (scalar):
        {
          "width": {"min": 10.0, "max": 20.0},
          "height": {"min": 5.0, "max": 12.0}
        }
    """

    def __init__(
        self,
        bounds: dict[str, dict[str, float]],
        config: dict,
        mode: str = "all",
        storage_path: Optional[str | Path] = None,
    ) -> None:
        self.bounds = bounds
        self.config = config
        self.mode = mode
        self._study: Optional[Study] = None
        self._current_trial: Optional[Trial] = None

        step = self.config.get("discretization_step")
        if step is not None:
            warnings.filterwarnings(
                "ignore",
                message=".*range is not divisible by.*",
                category=UserWarning,
                module="optuna.distributions",
            )

        if storage_path:
            storage_path = Path(storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage = f"sqlite:///{storage_path}"
        else:
            self._storage = None

        sampler_name = config.get("sampler", "TPE").upper()
        self._sampler = self._create_sampler(sampler_name)
        self._directions = config.get("directions", ["minimize"])

        logger.info(
            "Optimizer init sampler=%s, mode=%s, directions=%s",
            type(self._sampler).__name__,
            mode,
            self._directions,
        )

    def _create_sampler(self, name: str) -> Any:
        seed = self.config.get("seed")
        n_startup = self.config.get("n_startup_trials", 10)

        if name == "AUTO":
            if HAS_AUTO_SAMPLER:
                logger.info("Using AutoSampler")
                return AutoSampler(seed=seed)
            logger.warning("AutoSampler unavailable; falling back to TPE")
            return TPESampler(seed=seed)

        current_trials = 0
        if self._storage:
            try:
                summaries = optuna.study.get_all_study_summaries(self._storage)
                if summaries:
                    current_trials = summaries[0].n_trials
            except Exception as exc:
                logger.warning("Failed to read trial count: %s", exc)

        if current_trials < n_startup:
            logger.info("Startup phase: QMCSampler(Sobol)")
            return QMCSampler(scramble=True, seed=seed)

        logger.info("Main phase: %sSampler", name)

        if name == "TPE":
            return TPESampler(seed=seed, n_startup_trials=0)
        if name == "GP":
            if HAS_GP_SAMPLER:
                return GPSampler(seed=seed, n_startup_trials=0, warn_independent_sampling=False)
            logger.warning("GPSampler unavailable; falling back to TPE")
            return TPESampler(seed=seed, n_startup_trials=0)
        if name == "NSGAII":
            if HAS_NSGAII:
                return NSGAIISampler(seed=seed)
            logger.warning("NSGAIISampler unavailable; falling back to TPE")
            return TPESampler(seed=seed, n_startup_trials=0)
        if name == "RANDOM":
            return RandomSampler(seed=seed)
        if name in ["CMA-ES", "CMAES"]:
            return CmaEsSampler(seed=seed, n_startup_trials=0)

        logger.warning("Unknown sampler %s; falling back to TPE", name)
        return TPESampler(seed=seed, n_startup_trials=0)

    def create_study(self, study_name: str = "proto3_optimization") -> Study:
        self._study = optuna.create_study(
            study_name=study_name,
            sampler=self._sampler,
            directions=self._directions,
            storage=self._storage,
            load_if_exists=True,
        )
        logger.info("Study created: %s", study_name)
        return self._study

    def _suggest_params(self, trial: Trial) -> dict[str, float]:
        params: dict[str, float] = {}
        step = self.config.get("discretization_step")
        for name, bound in self.bounds.items():
            if not isinstance(bound, dict):
                continue
            low = bound.get("min")
            high = bound.get("max")
            if low is None or high is None:
                continue
            if step is not None:
                params[name] = trial.suggest_float(name, low, high, step=step)
            else:
                params[name] = trial.suggest_float(name, low, high)
        return params

    def run_optimization(
        self,
        objective_func: Callable[[dict[str, float]], float],
        base_params: dict,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        callbacks: Optional[list] = None,
    ) -> Study:
        if self._study is None:
            self.create_study()

        if n_trials is None:
            n_trials = self.config.get("max_trials", 100)

        completed_trials = self.get_n_trials()
        remaining_trials = n_trials - completed_trials
        if remaining_trials <= 0:
            logger.info(
                "Already completed %s trials; no additional trials", completed_trials
            )
            return self._study

        def wrapped_objective(trial: Trial) -> float:
            params = self._suggest_params(trial)
            return objective_func(params)

        logger.info(
            "Optimization start: total=%s, remaining=%s",
            n_trials,
            remaining_trials,
        )

        self._study.optimize(
            wrapped_objective,
            n_trials=remaining_trials,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=False,
        )

        logger.info("Optimization completed")
        return self._study

    def get_best_params(self, base_params: dict) -> Optional[dict[str, float]]:
        if self._study is None:
            return None
        try:
            return dict(self._study.best_params)
        except (ValueError, RuntimeError):
            return None

    def get_best_value(self) -> Optional[float]:
        if self._study is None:
            return None
        try:
            return self._study.best_value
        except (ValueError, RuntimeError):
            return None

    def get_n_trials(self) -> int:
        if self._study is None:
            return 0
        return len(self._study.trials)

    def is_converged(self, threshold: float) -> bool:
        best_value = self.get_best_value()
        if best_value is None:
            return False
        if isinstance(best_value, (list, tuple)):
            best_value = best_value[0] if best_value else None
            if best_value is None:
                return False
        return best_value <= threshold

    def get_study_summary(self, base_params: dict) -> dict:
        if self._study is None:
            return {}
        return {
            "n_trials": self.get_n_trials(),
            "best_params": self.get_best_params(base_params),
            "best_value": self.get_best_value(),
            "sampler": type(self._sampler).__name__,
            "optimization_mode": self.mode,
        }


class ConvergenceCallback:
    def __init__(self, threshold: float, patience: int = 20) -> None:
        self.threshold = threshold
        self.patience = patience
        self._best_value = float("inf")
        self._no_improvement_count = 0

    def __call__(self, study: Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            current_value = trial.value
        except RuntimeError:
            if trial.values:
                current_value = trial.values[0]
            else:
                return

        if current_value is None:
            return

        if current_value <= self.threshold:
            logger.info("Converged: %s <= %s", current_value, self.threshold)
            study.stop()
            return

        if current_value < self._best_value:
            self._best_value = current_value
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        if self._no_improvement_count >= self.patience:
            logger.info("No improvement for %s trials; stopping", self.patience)
            study.stop()
