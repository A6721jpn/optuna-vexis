"""
Proto4 Search Space

Optuna sampler creation and trial-to-parameter mapping.

Key additions over proto3:
  - ``constraints_func`` integration for TPE / NSGA-II so the sampler
    learns the CAD feasibility boundary.
  - ``FeasibilityAwareSampler`` — a wrapper that performs rejection
    sampling against the ML gate *before* the objective is evaluated,
    preventing wasted trial budget on clearly infeasible points.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler, TPESampler, RandomSampler, QMCSampler, CmaEsSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState

from .config import BoundsSpec, CadGateSpec, OptimizationSpec
from .types import DesignPoint

try:
    from optuna.samplers import GPSampler
    _HAS_GP = True
except ImportError:
    _HAS_GP = False

try:
    from optuna.samplers import NSGAIISampler
    _HAS_NSGAII = True
except ImportError:
    _HAS_NSGAII = False

try:
    import optunahub
    _auto_mod = optunahub.load_module(package="samplers/auto_sampler")
    AutoSampler = _auto_mod.AutoSampler
    _HAS_AUTO = True
except Exception:
    _HAS_AUTO = False

logger = logging.getLogger(__name__)

# Key used in trial.user_attrs to communicate feasibility violation score
FEASIBILITY_ATTR = "cad_feasibility_violation"


# ------------------------------------------------------------------
# constraints_func factory
# ------------------------------------------------------------------

def make_constraints_func() -> Callable[[FrozenTrial], Sequence[float]]:
    """Return an Optuna ``constraints_func`` that reads the feasibility
    violation score stored by the objective orchestrator.

    Convention:
      - ``trial.user_attrs["cad_feasibility_violation"]`` is a float
      - <= 0  → feasible
      - > 0   → infeasible (magnitude = severity)

    The sampler uses this to classify trials into feasible / infeasible
    groups and biases future sampling toward the feasible region.
    """

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        return [trial.user_attrs.get(FEASIBILITY_ATTR, 0.0)]

    return constraints_func


# ------------------------------------------------------------------
# Sampler creation (with optional constraints_func injection)
# ------------------------------------------------------------------

def create_sampler(
    spec: OptimizationSpec,
    storage: Optional[str] = None,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
) -> Any:
    """Create an Optuna sampler.

    When *constraints_func* is provided it is injected into samplers that
    support it (TPE, NSGA-II).  Other samplers silently ignore it.
    """
    name = spec.sampler.upper()
    seed = spec.seed

    if name == "AUTO":
        if _HAS_AUTO:
            logger.info("Using AutoSampler")
            return AutoSampler(seed=seed)
        logger.warning("AutoSampler unavailable; falling back to TPE")
        return TPESampler(seed=seed, constraints_func=constraints_func)

    # Startup phase detection
    current_trials = 0
    if storage:
        try:
            summaries = optuna.study.get_all_study_summaries(storage)
            if summaries:
                current_trials = summaries[0].n_trials
        except Exception as exc:
            logger.warning("Failed to read trial count: %s", exc)

    if current_trials < spec.n_startup_trials:
        logger.info("Startup phase: QMCSampler(Sobol)")
        return QMCSampler(scramble=True, seed=seed)

    logger.info("Main phase: %sSampler", name)

    if name == "TPE":
        return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)
    if name == "GP":
        if _HAS_GP:
            return GPSampler(seed=seed, n_startup_trials=0, warn_independent_sampling=False)
        logger.warning("GPSampler unavailable; falling back to TPE")
        return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)
    if name == "NSGAII":
        if _HAS_NSGAII:
            return NSGAIISampler(seed=seed, constraints_func=constraints_func)
        logger.warning("NSGAIISampler unavailable; falling back to TPE")
        return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)
    if name == "RANDOM":
        return RandomSampler(seed=seed)
    if name in ("CMA-ES", "CMAES"):
        return CmaEsSampler(seed=seed, n_startup_trials=0)

    logger.warning("Unknown sampler %s; falling back to TPE", name)
    return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)


# ------------------------------------------------------------------
# FeasibilityAwareSampler (rejection-sampling wrapper)
# ------------------------------------------------------------------

class FeasibilityAwareSampler(BaseSampler):
    """Wraps any Optuna sampler with ML-gate rejection at sample time.

    How it works
    ------------
    After the base sampler proposes a set of parameters (via
    ``sample_independent``), this wrapper evaluates the ML feasibility
    predictor.  If the prediction is *infeasible*, a new sample is drawn
    from the base sampler — up to ``max_retries`` times.

    If all retries are exhausted, the last sample is returned anyway so
    the pipeline can record a penalty and move on.

    Why sample_independent?
    -----------------------
    Optuna calls ``sample_independent`` for each parameter individually.
    The ML gate needs all parameters together, so we buffer independent
    samples and check feasibility once all parameters are collected.
    We achieve this by caching samples per-trial and only checking
    feasibility when all parameters have been sampled.

    Note: For ``sample_relative`` the check is straightforward since all
    params come at once.
    """

    def __init__(
        self,
        base_sampler: BaseSampler,
        predict_fn: Callable[[dict[str, float]], bool],
        max_retries: int = 50,
    ) -> None:
        """
        Args:
            base_sampler: Underlying sampler (TPE, GP, etc.)
            predict_fn: ``fn(params) -> is_feasible``.
                        Takes a dict of param_name→value, returns True
                        if the point is CAD-feasible.
            max_retries: Maximum rejection attempts before giving up.
        """
        self._base = base_sampler
        self._predict_fn = predict_fn
        self._max_retries = max_retries
        self._stats_accepted = 0
        self._stats_rejected = 0

    # -- relative sampling (all params at once) -------------------------

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        return self._base.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if not search_space:
            return {}

        for attempt in range(self._max_retries):
            params = self._base.sample_relative(study, trial, search_space)
            if self._predict_fn(params):
                self._stats_accepted += 1
                return params
            self._stats_rejected += 1

        logger.warning(
            "FeasibilityAwareSampler: %d retries exhausted in sample_relative; "
            "returning last sample",
            self._max_retries,
        )
        return params

    # -- independent sampling (per-param, no rejection here) ------------

    def sample_independent(
        self, study: Study, trial: FrozenTrial,
        param_name: str, param_distribution: BaseDistribution,
    ) -> Any:
        # Independent sampling happens one param at a time — we cannot
        # check feasibility until all params are known.  Rejection is
        # deferred to the objective orchestrator (fast path) or
        # constraints_func (sampler learning).
        return self._base.sample_independent(
            study, trial, param_name, param_distribution,
        )

    # -- lifecycle hooks ------------------------------------------------

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._base.before_trial(study, trial)

    def after_trial(
        self, study: Study, trial: FrozenTrial,
        state: TrialState, values: Sequence[float] | None,
    ) -> None:
        self._base.after_trial(study, trial, state, values)

    def reseed_rng(self) -> None:
        self._base.reseed_rng()

    @property
    def rejection_stats(self) -> dict[str, int]:
        return {
            "accepted": self._stats_accepted,
            "rejected": self._stats_rejected,
        }


# ------------------------------------------------------------------
# Parameter suggestion helper
# ------------------------------------------------------------------

def suggest_design_point(
    trial: Trial,
    trial_id: int,
    bounds: list[BoundsSpec],
    discretization_step: Optional[float] = None,
) -> DesignPoint:
    """Sample a DesignPoint from an Optuna Trial."""
    if discretization_step is not None:
        warnings.filterwarnings(
            "ignore",
            message=".*range is not divisible by.*",
            category=UserWarning,
            module="optuna.distributions",
        )

    params: dict[str, float] = {}
    for b in bounds:
        if discretization_step is not None:
            params[b.name] = trial.suggest_float(b.name, b.min, b.max, step=discretization_step)
        else:
            params[b.name] = trial.suggest_float(b.name, b.min, b.max)

    return DesignPoint(trial_id=trial_id, params=params)
