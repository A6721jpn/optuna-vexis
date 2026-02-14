"""
v1.0 Search Space

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
import math
import warnings
from typing import Any, Callable, Optional, Sequence

import optuna
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import BaseSampler, TPESampler, RandomSampler, QMCSampler, CmaEsSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial, TrialState

from .config import BoundsSpec, OptimizationSpec
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
    from optuna.samplers import NSGAIIISampler
    _HAS_NSGAIII = True
except ImportError:
    _HAS_NSGAIII = False

try:
    import optunahub
    _auto_mod = optunahub.load_module(package="samplers/auto_sampler")
    AutoSampler = _auto_mod.AutoSampler
    _HAS_AUTO = True
    _AUTO_LOAD_ERROR: str | None = None
except Exception as exc:
    _HAS_AUTO = False
    _AUTO_LOAD_ERROR = str(exc)

logger = logging.getLogger(__name__)

# Key used in trial.user_attrs to communicate feasibility violation score
FEASIBILITY_ATTR = "v1_0_feasibility_violation"
LEGACY_FEASIBILITY_ATTR = "proto4_feasibility_violation"


# ------------------------------------------------------------------
# constraints_func factory
# ------------------------------------------------------------------

def make_constraints_func() -> Callable[[FrozenTrial], Sequence[float]]:
    """Return an Optuna ``constraints_func`` that reads the feasibility
    violation score stored by the objective orchestrator.

    Convention:
      - ``trial.user_attrs["v1_0_feasibility_violation"]`` is a float
      - <= 0  → feasible
      - > 0   → infeasible (magnitude = severity)

    The sampler uses this to classify trials into feasible / infeasible
    groups and biases future sampling toward the feasible region.
    """

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        if FEASIBILITY_ATTR in trial.user_attrs:
            return [trial.user_attrs[FEASIBILITY_ATTR]]
        if LEGACY_FEASIBILITY_ATTR in trial.user_attrs:
            return [trial.user_attrs[LEGACY_FEASIBILITY_ATTR]]
        return [0.0]

    return constraints_func


# ------------------------------------------------------------------
# Static search-space builder
# ------------------------------------------------------------------

def _base_value_for_bound(bound: BoundsSpec) -> float:
    try:
        base = float(bound.base_value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(base) or base == 0.0:
        return 1.0
    return base


def _physical_discretization_step(
    bound: BoundsSpec,
    optimization: Optional[OptimizationSpec] = None,
    discretization_step: Optional[float] = None,
) -> Optional[float]:
    physical_step: float | None = None
    angle_token = ""
    if optimization is not None:
        angle_token = optimization.angle_name_token.upper()

    if optimization is not None and optimization.enable_dimension_discretization:
        is_angle = bool(angle_token) and angle_token in bound.name.upper()
        physical_step = optimization.angle_step if is_angle else optimization.non_angle_step
    elif discretization_step is not None:
        # Backward compatibility: legacy step is now interpreted in physical domain.
        physical_step = discretization_step

    if physical_step is None:
        return None

    try:
        step = float(physical_step)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(step) or step <= 0.0:
        return None
    return step


def _ratio_sampling_spec(
    bound: BoundsSpec,
    optimization: Optional[OptimizationSpec] = None,
    discretization_step: Optional[float] = None,
) -> tuple[float, float, Optional[float], Optional[float]]:
    """Return ratio-domain (low, high, step) aligned to physical-step grid.

    The physical grid is anchored at multiples of `physical_step` (e.g. 0.01),
    not at the ratio lower bound.
    """
    ratio_lo = float(bound.min)
    ratio_hi = float(bound.max)
    if ratio_lo > ratio_hi:
        ratio_lo, ratio_hi = ratio_hi, ratio_lo

    physical_step = _physical_discretization_step(
        bound,
        optimization=optimization,
        discretization_step=discretization_step,
    )
    if physical_step is None:
        return ratio_lo, ratio_hi, None, None

    base = _base_value_for_bound(bound)
    abs_base = abs(base)
    if abs_base <= 0.0:
        return ratio_lo, ratio_hi, None, physical_step

    physical_lo = min(ratio_lo * base, ratio_hi * base)
    physical_hi = max(ratio_lo * base, ratio_hi * base)
    tol = physical_step * 1e-9
    idx_lo = math.ceil((physical_lo - tol) / physical_step)
    idx_hi = math.floor((physical_hi + tol) / physical_step)

    # If no grid point falls inside bounds, fall back to continuous ratio domain.
    if idx_lo > idx_hi:
        logger.warning(
            "No physical discretization grid point for %s in [%.6g, %.6g] with step=%.6g; "
            "fallback to continuous sampling",
            bound.name,
            physical_lo,
            physical_hi,
            physical_step,
        )
        return ratio_lo, ratio_hi, None, physical_step

    grid_lo = idx_lo * physical_step
    grid_hi = idx_hi * physical_step
    ratio_grid_lo = grid_lo / base
    ratio_grid_hi = grid_hi / base
    if ratio_grid_lo <= ratio_grid_hi:
        ratio_lo_adj = ratio_grid_lo
        ratio_hi_adj = ratio_grid_hi
    else:
        ratio_lo_adj = ratio_grid_hi
        ratio_hi_adj = ratio_grid_lo

    ratio_step = physical_step / abs_base
    span = ratio_hi_adj - ratio_lo_adj
    if span <= 0.0 or ratio_step > span:
        return ratio_lo_adj, ratio_hi_adj, None, physical_step
    return ratio_lo_adj, ratio_hi_adj, ratio_step, physical_step


def _step_decimal_digits(step: float) -> int:
    s = f"{step:.12f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0


def _ratio_to_physical_value(
    bound: BoundsSpec,
    ratio_value: float,
    *,
    physical_step: Optional[float] = None,
) -> float:
    base = _base_value_for_bound(bound)
    physical = float(ratio_value) * base
    if physical_step is None:
        return physical

    snapped = round(physical / physical_step) * physical_step
    low = min(float(bound.min) * base, float(bound.max) * base)
    high = max(float(bound.min) * base, float(bound.max) * base)
    if snapped < low:
        snapped = low
    elif snapped > high:
        snapped = high
    return round(snapped, _step_decimal_digits(physical_step))


def build_fixed_search_space(
    bounds: list[BoundsSpec],
    optimization: Optional[OptimizationSpec] = None,
    discretization_step: Optional[float] = None,
) -> dict[str, BaseDistribution]:
    """Build a fixed Optuna distribution map from bounds/settings.

    This is used by ``FeasibilityAwareSampler`` so CAD-gate rejection can
    evaluate complete parameter vectors (all dimensions) before accepting
    a sample.
    """

    space: dict[str, BaseDistribution] = {}
    for b in bounds:
        low, high, step, _ = _ratio_sampling_spec(
            b,
            optimization=optimization,
            discretization_step=discretization_step,
        )
        space[b.name] = FloatDistribution(low=low, high=high, step=step)

    return space


# ------------------------------------------------------------------
# Sampler creation (with optional constraints_func injection)
# ------------------------------------------------------------------

def create_sampler(
    spec: OptimizationSpec,
    storage: Optional[str] = None,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
    n_objectives: int = 1,
) -> Any:
    """Create an Optuna sampler.

    When *constraints_func* is provided it is injected into samplers that
    support it (TPE, NSGA-II).  Other samplers silently ignore it.
    """
    name = spec.sampler.upper()
    seed = spec.seed

    # AUTO: use OptunaHub AutoSampler when available.
    # If unavailable, fall back to built-in samplers without long startup,
    # so feasibility feedback can be used from the earliest trials.
    if name == "AUTO":
        if _HAS_AUTO:
            logger.info("Using AutoSampler")
            return AutoSampler(seed=seed)
        logger.warning(
            "AutoSampler unavailable; using built-in fallback "
            "(startup disabled, objective_count=%d, reason=%s)",
            n_objectives,
            _AUTO_LOAD_ERROR or "unknown",
        )
        if n_objectives > 2 and _HAS_NSGAIII:
            return NSGAIIISampler(seed=seed, constraints_func=constraints_func)
        if n_objectives > 1 and _HAS_NSGAII:
            return NSGAIISampler(seed=seed, constraints_func=constraints_func)
        return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)

    # QMC startup phase is applied only to TPE-family samplers.
    # For explicitly selected GP/NSGA/CMA/RANDOM, we use the requested
    # sampler directly to preserve user intent.
    if name in ("TPE", "MOTPE", "MO-TPE"):
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
            return GPSampler(
                seed=seed,
                n_startup_trials=max(0, int(spec.n_startup_trials)),
                warn_independent_sampling=False,
                constraints_func=constraints_func,
            )
        logger.warning("GPSampler unavailable; falling back to TPE")
        return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)
    if name == "NSGAII":
        if _HAS_NSGAII:
            return NSGAIISampler(seed=seed, constraints_func=constraints_func)
        logger.warning("NSGAIISampler unavailable; falling back to TPE")
        return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)
    if name == "NSGAIII":
        if _HAS_NSGAIII:
            return NSGAIIISampler(seed=seed, constraints_func=constraints_func)
        logger.warning("NSGAIIISampler unavailable; falling back to NSGAII/TPE")
        if _HAS_NSGAII:
            return NSGAIISampler(seed=seed, constraints_func=constraints_func)
        return TPESampler(seed=seed, n_startup_trials=0, constraints_func=constraints_func)
    if name in ("MOTPE", "MO-TPE"):
        return TPESampler(
            seed=seed,
            n_startup_trials=0,
            multivariate=True,
            group=True,
            constraints_func=constraints_func,
        )
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
        predict_score_fn: Optional[Callable[[dict[str, float]], float]] = None,
        expected_param_names: Optional[Sequence[str]] = None,
        fixed_search_space: Optional[dict[str, BaseDistribution]] = None,
        repair_fn: Optional[Callable[[dict[str, float]], dict[str, float] | None]] = None,
        max_retries: int = 50,
    ) -> None:
        """
        Args:
            base_sampler: Underlying sampler (TPE, GP, etc.)
            predict_fn: ``fn(params) -> is_feasible``.
                        Takes a dict of param_name→value, returns True
                        if the point is CAD-feasible.
            predict_score_fn: Optional quality score in [0, 1] used to keep
                        the best candidate if retries are exhausted.
            expected_param_names: Full parameter-name list used to avoid
                        evaluating partial vectors during independent sampling.
            fixed_search_space: Static search-space map. When provided, this
                        wrapper performs full-vector sampling in
                        ``sample_relative`` and applies CAD-gate rejection
                        before any parameter is finalized.
            repair_fn: Optional repair callback invoked after retries are
                        exhausted. Must return a full parameter dict or None.
            max_retries: Maximum rejection attempts before giving up.
        """
        self._base = base_sampler
        self._predict_fn = predict_fn
        self._predict_score_fn = predict_score_fn
        self._expected_param_names = (
            set(expected_param_names) if expected_param_names else None
        )
        self._fixed_search_space = dict(fixed_search_space) if fixed_search_space else None
        self._repair_fn = repair_fn
        self._max_retries = max_retries
        self._stats_accepted = 0
        self._stats_rejected = 0
        self._stats_repaired = 0
        self._independent_cache: dict[int, dict[str, float]] = {}

    def _score(self, params: dict[str, float]) -> float:
        if self._predict_score_fn is not None:
            try:
                return float(self._predict_score_fn(params))
            except Exception as exc:
                logger.debug("predict_score_fn failed; fallback to binary score: %s", exc)
        return 1.0 if self._predict_fn(params) else 0.0

    def _sample_full_candidate(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
        # Some base samplers return only a subset from sample_relative.
        # Fill missing dimensions via sample_independent to always evaluate
        # a complete vector at the CAD gate.
        params = dict(self._base.sample_relative(study, trial, search_space) or {})
        for name, dist in search_space.items():
            if name in params:
                continue
            params[name] = self._base.sample_independent(study, trial, name, dist)
        return params

    @staticmethod
    def _aligned_value(value: float, dist: BaseDistribution) -> float:
        if not isinstance(dist, FloatDistribution):
            return float(value)

        v = float(value)
        lo = float(dist.low)
        hi = float(dist.high)
        if v < lo:
            v = lo
        elif v > hi:
            v = hi

        if dist.step is not None:
            step = float(dist.step)
            idx = round((v - lo) / step)
            v = lo + idx * step
            if v < lo:
                v = lo
            elif v > hi:
                v = hi

            step_str = f"{step:.12f}".rstrip("0")
            if "." in step_str:
                digits = len(step_str.split(".")[1])
                v = round(v, digits + 2)

        return float(v)

    def _align_params_to_space(
        self,
        params: dict[str, float],
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
        aligned = dict(params)
        for name, dist in search_space.items():
            if name not in aligned:
                continue
            aligned[name] = self._aligned_value(float(aligned[name]), dist)
        return aligned

    # -- relative sampling (all params at once) -------------------------

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        # Always call base sampler first because some samplers (e.g. TPE
        # with group=True) initialize internal state in this hook.
        base_space = self._base.infer_relative_search_space(study, trial)
        if self._fixed_search_space is not None:
            return dict(self._fixed_search_space)
        return base_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if not search_space:
            return {}

        best_params: dict[str, float] | None = None
        best_score = float("-inf")
        last_params: dict[str, float] = {}

        retries = max(1, self._max_retries)
        for _ in range(retries):
            params = self._sample_full_candidate(study, trial, search_space)
            params = self._align_params_to_space(params, search_space)
            last_params = params
            if self._predict_fn(params):
                self._stats_accepted += 1
                return params
            self._stats_rejected += 1
            score = self._score(params)
            if score > best_score:
                best_score = score
                best_params = dict(params)

        if self._repair_fn is not None:
            candidate = dict(best_params) if best_params is not None else dict(last_params)
            repaired = self._repair_fn(candidate)
            if repaired is not None:
                repaired = self._align_params_to_space(repaired, search_space)
            if repaired is not None and self._predict_fn(repaired):
                self._stats_accepted += 1
                self._stats_repaired += 1
                return repaired

        logger.warning(
            "FeasibilityAwareSampler: %d retries exhausted in sample_relative; "
            "returning best candidate (score=%.4f)",
            self._max_retries,
            best_score if best_score != float("-inf") else -1.0,
        )
        if best_params is not None:
            return best_params
        return last_params

    # -- independent sampling (per-param, with feasibility rejection) ---

    def sample_independent(
        self, study: Study, trial: FrozenTrial,
        param_name: str, param_distribution: BaseDistribution,
    ) -> Any:
        # When fixed search-space is enabled we evaluate full vectors in
        # sample_relative. Keep independent path pass-through to avoid
        # partial-vector CAD checks.
        if self._fixed_search_space is not None:
            return self._base.sample_independent(
                study, trial, param_name, param_distribution,
            )

        trial_id = trial.number
        cache = self._independent_cache.setdefault(trial_id, {})

        expected_names = (
            set(self._expected_param_names)
            if self._expected_param_names is not None
            else set(getattr(trial, "distributions", {}).keys())
        )
        expected_names.add(param_name)

        last_value = None
        for _ in range(self._max_retries):
            last_value = self._base.sample_independent(
                study, trial, param_name, param_distribution,
            )
            cache[param_name] = last_value

            if len(cache) < len(expected_names):
                return last_value

            if self._predict_fn(dict(cache)):
                self._stats_accepted += 1
                self._independent_cache.pop(trial_id, None)
                return last_value

            self._stats_rejected += 1

        self._independent_cache.pop(trial_id, None)
        return last_value

    # -- lifecycle hooks ------------------------------------------------

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._base.before_trial(study, trial)

    def after_trial(
        self, study: Study, trial: FrozenTrial,
        state: TrialState, values: Sequence[float] | None,
    ) -> None:
        self._independent_cache.pop(trial.number, None)
        self._base.after_trial(study, trial, state, values)

    def reseed_rng(self) -> None:
        self._base.reseed_rng()

    @property
    def rejection_stats(self) -> dict[str, int]:
        return {
            "accepted": self._stats_accepted,
            "rejected": self._stats_rejected,
            "repaired": self._stats_repaired,
        }


# ------------------------------------------------------------------
# Parameter suggestion helper
# ------------------------------------------------------------------

def suggest_design_point(
    trial: Trial,
    trial_id: int,
    bounds: list[BoundsSpec],
    discretization_step: Optional[float] = None,
    optimization: Optional[OptimizationSpec] = None,
) -> DesignPoint:
    """Sample a DesignPoint from an Optuna Trial."""
    if discretization_step is not None or (
        optimization is not None and optimization.enable_dimension_discretization
    ):
        warnings.filterwarnings(
            "ignore",
            message=".*range is not divisible by.*",
            category=UserWarning,
            module="optuna.distributions",
        )

    params: dict[str, float] = {}
    physical_params: dict[str, float] = {}
    for b in bounds:
        ratio_low, ratio_high, ratio_step, physical_step = _ratio_sampling_spec(
            b,
            optimization=optimization,
            discretization_step=discretization_step,
        )
        if ratio_low == ratio_high:
            params[b.name] = ratio_low
            physical_params[b.name] = _ratio_to_physical_value(
                b,
                params[b.name],
                physical_step=physical_step,
            )
            continue
        if ratio_step is not None:
            params[b.name] = trial.suggest_float(
                b.name,
                ratio_low,
                ratio_high,
                step=ratio_step,
            )
        else:
            params[b.name] = trial.suggest_float(
                b.name,
                ratio_low,
                ratio_high,
            )
        physical_params[b.name] = _ratio_to_physical_value(
            b,
            params[b.name],
            physical_step=physical_step,
        )

    return DesignPoint(
        trial_id=trial_id,
        params=params,
        physical_params=physical_params,
    )
