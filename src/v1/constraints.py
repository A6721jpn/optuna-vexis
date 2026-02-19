"""
v1.0 Constraints

Hard / soft constraint checking and penalty computation.
Replaces proto3 PenaltyConfig with a more structured approach.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .config import BoundsSpec, OptimizationSpec, PenaltySpec
from .search_space import sampling_spec_for_bound
from .types import DesignPoint, TrialOutcome

logger = logging.getLogger(__name__)


def distance_from_bounds(point: DesignPoint, bounds: list[BoundsSpec]) -> float:
    """Sum of per-variable distances outside the feasible bounds."""
    bounds_map = {b.name: b for b in bounds}
    dist = 0.0
    for name, value in point.params.items():
        b = bounds_map.get(name)
        if b is None:
            continue
        if value < b.min:
            dist += b.min - value
        elif value > b.max:
            dist += value - b.max
    return dist


def penalty_value(
    point: DesignPoint,
    bounds: list[BoundsSpec],
    penalty_spec: PenaltySpec,
    outcome: TrialOutcome,
    distance_override: Optional[float] = None,
) -> float:
    """Compute a scalar penalty for a failed trial.

    penalty = weight * (base_penalty + alpha * distance_from_bounds)
    """
    weight = penalty_spec.failure_weights.get(outcome.value, 1.0)
    dist = (
        distance_override
        if distance_override is not None
        else distance_from_bounds(point, bounds)
    )
    return weight * (penalty_spec.base_penalty + penalty_spec.alpha * dist)


def check_hard_constraints(
    point: DesignPoint,
    bounds: list[BoundsSpec],
    *,
    optimization: Optional[OptimizationSpec] = None,
) -> Optional[str]:
    """Return a reason string if any hard constraint is violated, else None."""
    bounds_map = {b.name: b for b in bounds}
    sampling_bounds: dict[str, tuple[float, float, Optional[float]]] = {}
    if optimization is not None:
        for b in bounds:
            low, high, step, _ = sampling_spec_for_bound(
                b,
                optimization=optimization,
                discretization_step=optimization.discretization_step,
            )
            sampling_bounds[b.name] = (float(low), float(high), step)

    for name, value in point.params.items():
        b = bounds_map.get(name)
        if b is None:
            continue
        low = float(b.min)
        high = float(b.max)
        step: Optional[float] = None
        if name in sampling_bounds:
            low, high, step = sampling_bounds[name]

        v = float(value)
        if step is not None and step > 0.0:
            tol = max(abs(step) * 1e-9, 1e-12)
            if v < low - tol or v > high + tol:
                return f"{name}={v:.6f} outside [{low}, {high}]"

            # Integer-lattice check: convert to discrete ticks and validate.
            span = high - low
            n_steps = int(round(span / step)) if span > 0.0 else 0
            tick = int(round((v - low) / step))
            snapped = low + (tick * step)
            scale = max(1.0, abs(v), abs(snapped))
            if tick < 0 or tick > n_steps:
                return f"{name}={v:.6f} outside [{low}, {high}]"
            if not math.isclose(v, snapped, rel_tol=0.0, abs_tol=tol * scale):
                return f"{name}={v:.6f} off lattice [low={low}, high={high}, step={step}]"
            continue

        tol = 1e-12
        if v < low - tol or v > high + tol:
            return f"{name}={v:.6f} outside [{low}, {high}]"
    return None
