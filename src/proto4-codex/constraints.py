"""
Proto4 Constraints

Hard / soft constraint checking and penalty computation.
Replaces proto3 PenaltyConfig with a more structured approach.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import BoundsSpec, PenaltySpec
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
) -> Optional[str]:
    """Return a reason string if any hard constraint is violated, else None."""
    bounds_map = {b.name: b for b in bounds}
    for name, value in point.params.items():
        b = bounds_map.get(name)
        if b is None:
            continue
        if value < b.min or value > b.max:
            return f"{name}={value:.6f} outside [{b.min}, {b.max}]"
    return None
