"""
v1.0 Shared Types

Design-level DTOs used across all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class CaeStatus(Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class TrialOutcome(Enum):
    CAD_INFEASIBLE = "cad_infeasible"
    CAE_FAIL = "cae_fail"
    CAE_SUCCESS = "cae_success"
    CONSTRAINT_VIOLATION = "constraint_violation"


@dataclass
class DesignPoint:
    """Sampled parameter set for a single trial."""

    trial_id: int
    params: dict[str, float]
    physical_params: Optional[dict[str, float]] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"trial_id": self.trial_id, "params": dict(self.params)}
        if self.physical_params is not None:
            payload["physical_params"] = dict(self.physical_params)
        return payload


@dataclass
class CadFeasibilityResult:
    """Output of the CAD feasibility gate."""

    is_feasible: bool
    confidence: Optional[float] = None
    reason_code: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_feasible": self.is_feasible,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "metadata": self.metadata,
        }


@dataclass
class CaeResult:
    """Output of a single CAE evaluation."""

    status: CaeStatus
    metrics: dict[str, float] = field(default_factory=dict)
    runtime_sec: float = 0.0
    artifact_paths: list[str] = field(default_factory=list)
    failure_reason: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "runtime_sec": self.runtime_sec,
            "artifact_paths": list(self.artifact_paths),
        }
        if self.failure_reason:
            payload["failure_reason"] = self.failure_reason
        if self.started_at:
            payload["started_at"] = self.started_at
        if self.finished_at:
            payload["finished_at"] = self.finished_at
        return payload


@dataclass
class TrialRecord:
    """Per-trial traceability record."""

    trial_id: int
    design_point: DesignPoint
    feasibility: Optional[CadFeasibilityResult] = None
    cae_result: Optional[CaeResult] = None
    objective_values: Optional[dict[str, float]] = None
    outcome: Optional[TrialOutcome] = None
    wall_clock_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "trial_id": self.trial_id,
            "design_point": self.design_point.to_dict(),
            "outcome": self.outcome.value if self.outcome else None,
            "wall_clock_sec": self.wall_clock_sec,
        }
        if self.feasibility:
            d["feasibility"] = self.feasibility.to_dict()
        if self.cae_result:
            d["cae_result"] = self.cae_result.to_dict()
        if self.objective_values:
            d["objective_values"] = self.objective_values
        return d
