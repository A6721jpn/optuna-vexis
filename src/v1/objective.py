"""
v1.0 Objective Orchestrator

Unified objective function that implements the full decision policy:
  1. Check hard constraints
  2. CAD feasibility gate
  3. Geometry generation
  4. CAE evaluation
  5. Metric -> objective value(s)

Returns penalty for infeasible / failed trials.

Stores feasibility violation score in ``trial.user_attrs`` so that the
sampler's ``constraints_func`` can learn the feasible boundary.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import optuna

from .cad_gate import CadGate
from .cae_evaluator import CaeEvaluator
from .config import V1Config
from .constraints import check_hard_constraints, penalty_value
from .geometry_adapter import GeometryAdapter, GeometryError
from .persistence import TrialPersistence
from .search_space import FEASIBILITY_ATTR, LEGACY_FEASIBILITY_ATTR
from .types import (
    CaeResult,
    CaeStatus,
    DesignPoint,
    TrialOutcome,
    TrialRecord,
)

logger = logging.getLogger(__name__)

FAILURE_STAGE_ATTR = "v1_0_failure_stage"
FAILURE_REASON_ATTR = "v1_0_failure_reason"
LEGACY_FAILURE_STAGE_ATTR = "proto4_failure_stage"
LEGACY_FAILURE_REASON_ATTR = "proto4_failure_reason"


class ObjectiveOrchestrator:
    """Wire together all v1.0 components into one callable objective.

    The orchestrator accepts an optional ``optuna.trial.Trial`` so it can
    write the feasibility violation score into ``trial.user_attrs``.
    This allows the ``constraints_func`` passed to TPE / NSGA-II to read
    the score and bias future sampling toward feasible regions.
    """

    def __init__(
        self,
        cfg: V1Config,
        cad_gate: CadGate,
        geometry_adapter: GeometryAdapter,
        cae_evaluator: CaeEvaluator,
        persistence: TrialPersistence,
    ) -> None:
        self._cfg = cfg
        self._gate = cad_gate
        self._geom = geometry_adapter
        self._cae = cae_evaluator
        self._persist = persistence
        self._use_features = cfg.optimization.objective_type == "multi"

    def _selected_multi_objective_names(self) -> list[str]:
        if self._cfg.objective.multi_objectives:
            return list(self._cfg.objective.multi_objectives)
        return list(self._cfg.objective.features.keys())

    def _metric_key_for_objective(self, name: str) -> str:
        if self._cfg.objective.multi_objectives_use_error:
            return f"{name}_error"
        return name

    def _resolve_directions(self, n_obj: int) -> list[str]:
        if n_obj <= 0:
            return ["minimize"]
        dirs = [d.lower() for d in self._cfg.optimization.directions]
        if len(dirs) == n_obj:
            return dirs
        if len(dirs) == 1:
            return dirs * n_obj
        return ["minimize"] * n_obj

    def __call__(
        self,
        point: DesignPoint,
        *,
        trial: Optional[optuna.trial.Trial] = None,
        dry_run: bool = False,
    ) -> float | tuple[float, ...]:
        """Evaluate a single DesignPoint.

        Args:
            point: Design variables to evaluate.
            trial: Live Optuna Trial (used to store feasibility attrs).
            dry_run: Skip FreeCAD / VEXIS execution.

        Returns:
            Scalar objective or tuple (multi-objective).
        """
        t0 = time.time()
        record = TrialRecord(trial_id=point.trial_id, design_point=point)

        try:
            result = self._evaluate(point, record, trial=trial, dry_run=dry_run)
        except Exception as exc:
            logger.exception("Unexpected error in trial %d: %s", point.trial_id, exc)
            record.outcome = TrialOutcome.CAE_FAIL
            self._store_feasibility(trial, 1.0)  # unknown → treat as infeasible
            self._store_failure_context(
                trial,
                stage="objective_exception",
                reason=f"{exc.__class__.__name__}: {exc}",
            )
            result = self._penalty(point, TrialOutcome.CAE_FAIL)
        finally:
            record.wall_clock_sec = time.time() - t0
            self._persist.save_trial(record)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _store_feasibility(
        trial: Optional[optuna.trial.Trial],
        violation: float,
    ) -> None:
        """Write feasibility violation score to the Optuna trial.

        * violation <= 0  → feasible
        * violation > 0   → infeasible (higher = worse)
        """
        if trial is not None:
            trial.set_user_attr(FEASIBILITY_ATTR, violation)
            trial.set_user_attr(LEGACY_FEASIBILITY_ATTR, violation)

    @staticmethod
    def _store_failure_context(
        trial: Optional[optuna.trial.Trial],
        *,
        stage: str,
        reason: Optional[str] = None,
    ) -> None:
        if trial is None:
            return
        trial.set_user_attr(FAILURE_STAGE_ATTR, stage)
        trial.set_user_attr(LEGACY_FAILURE_STAGE_ATTR, stage)
        if reason:
            trial.set_user_attr(FAILURE_REASON_ATTR, reason)
            trial.set_user_attr(LEGACY_FAILURE_REASON_ATTR, reason)

    @staticmethod
    def _cae_failure_violation(cae_result: CaeResult) -> float:
        """Map CAE failure reason to feasibility violation severity."""
        reason = (cae_result.failure_reason or "").lower()
        if not reason:
            return 0.8
        if any(token in reason for token in (
            "timeout",
            "process_exit_",
            "fatal",
            "execution_error",
            "subprocess_failed",
            "stall",
            "solver_error_marker",
        )):
            return 1.0
        if any(token in reason for token in ("result_load_failed", "result_csv_missing")):
            return 0.9
        return 0.8

    def _evaluate(
        self,
        point: DesignPoint,
        record: TrialRecord,
        *,
        trial: Optional[optuna.trial.Trial],
        dry_run: bool,
    ) -> float | tuple[float, ...]:
        # 1. Hard constraint check
        violation = check_hard_constraints(point, self._cfg.bounds)
        if violation:
            logger.info("Trial %d: constraint violation — %s", point.trial_id, violation)
            record.outcome = TrialOutcome.CONSTRAINT_VIOLATION
            self._store_feasibility(trial, 1.0)
            self._store_failure_context(trial, stage="hard_constraint", reason=violation)
            return self._penalty(point, TrialOutcome.CONSTRAINT_VIOLATION)

        # 2. CAD feasibility gate
        feas = self._gate.predict(point)
        record.feasibility = feas
        if not feas.is_feasible:
            # Store violation magnitude (1 - confidence so higher = worse)
            score = 1.0 - (feas.confidence or 0.0)
            logger.info(
                "Trial %d: CAD infeasible (conf=%.3f, reason=%s)",
                point.trial_id, feas.confidence or 0, feas.reason_code,
            )
            record.outcome = TrialOutcome.CAD_INFEASIBLE
            self._store_feasibility(trial, score)
            self._store_failure_context(
                trial,
                stage="cad_gate",
                reason=feas.reason_code,
            )
            return self._penalty(point, TrialOutcome.CAD_INFEASIBLE)

        if dry_run:
            record.outcome = TrialOutcome.CAE_SUCCESS
            self._store_feasibility(trial, -1.0)
            if self._use_features:
                values: list[float] = []
                dry_metrics: dict[str, float] = {}
                if self._cfg.objective.include_rmse_in_multi:
                    values.append(0.0)
                    dry_metrics["rmse"] = 0.0
                for name in self._selected_multi_objective_names():
                    metric_key = self._metric_key_for_objective(name)
                    default = 1.0 if metric_key.endswith("_error") else 0.0
                    dry_metrics[metric_key] = default
                    values.append(default)
                if not values:
                    values = [0.0]
                record.objective_values = dry_metrics or {"rmse": 0.0}
                return tuple(values)

            record.objective_values = {"rmse": 0.0}
            return 0.0

        # 3. Generate STEP geometry
        try:
            step_path = self._geom.generate_step(point)
        except GeometryError as exc:
            logger.warning("Trial %d: geometry error — %s", point.trial_id, exc)
            record.outcome = TrialOutcome.CAD_INFEASIBLE
            self._store_feasibility(trial, 1.0)
            self._store_failure_context(
                trial,
                stage="geometry_generation",
                reason=str(exc),
            )
            return self._penalty(point, TrialOutcome.CAD_INFEASIBLE)

        # 4. CAE evaluation — point was CAD-feasible
        cae_result = self._cae.evaluate(step_path, point)
        record.cae_result = cae_result

        if cae_result.status != CaeStatus.SUCCESS:
            logger.warning(
                "Trial %d: CAE failed (%s)",
                point.trial_id,
                cae_result.failure_reason or "unknown",
            )
            record.outcome = TrialOutcome.CAE_FAIL
            self._store_feasibility(trial, self._cae_failure_violation(cae_result))
            self._store_failure_context(
                trial,
                stage="cae_evaluation",
                reason=cae_result.failure_reason,
            )
            return self._penalty(point, TrialOutcome.CAE_FAIL)

        # 5. Success
        self._store_feasibility(trial, -1.0)  # feasible (<= 0)
        record.outcome = TrialOutcome.CAE_SUCCESS
        record.objective_values = cae_result.metrics

        rmse = cae_result.metrics.get("rmse", float("inf"))
        logger.info("Trial %d: RMSE=%.6f", point.trial_id, rmse)

        if self._use_features:
            values: list[float] = []
            if self._cfg.objective.include_rmse_in_multi:
                values.append(rmse)
            for name in self._selected_multi_objective_names():
                metric_key = self._metric_key_for_objective(name)
                default = 1.0 if metric_key.endswith("_error") else 0.0
                values.append(float(cae_result.metrics.get(metric_key, default)))
            return tuple(values)

        return rmse

    def _penalty(self, point: DesignPoint, outcome: TrialOutcome) -> float | tuple[float, ...]:
        p = penalty_value(point, self._cfg.bounds, self._cfg.penalty, outcome)
        if self._use_features:
            n_obj = len(self._selected_multi_objective_names())
            if self._cfg.objective.include_rmse_in_multi:
                n_obj += 1
            n_obj = max(1, n_obj)
            dirs = self._resolve_directions(n_obj)
            return tuple(p if d == "minimize" else -p for d in dirs)

        direction = self._resolve_directions(1)[0]
        return p if direction == "minimize" else -p
