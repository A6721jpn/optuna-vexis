"""
Proto4 Runner

CLI entrypoint and Optuna study lifecycle management.

Key features over proto3:
  - ``constraints_func`` injected into TPE/NSGA-II so the sampler
    learns the CAD feasibility boundary from trial history.
  - ``FeasibilityAwareSampler`` wrapper performs rejection sampling
    at the sampler level when an ML gate model is available.
  - Feasibility violation score stored per-trial in user_attrs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import optuna
from optuna.study import Study
from optuna.trial import Trial

from .cad_gate import CadGate
from .cae_evaluator import CaeEvaluator, load_curve, extract_range, extract_features
from .config import Proto4Config, load_config
from .geometry_adapter import GeometryAdapter
from .objective import ObjectiveOrchestrator
from .persistence import TrialPersistence
from .search_space import (
    FeasibilityAwareSampler,
    create_sampler,
    make_constraints_func,
    suggest_design_point,
)
from .types import DesignPoint

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Convergence callback
# ------------------------------------------------------------------

class ConvergenceCallback:
    def __init__(self, threshold: float, patience: int = 20) -> None:
        self.threshold = threshold
        self.patience = patience
        self._best = float("inf")
        self._no_improve = 0

    def __call__(self, study: Study, trial: optuna.trial.FrozenTrial) -> None:
        val = trial.value if trial.value is not None else (
            trial.values[0] if trial.values else None
        )
        if val is None:
            return
        if val <= self.threshold:
            logger.info("Converged: %.6f <= %.6f", val, self.threshold)
            study.stop()
            return
        if val < self._best:
            self._best = val
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self.patience:
            logger.info("No improvement for %d trials; stopping", self.patience)
            study.stop()


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def _setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"proto4_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root.addHandler(ch)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proto4: CAD feasibility-gated optimization with VEXIS"
    )
    parser.add_argument(
        "--config", "-c", default="config/optimizer_config.yaml",
        help="Optimizer config YAML",
    )
    parser.add_argument(
        "--limits", "-l", default="config/proto4_limitations.yaml",
        help="Proto4 limitations YAML",
    )
    parser.add_argument("--max-trials", "-n", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent

    log_dir = project_root / "output" / "logs"
    _setup_logging(log_dir, level="DEBUG" if args.verbose else "INFO")

    start_time = datetime.now()
    logger.info("Proto4 optimization start")

    try:
        cfg = load_config(
            project_root / args.config,
            project_root / args.limits,
        )

        max_trials = args.max_trials or cfg.optimization.max_trials

        # Directions
        if cfg.optimization.objective_type == "multi":
            n_obj = 1 + len(cfg.objective.features)
            directions = ["minimize"] * n_obj
        else:
            directions = ["minimize"]

        # Paths
        result_dir = project_root / cfg.paths.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        storage_path = result_dir / "optuna_study_proto4.db"
        storage_url = f"sqlite:///{storage_path}"
        
        logger.info(f"Using config: {args.config}")
        logger.info(f"Using limitations: {args.limits}")
        logger.info(f"Storage URL: {storage_url}")

        # Target curve
        target_path = project_root / cfg.paths.target_curve
        target_curve_raw = load_curve(target_path)
        target_curve = extract_range(
            target_curve_raw,
            cfg.cae.stroke_range_min,
            cfg.cae.stroke_range_max,
        )

        # Target features
        feat_cfg = (
            cfg.objective.features
            if cfg.optimization.objective_type == "multi"
            else {}
        )
        target_features = extract_features(target_curve, feat_cfg) if feat_cfg else {}

        # --- Components ---
        persistence = TrialPersistence(result_dir)
        cad_gate = CadGate(cfg.cad_gate)
        geometry_adapter = GeometryAdapter(cfg.freecad, project_root)
        cae_evaluator = CaeEvaluator(
            vexis_path=project_root / cfg.paths.vexis_path,
            cae_spec=cfg.cae,
            obj_spec=cfg.objective,
            target_curve=target_curve,
            target_features=target_features,
        )
        orchestrator = ObjectiveOrchestrator(
            cfg=cfg,
            cad_gate=cad_gate,
            geometry_adapter=geometry_adapter,
            cae_evaluator=cae_evaluator,
            persistence=persistence,
        )

        # Snapshot config
        persistence.save_run_config({
            "optimizer_config": args.config,
            "limits": args.limits,
            "max_trials": max_trials,
            "seed": cfg.optimization.seed,
            "sampler": cfg.optimization.sampler,
            "cad_gate_enabled": cfg.cad_gate.enabled,
            "rejection_max_retries": cfg.cad_gate.rejection_max_retries,
            "start_time": start_time.isoformat(),
        })

        # --- Sampler with constraints_func ---
        # A) constraints_func: TPE/NSGA-II learn the feasible boundary
        #    from trial history (soft guidance).
        constraints_func = make_constraints_func()

        base_sampler = create_sampler(
            cfg.optimization,
            storage=storage_url,
            constraints_func=constraints_func,
        )

        # B) FeasibilityAwareSampler: rejection sampling at the sampler
        #    level to avoid wasting trial budget on infeasible points.
        #    Only wraps when ML gate model is loaded.
        if cad_gate._model is not None:
            def _predict_fn(params: dict[str, float]) -> bool:
                dummy = DesignPoint(trial_id=-1, params=params)
                return cad_gate.predict(dummy).is_feasible

            sampler = FeasibilityAwareSampler(
                base_sampler=base_sampler,
                predict_fn=_predict_fn,
                max_retries=cfg.cad_gate.rejection_max_retries,
            )
            logger.info(
                "FeasibilityAwareSampler enabled (max_retries=%d)",
                cfg.cad_gate.rejection_max_retries,
            )
        else:
            sampler = base_sampler
            logger.info(
                "No ML gate model loaded; using base sampler only "
                "(constraints_func still active for boundary learning)"
            )

        # Optuna study
        study = optuna.create_study(
            study_name="proto4_optimization",
            sampler=sampler,
            directions=directions,
            storage=storage_url,
            load_if_exists=True,
        )

        completed = len(study.trials)
        remaining = max_trials - completed
        if remaining <= 0:
            logger.info("Already completed %d trials; nothing to do", completed)
            return 0

        trial_counter = completed

        def wrapped_objective(trial: Trial) -> float | tuple[float, ...]:
            nonlocal trial_counter
            tid = trial_counter
            trial_counter += 1
            point = suggest_design_point(
                trial, tid, cfg.bounds, cfg.optimization.discretization_step,
            )
            # Pass the live Trial to the orchestrator so it can
            # write feasibility attrs for constraints_func.
            return orchestrator(point, trial=trial, dry_run=args.dry_run)

        callbacks = [
            ConvergenceCallback(
                cfg.optimization.convergence_threshold,
                patience=cfg.optimization.patience,
            ),
        ]

        logger.info("Optimization: total=%d, remaining=%d", max_trials, remaining)
        study.optimize(
            wrapped_objective,
            n_trials=remaining,
            callbacks=callbacks,
            show_progress_bar=False,
        )

        # --- Summary ---
        best_value = None
        best_params = None
        try:
            best_value = study.best_value
            best_params = dict(study.best_params)
        except (ValueError, RuntimeError):
            pass

        # Rejection stats
        rejection_stats = {}
        if isinstance(sampler, FeasibilityAwareSampler):
            rejection_stats = sampler.rejection_stats
            logger.info(
                "Rejection stats: accepted=%d, rejected=%d",
                rejection_stats["accepted"],
                rejection_stats["rejected"],
            )

        summary = {
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_trials": len(study.trials),
            "best_value": best_value,
            "best_params": best_params,
            "rejection_stats": rejection_stats,
        }
        persistence.save_summary(summary)

        logger.info("Proto4 optimization end — best_value=%s", best_value)
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
