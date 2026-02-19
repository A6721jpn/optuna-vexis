"""
v1.0 Runner

CLI entrypoint and Optuna study lifecycle management.

Key features:
  - ``constraints_func`` injected into TPE/NSGA-II so the sampler
    learns the CAD feasibility boundary from trial history.
  - ``FeasibilityAwareSampler`` wrapper performs rejection sampling
    at the sampler level when an ML gate model is available.
  - Feasibility violation score stored per-trial in user_attrs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import logging
import sys
from datetime import datetime
from pathlib import Path
import json

import optuna
from optuna.study import Study
from optuna.trial import Trial

from .cad_gate import CadGate
from .cae_evaluator import CaeEvaluator, load_curve, extract_range, extract_features
from .config import load_config
from .geometry_adapter import GeometryAdapter
from .objective import ObjectiveOrchestrator
from .persistence import TrialPersistence
from .reporting import generate_markdown_report
from .search_space import (
    FeasibilityAwareSampler,
    build_fixed_search_space,
    create_sampler,
    make_constraints_func,
    normalize_bounds_to_sampling_grid,
    suggest_design_point,
)
from .types import DesignPoint
from .versioning import get_version_info

logger = logging.getLogger(__name__)


def _objective_metric_key(cfg, name: str) -> str:
    if cfg.objective.multi_objectives_use_error or name in cfg.objective.target_values:
        return f"{name}_error"
    return name


def _adjust_multi_directions_for_targets(
    cfg,
    directions: list[str],
    objective_names: list[str],
) -> list[str]:
    adjusted = list(directions)
    idx = 0
    if cfg.objective.include_rmse_in_multi:
        idx += 1
    for name in objective_names:
        if idx >= len(adjusted):
            break
        metric_key = _objective_metric_key(cfg, name)
        if metric_key.endswith("_error") and adjusted[idx] == "maximize":
            logger.warning(
                "Direction for %s changed maximize->minimize "
                "because target optimization uses error distance",
                metric_key,
            )
            adjusted[idx] = "minimize"
        idx += 1
    return adjusted


def _convert_physical_bounds_to_ratio(cfg) -> None:
    """Convert physical-domain bounds into ratio-domain bounds in-place.

    When ``freecad.constraints_domain`` is ``physical``, values in
    ``freecad.constraints[*].min/max`` are interpreted as real dimensions.
    Optuna / CAD gate / hard-constraint checks still operate in ratio domain,
    so we convert using each bound's resolved ``base_value``.
    """
    if getattr(cfg.freecad, "constraints_domain", "ratio") != "physical":
        return

    converted = 0
    skipped: list[str] = []
    for b in cfg.bounds:
        try:
            base = float(b.base_value)
            raw_min = float(b.min)
            raw_max = float(b.max)
        except (TypeError, ValueError):
            skipped.append(b.name)
            continue
        if not math.isfinite(base) or base == 0.0:
            skipped.append(b.name)
            continue

        ratio_min = raw_min / base
        ratio_max = raw_max / base
        if ratio_min <= ratio_max:
            b.min = ratio_min
            b.max = ratio_max
        else:
            b.min = ratio_max
            b.max = ratio_min
        converted += 1

    logger.info(
        "Converted physical constraints to ratio domain: %d/%d dimensions",
        converted,
        len(cfg.bounds),
    )
    if skipped:
        logger.warning(
            "Skipped ratio conversion for %d dimensions due to invalid base/min/max: %s",
            len(skipped),
            ", ".join(sorted(skipped)),
        )


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
        # For multi-objective studies we currently skip early-stop criteria.
        if len(study.directions) > 1:
            return

        if trial.values:
            val = trial.values[0]
        else:
            try:
                val = trial.value
            except RuntimeError:
                val = None
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
    log_file = log_dir / f"v1_0_{timestamp}.log"

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
        description="Production v1.0: CAD feasibility-gated optimization with VEXIS"
    )
    parser.add_argument(
        "--config", "-c", default="config/optimizer_config.yaml",
        help="Optimizer config YAML",
    )
    parser.add_argument(
        "--limits", "-l", default="config/v1_0_limitations.yaml",
        help="v1.0 limitations YAML",
    )
    parser.add_argument("--max-trials", "-n", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--version", action="store_true", help="Print version info and exit")
    return parser.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    version_info = get_version_info(project_root)

    if args.version:
        print(json.dumps(version_info, ensure_ascii=False, indent=2))
        return 0

    start_time = datetime.now()
    try:
        cfg = load_config(
            project_root / args.config,
            project_root / args.limits,
        )
    except Exception as exc:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        logger.exception("Failed to load config: %s", exc)
        return 1

    log_dir = project_root / cfg.logging.output_dir
    log_level = "DEBUG" if args.verbose else cfg.logging.level
    _setup_logging(log_dir, level=log_level)

    logger.info("v1.0 optimization start")

    try:
        max_trials = args.max_trials or cfg.optimization.max_trials

        # Directions
        if cfg.optimization.objective_type == "multi":
            objective_names = cfg.objective.multi_objectives or list(cfg.objective.features.keys())
            n_obj = len(objective_names) + (1 if cfg.objective.include_rmse_in_multi else 0)
            n_obj = max(1, n_obj)
            if len(cfg.optimization.directions) == n_obj:
                directions = list(cfg.optimization.directions)
            elif len(cfg.optimization.directions) == 1:
                directions = list(cfg.optimization.directions) * n_obj
            else:
                logger.warning(
                    "optimization.directions length (%d) does not match objective count (%d); "
                    "fallback to all minimize",
                    len(cfg.optimization.directions),
                    n_obj,
                )
                directions = ["minimize"] * n_obj
            directions = _adjust_multi_directions_for_targets(
                cfg,
                directions,
                objective_names,
            )
        else:
            directions = [cfg.optimization.directions[0]]
        cfg.optimization.directions = list(directions)

        # Paths
        result_dir = project_root / cfg.paths.result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        storage_path = result_dir / "optuna_study_v1_0.db"
        storage_url = f"sqlite:///{storage_path}"

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
        for name, target in cfg.objective.target_values.items():
            target_features[name] = float(target)

        # --- Components ---
        persistence = TrialPersistence(result_dir)
        cad_gate_spec = cfg.cad_gate
        if cad_gate_spec.model_path:
            raw_gate_path = Path(cad_gate_spec.model_path)
            resolved_gate_path = raw_gate_path if raw_gate_path.is_absolute() else (project_root / raw_gate_path)
            cad_gate_spec = replace(cad_gate_spec, model_path=str(resolved_gate_path.resolve()))
        cad_gate = CadGate(cad_gate_spec)
        if cad_gate_spec.enabled and cad_gate_spec.model_path and cad_gate._model is None:
            logger.error(
                "CAD gate is enabled but model failed to load: %s (reason=%s). "
                "Install required dependencies in this interpreter (e.g. joblib, scikit-learn).",
                cad_gate_spec.model_path,
                cad_gate._load_error or "unknown",
            )
            return 1
        geometry_adapter = GeometryAdapter(
            cfg.freecad,
            project_root,
            cfg.optimization,
        )
        try:
            base_values = geometry_adapter.probe_base_values([b.name for b in cfg.bounds])
            updated = 0
            for b in cfg.bounds:
                if b.name not in base_values:
                    continue
                base = float(base_values[b.name])
                if not math.isfinite(base) or base == 0.0:
                    continue
                b.base_value = base
                updated += 1
            logger.info(
                "Resolved physical base values from FCStd: %d/%d dimensions",
                updated,
                len(cfg.bounds),
            )
        except Exception as exc:
            logger.warning(
                "Failed to resolve physical base values from FCStd; "
                "falling back to configured base_value (reason=%s)",
                exc,
            )
        _convert_physical_bounds_to_ratio(cfg)
        normalized = normalize_bounds_to_sampling_grid(
            cfg.bounds,
            optimization=cfg.optimization,
            discretization_step=cfg.optimization.discretization_step,
        )
        logger.info(
            "Normalized bounds to sampling grid: %d dimensions",
            normalized,
        )

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
            "version_info": version_info,
        })

        # --- Sampler with constraints_func ---
        # A) constraints_func: TPE/NSGA-II learn the feasible boundary
        #    from trial history (soft guidance).
        constraints_func = make_constraints_func()

        base_sampler = create_sampler(
            cfg.optimization,
            storage=storage_url,
            constraints_func=constraints_func,
            n_objectives=len(directions),
        )

        # B) FeasibilityAwareSampler: rejection sampling at the sampler
        #    level to avoid wasting trial budget on infeasible points.
        #    Only wraps when ML gate model is loaded.
        if cad_gate._model is not None:
            def _predict_fn(params: dict[str, float]) -> bool:
                dummy = DesignPoint(trial_id=-1, params=params)
                return cad_gate.predict(dummy).is_feasible

            def _predict_score_fn(params: dict[str, float]) -> float:
                dummy = DesignPoint(trial_id=-1, params=params)
                feas = cad_gate.predict(dummy)
                if feas.confidence is None:
                    return 1.0 if feas.is_feasible else 0.0
                return float(feas.confidence)

            bounds_map = {b.name: b for b in cfg.bounds}
            baseline_params = {
                b.name: (1.0 if b.min <= 1.0 <= b.max else (b.min + b.max) / 2.0)
                for b in cfg.bounds
            }

            def _clip_params(params: dict[str, float]) -> dict[str, float]:
                clipped: dict[str, float] = {}
                for name, b in bounds_map.items():
                    raw = float(params.get(name, baseline_params[name]))
                    clipped[name] = min(max(raw, b.min), b.max)
                return clipped

            baseline_feas = cad_gate.predict(DesignPoint(trial_id=-1, params=baseline_params))
            logger.info(
                "CAD gate baseline check: feasible=%s, conf=%s",
                baseline_feas.is_feasible,
                f"{baseline_feas.confidence:.4f}" if baseline_feas.confidence is not None else "n/a",
            )

            def _repair_fn(params: dict[str, float]) -> dict[str, float] | None:
                candidate = _clip_params(params)
                if _predict_fn(candidate):
                    return candidate

                # If baseline is not feasible, deterministic repair is unsafe.
                if not baseline_feas.is_feasible:
                    return None

                # Move from feasible baseline toward candidate and keep the
                # farthest feasible interpolation point.
                best = dict(baseline_params)
                lo = 0.0
                hi = 1.0
                for _ in range(14):
                    mid = (lo + hi) / 2.0
                    mixed = {
                        name: baseline_params[name] + mid * (candidate[name] - baseline_params[name])
                        for name in baseline_params
                    }
                    if _predict_fn(mixed):
                        best = mixed
                        lo = mid
                    else:
                        hi = mid
                return best

            fixed_search_space = build_fixed_search_space(
                cfg.bounds,
                optimization=cfg.optimization,
                discretization_step=cfg.optimization.discretization_step,
            )

            sampler = FeasibilityAwareSampler(
                base_sampler=base_sampler,
                predict_fn=_predict_fn,
                predict_score_fn=_predict_score_fn,
                expected_param_names=list(bounds_map.keys()),
                fixed_search_space=fixed_search_space,
                repair_fn=_repair_fn,
                max_retries=cfg.cad_gate.rejection_max_retries,
            )
            logger.info(
                "FeasibilityAwareSampler enabled (max_retries=%d, fixed_dims=%d)",
                cfg.cad_gate.rejection_max_retries,
                len(fixed_search_space),
            )
        else:
            sampler = base_sampler
            logger.info(
                "No ML gate model loaded; using base sampler only "
                "(constraints_func still active for boundary learning)"
            )

        # Optuna study
        study = optuna.create_study(
            study_name="v1_0_optimization",
            sampler=sampler,
            directions=directions,
            storage=storage_url,
            load_if_exists=True,
        )

        completed = len(study.trials)
        remaining = max_trials - completed
        if remaining <= 0:
            logger.info("Already completed %d trials; skip optimize and write report", completed)
        else:
            trial_counter = completed

            def wrapped_objective(trial: Trial) -> float | tuple[float, ...]:
                nonlocal trial_counter
                tid = trial_counter
                trial_counter += 1
                point = suggest_design_point(
                    trial,
                    tid,
                    cfg.bounds,
                    cfg.optimization.discretization_step,
                    cfg.optimization,
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
                "Rejection stats: accepted=%d, rejected=%d, repaired=%d",
                rejection_stats["accepted"],
                rejection_stats["rejected"],
                rejection_stats.get("repaired", 0),
            )

        end_time = datetime.now()
        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_trials": len(study.trials),
            "best_value": best_value,
            "best_params": best_params,
            "rejection_stats": rejection_stats,
        }
        persistence.save_summary(summary)

        try:
            report_path = generate_markdown_report(
                result_dir=result_dir,
                study=study,
                cfg=cfg,
                optimizer_config_path=args.config,
                limits_config_path=args.limits,
                start_time=start_time,
                end_time=end_time,
                actual_sampler_name=type(sampler).__name__,
                rejection_stats=rejection_stats,
                version_info=version_info,
            )
            logger.info("Markdown report saved: %s", report_path)
        except Exception as exc:
            logger.warning("Failed to generate markdown report: %s", exc)

        logger.info("v1.0 optimization end — best_value=%s", best_value)
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
