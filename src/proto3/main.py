"""
Proto3 Main Module

FreeCAD headless constraint optimization + VEXIS + Optuna.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PenaltyConfig:
    base_penalty: float = 50.0
    alpha: float = 10.0
    failure_weights: dict[str, float] | None = None

    @classmethod
    def from_limits(cls, limits: dict) -> "PenaltyConfig":
        penalty = limits.get("penalty", {}) if limits else {}
        return cls(
            base_penalty=float(penalty.get("base_penalty", 50.0)),
            alpha=float(penalty.get("alpha", 10.0)),
            failure_weights=penalty.get("failure_weights", {}),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proto3: FreeCAD constraints optimization with VEXIS and Optuna"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/optimizer_config.yaml",
        help="Optimizer config path",
    )
    parser.add_argument(
        "--limits", "-l",
        type=str,
        default="config/proto3_limitations.yaml",
        help="Proto3 limitations config path",
    )
    parser.add_argument(
        "--max-trials", "-n",
        type=int,
        default=None,
        help="Max trials (override config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip FreeCAD/VEXIS execution for testing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing Optuna study",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--penalty-selftest",
        action="store_true",
        help="Run a local penalty test without FreeCAD/VEXIS",
    )
    return parser.parse_args()


def load_configs(config_path: Path, limits_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not limits_path.exists():
        raise FileNotFoundError(f"Limits not found: {limits_path}")

    from .utils import load_yaml

    return load_yaml(config_path), load_yaml(limits_path)


def compute_distance_from_bounds(
    params: dict[str, float],
    bounds: dict[str, dict[str, float]],
) -> float:
    distance = 0.0
    for name, value in params.items():
        if name not in bounds:
            continue
        bound = bounds[name]
        low = bound.get("min")
        high = bound.get("max")
        if low is not None and value < low:
            distance += (low - value)
        if high is not None and value > high:
            distance += (value - high)
    return distance


def penalty_value(
    params: dict[str, float],
    bounds: dict[str, dict[str, float]],
    penalty_cfg: PenaltyConfig,
    failure_reason: str,
    distance_override: Optional[float] = None,
) -> float:
    weight = 1.0
    if penalty_cfg.failure_weights:
        weight = float(penalty_cfg.failure_weights.get(failure_reason, 1.0))

    distance = (
        distance_override
        if distance_override is not None
        else compute_distance_from_bounds(params, bounds)
    )

    return weight * (penalty_cfg.base_penalty + penalty_cfg.alpha * distance)


def penalty_selftest() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    bounds = {
        "width": {"min": 10.0, "max": 20.0},
        "height": {"min": 5.0, "max": 12.0},
    }
    cfg = PenaltyConfig(base_penalty=50.0, alpha=10.0, failure_weights={"sketch_failed": 1.0})

    in_bounds = {"width": 12.0, "height": 6.0}
    out_bounds = {"width": 25.0, "height": 3.0}

    p_in = penalty_value(in_bounds, bounds, cfg, "sketch_failed")
    p_out = penalty_value(out_bounds, bounds, cfg, "sketch_failed")

    if p_out <= p_in:
        logger.error("Penalty self-test failed: p_out <= p_in (%s <= %s)", p_out, p_in)
        return 1

    logger.info("Penalty self-test passed: in=%s out=%s", p_in, p_out)
    return 0


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).parent.parent.parent

    if args.penalty_selftest:
        return penalty_selftest()

    from .utils import setup_logger, ensure_dir, save_json
    from .cad_editor import CadEditor, CadError
    from .vexis_runner import VexisRunner
    from .curve_processor import CurveProcessor
    from .result_loader import ResultLoader
    from .objective import ObjectiveCalculator, FatalOptimizationError
    from .optimizer import Optimizer, ConvergenceCallback

    log_dir = project_root / "output" / "logs"
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger("proto3", str(log_dir), level=log_level)

    start_time = datetime.now()
    logger.info("Proto3 optimization start")

    try:
        config_path = project_root / args.config
        limits_path = project_root / args.limits
        config, limits = load_configs(config_path, limits_path)
        opt_config = config.get("optimization", {})
        obj_config = config.get("objective", {})
        paths_config = config.get("paths", {})

        max_trials = args.max_trials or opt_config.get("max_trials", 100)
        convergence_threshold = opt_config.get("convergence_threshold", 0.01)
        patience = opt_config.get("patience", 20)

        # FreeCAD settings
        fc_cfg = limits.get("freecad", {})
        fcstd_path = project_root / fc_cfg.get("fcstd_path", "input/model.FCStd")
        sketch_name = fc_cfg.get("sketch_name", "Sketch001")
        constraints = fc_cfg.get("constraints", {})
        step_output_dir = project_root / fc_cfg.get("step_output_dir", "input/step")
        step_filename_template = fc_cfg.get("step_filename_template", "proto3_trial_{trial_id}.step")
        fc_timeout = fc_cfg.get("timeout_sec", 300)

        # CAE range
        cae_range = limits.get("cae", {}).get("stroke_range", {})
        stroke_min = cae_range.get("min", 0.0)
        stroke_max = cae_range.get("max", 0.5)

        # Penalty
        penalty_cfg = PenaltyConfig.from_limits(limits)

        # Paths
        result_dir = project_root / paths_config.get("result_dir", "output")
        ensure_dir(result_dir)
        storage_path = result_dir / "optuna_study_proto3.db"

        # Target curve
        target_path = project_root / paths_config.get("target_curve", "input/target_curve.csv")
        curve_processor = CurveProcessor()
        target_curve = curve_processor.process_target_curve(
            target_path, (stroke_min, stroke_max), use_polynomial=False, num_points=100
        )

        # Feature config (optional)
        result_loader = ResultLoader()
        obj_type = obj_config.get("type", "rmse")
        use_features = obj_type == "multi"
        feature_config = obj_config.get("features", {}) if use_features else {}
        target_features = result_loader.extract_features(target_curve, feature_config) if use_features else {}
        if use_features:
            feature_keys = sorted(target_features.keys())
            opt_config["directions"] = ["minimize"] * (1 + len(feature_keys))
        else:
            opt_config["directions"] = ["minimize"]

        obj_calculator = ObjectiveCalculator(target_curve, target_features, obj_config)
        vexis_runner = VexisRunner(project_root / paths_config.get("vexis_path", "vexis"))
        cad_editor = CadEditor()

        optimizer = Optimizer(
            bounds=constraints,
            config=opt_config,
            mode="all",
            storage_path=storage_path,
        )
        optimizer.create_study("proto3_shape_optimization")

        callbacks = [ConvergenceCallback(convergence_threshold, patience=patience)]

        trial_count = optimizer.get_n_trials()

        def objective_function(params: dict[str, float]):
            nonlocal trial_count
            trial_id = trial_count
            trial_count += 1

            def penalty_result(reason: str) -> float | tuple[float, ...]:
                penalty = penalty_value(params, constraints, penalty_cfg, reason)
                if use_features:
                    return tuple([penalty] * (1 + len(target_features)))
                return penalty

            if args.dry_run:
                return penalty_result("sketch_failed")

            step_name = step_filename_template.format(trial_id=trial_id)
            step_path = step_output_dir / step_name
            job_name = f"proto3_trial_{trial_id}"

            try:
                cad_editor.update_constraints_and_export_step(
                    fcstd_path=fcstd_path,
                    sketch_name=sketch_name,
                    constraints=params,
                    step_path=step_path,
                    timeout_sec=fc_timeout,
                )
            except CadError:
                return penalty_result("sketch_failed")

            try:
                vexis_runner.setup_input_step(step_path, job_name)
            except FileNotFoundError:
                return penalty_result("step_missing")

            result_csv = vexis_runner.run_analysis(job_name)
            if result_csv is None:
                return penalty_result("vexis_failed")

            try:
                result_curve = result_loader.load_curve(result_csv)
            except FileNotFoundError:
                return penalty_result("result_missing")

            result_trimmed = curve_processor.extract_range(result_curve, stroke_min, stroke_max)

            if use_features:
                result_features = result_loader.extract_features(result_trimmed, feature_config)
            else:
                result_features = {}

            try:
                objectives = obj_calculator.evaluate(result_trimmed, result_features)
            except FatalOptimizationError:
                return penalty_result("vexis_failed")

            rmse = objectives.get("rmse", float("inf"))
            if use_features:
                ret_vals = [rmse]
                for key in sorted(target_features.keys()):
                    ret_vals.append(objectives.get(f"{key}_error", float("inf")))
                return tuple(ret_vals)

            return rmse

        optimizer.run_optimization(
            objective_func=objective_function,
            base_params={},
            n_trials=max_trials,
            callbacks=callbacks,
        )

        summary = {
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_trials": optimizer.get_n_trials(),
            "best_value": optimizer.get_best_value(),
        }
        save_json(summary, result_dir / "summary_proto3.json")

        logger.info("Proto3 optimization end")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
