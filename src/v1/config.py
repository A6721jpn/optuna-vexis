"""
v1.0 Configuration

Load, validate, and expose typed configuration from YAML files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BoundsSpec:
    """Single parameter bounds."""

    name: str
    min: float
    max: float
    base_value: float = 1.0


@dataclass
class PenaltySpec:
    base_penalty: float = 50.0
    alpha: float = 10.0
    failure_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class CadGateSpec:
    model_path: Optional[str] = None
    threshold: float = 0.5
    enabled: bool = True
    rejection_max_retries: int = 50


@dataclass
class CaeSpec:
    stroke_range_min: float = 0.0
    stroke_range_max: float = 0.5
    solver_progress_stall_sec: int = 300
    solver_log_poll_sec: float = 1.0
    solver_log_activity_resets_progress_stall: bool = True
    solver_hard_timeout_sec: Optional[int] = 3600
    solver_error_markers: list[str] = field(default_factory=lambda: [
        "error termination",
        "fatal error",
    ])
    max_retries: int = 1
    stream_stdout: bool = False
    stdout_log_dir: Optional[str] = None
    stdout_console_level: str = "INFO"


@dataclass
class OptimizationSpec:
    sampler: str = "AUTO"
    max_trials: int = 300
    convergence_threshold: float = 0.001
    patience: int = 100
    seed: int = 42
    n_startup_trials: int = 70
    discretization_step: Optional[float] = None
    enable_dimension_discretization: bool = True
    non_angle_step: float = 0.01
    angle_step: float = 0.001
    angle_name_token: str = "ANGLE"
    objective_type: str = "single"
    directions: list[str] = field(default_factory=lambda: ["minimize"])


@dataclass
class ObjectiveSpec:
    type: str = "rmse"
    weights: dict[str, float] = field(default_factory=lambda: {"rmse": 1.0})
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    include_rmse_in_multi: bool = True
    multi_objectives: list[str] = field(default_factory=list)
    multi_objectives_use_error: bool = True
    target_values: dict[str, float] = field(default_factory=dict)


@dataclass
class LoggingSpec:
    level: str = "INFO"
    output_dir: str = "output/logs"


@dataclass
class PathsSpec:
    target_curve: str = "input/target_curve.csv"
    input_dir: str = "input"
    result_dir: str = "output"
    vexis_path: str = "vexis"


@dataclass
class FreecadSpec:
    fcstd_path: str = "input/model.FCStd"
    sketch_name: str = "Sketch001"
    surface_name: str = "Face"
    surface_label: str = "SURFACE"
    constraints_domain: str = "ratio"
    constraints: dict[str, dict[str, float]] = field(default_factory=dict)
    step_output_dir: str = "input/step"
    step_filename_template: str = "v1_0_trial_{trial_id}.step"
    timeout_sec: int = 300


@dataclass
class V1Config:
    """Aggregated configuration for a v1.0 run."""

    optimization: OptimizationSpec = field(default_factory=OptimizationSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    paths: PathsSpec = field(default_factory=PathsSpec)
    freecad: FreecadSpec = field(default_factory=FreecadSpec)
    cad_gate: CadGateSpec = field(default_factory=CadGateSpec)
    cae: CaeSpec = field(default_factory=CaeSpec)
    penalty: PenaltySpec = field(default_factory=PenaltySpec)
    bounds: list[BoundsSpec] = field(default_factory=list)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def load_config(
    optimizer_config_path: Path,
    limits_path: Path,
) -> V1Config:
    """Load and validate both YAML files into a single v1.0 config object."""

    if not optimizer_config_path.exists():
        raise FileNotFoundError(f"Config not found: {optimizer_config_path}")
    if not limits_path.exists():
        raise FileNotFoundError(f"Limits not found: {limits_path}")

    raw_opt = _load_yaml(optimizer_config_path)
    raw_lim = _load_yaml(limits_path)

    # --- Optimization ---
    opt_raw = raw_opt.get("optimization", {})
    optimization = OptimizationSpec(
        sampler=opt_raw.get("sampler", "AUTO"),
        max_trials=int(opt_raw.get("max_trials", 300)),
        convergence_threshold=float(opt_raw.get("convergence_threshold", 0.001)),
        patience=int(opt_raw.get("patience", 100)),
        seed=int(opt_raw.get("seed", 42)),
        n_startup_trials=int(opt_raw.get("n_startup_trials", 70)),
        discretization_step=opt_raw.get("discretization_step"),
        enable_dimension_discretization=bool(
            opt_raw.get("enable_dimension_discretization", True)
        ),
        non_angle_step=float(opt_raw.get("non_angle_step", 0.01)),
        angle_step=float(opt_raw.get("angle_step", 0.001)),
        angle_name_token=str(opt_raw.get("angle_name_token", "ANGLE")),
        objective_type=opt_raw.get("objective_type", "single"),
        directions=[str(d).lower() for d in opt_raw.get("directions", ["minimize"])],
    )

    # --- Objective ---
    obj_raw = raw_opt.get("objective", {})
    objective = ObjectiveSpec(
        type=obj_raw.get("type", "rmse"),
        weights=obj_raw.get("weights", {"rmse": 1.0}),
        features=obj_raw.get("features", {}),
        include_rmse_in_multi=bool(obj_raw.get("include_rmse_in_multi", True)),
        multi_objectives=[str(x) for x in obj_raw.get("multi_objectives", [])],
        multi_objectives_use_error=bool(obj_raw.get("multi_objectives_use_error", True)),
        target_values={
            str(k): float(v)
            for k, v in (obj_raw.get("target_values", {}) or {}).items()
        },
    )

    # --- Paths ---
    paths_raw = raw_opt.get("paths", {})
    paths = PathsSpec(
        target_curve=paths_raw.get("target_curve", "input/target_curve.csv"),
        input_dir=paths_raw.get("input_dir", "input"),
        result_dir=paths_raw.get("result_dir", "output"),
        vexis_path=paths_raw.get("vexis_path", "vexis"),
    )

    # --- Logging ---
    log_raw = raw_opt.get("logging", {})
    logging_spec = LoggingSpec(
        level=log_raw.get("level", "INFO"),
        output_dir=log_raw.get("output_dir", "output/logs"),
    )

    # --- FreeCAD ---
    fc_raw = raw_lim.get("freecad", {})
    constraints_raw = fc_raw.get("constraints", {})
    freecad = FreecadSpec(
        fcstd_path=fc_raw.get("fcstd_path", "input/model.FCStd"),
        sketch_name=fc_raw.get("sketch_name", "Sketch001"),
        surface_name=fc_raw.get("surface_name", "Face"),
        surface_label=fc_raw.get("surface_label", "SURFACE"),
        constraints_domain=str(fc_raw.get("constraints_domain", "ratio")).strip().lower(),
        constraints=constraints_raw,
        step_output_dir=fc_raw.get("step_output_dir", "input/step"),
        step_filename_template=fc_raw.get(
            "step_filename_template", "v1_0_trial_{trial_id}.step"
        ),
        timeout_sec=int(fc_raw.get("timeout_sec", 300)),
    )

    # --- CAD Gate ---
    gate_raw = raw_lim.get("cad_gate", {})
    cad_gate = CadGateSpec(
        model_path=gate_raw.get("model_path"),
        threshold=float(gate_raw.get("threshold", 0.5)),
        enabled=gate_raw.get("enabled", True),
        rejection_max_retries=int(gate_raw.get("rejection_max_retries", 50)),
    )

    # --- CAE ---
    cae_raw = raw_lim.get("cae", {})
    sr = cae_raw.get("stroke_range", {})
    raw_markers = cae_raw.get("solver_error_markers", ["error termination", "fatal error"])
    if isinstance(raw_markers, str):
        raw_markers = [raw_markers]
    solver_error_markers = [str(m).strip().lower() for m in raw_markers if str(m).strip()]
    if not solver_error_markers:
        solver_error_markers = ["error termination", "fatal error"]
    cae = CaeSpec(
        stroke_range_min=float(sr.get("min", 0.0)),
        stroke_range_max=float(sr.get("max", 0.5)),
        solver_progress_stall_sec=int(cae_raw.get("solver_progress_stall_sec", 300)),
        solver_log_poll_sec=float(cae_raw.get("solver_log_poll_sec", 1.0)),
        solver_log_activity_resets_progress_stall=bool(
            cae_raw.get("solver_log_activity_resets_progress_stall", True)
        ),
        solver_hard_timeout_sec=(
            int(cae_raw["solver_hard_timeout_sec"])
            if cae_raw.get("solver_hard_timeout_sec") is not None
            else 3600
        ),
        solver_error_markers=solver_error_markers,
        max_retries=int(cae_raw.get("max_retries", 1)),
        stream_stdout=bool(cae_raw.get("stream_stdout", False)),
        stdout_log_dir=cae_raw.get("stdout_log_dir"),
        stdout_console_level=cae_raw.get("stdout_console_level", "INFO"),
    )

    # --- Penalty ---
    pen_raw = raw_lim.get("penalty", {})
    penalty = PenaltySpec(
        base_penalty=float(pen_raw.get("base_penalty", 50.0)),
        alpha=float(pen_raw.get("alpha", 10.0)),
        failure_weights=pen_raw.get("failure_weights", {}),
    )

    # --- Bounds (from constraints) ---
    bounds: list[BoundsSpec] = []
    for name, spec in constraints_raw.items():
        if isinstance(spec, dict) and "min" in spec and "max" in spec:
            bounds.append(
                BoundsSpec(
                    name=name,
                    min=float(spec["min"]),
                    max=float(spec["max"]),
                    base_value=float(spec.get("base_value", 1.0)),
                )
            )

    cfg = V1Config(
        optimization=optimization,
        objective=objective,
        logging=logging_spec,
        paths=paths,
        freecad=freecad,
        cad_gate=cad_gate,
        cae=cae,
        penalty=penalty,
        bounds=bounds,
    )

    _validate(cfg)
    logger.info("v1.0 config loaded: %d design variables", len(cfg.bounds))
    return cfg


def _validate(cfg: V1Config) -> None:
    """Raise on obviously invalid configuration."""
    if not cfg.bounds:
        raise ValueError("No design variables defined in freecad.constraints")
    if cfg.freecad.constraints_domain not in {"ratio", "physical"}:
        raise ValueError(
            "freecad.constraints_domain must be 'ratio' or 'physical'"
        )
    for b in cfg.bounds:
        if b.min >= b.max:
            raise ValueError(f"Invalid bounds for {b.name}: min={b.min} >= max={b.max}")
    if cfg.optimization.max_trials < 1:
        raise ValueError("max_trials must be >= 1")
    if cfg.optimization.non_angle_step <= 0:
        raise ValueError("optimization.non_angle_step must be > 0")
    if cfg.optimization.angle_step <= 0:
        raise ValueError("optimization.angle_step must be > 0")
    if not cfg.optimization.angle_name_token:
        raise ValueError("optimization.angle_name_token must not be empty")
    valid_directions = {"minimize", "maximize"}
    if not cfg.optimization.directions:
        raise ValueError("optimization.directions must not be empty")
    for d in cfg.optimization.directions:
        if d not in valid_directions:
            raise ValueError(
                f"Invalid direction '{d}'. Use one of {sorted(valid_directions)}"
            )
    if cfg.optimization.objective_type == "multi":
        feature_names = set(cfg.objective.features.keys())
        objective_names = cfg.objective.multi_objectives or list(feature_names)
        if not cfg.objective.include_rmse_in_multi and not objective_names:
            raise ValueError(
                "objective_type=multi requires at least one target in objective.multi_objectives "
                "or objective.features when include_rmse_in_multi=false"
            )
        for name in objective_names:
            if name not in feature_names:
                raise ValueError(
                    f"Unknown multi objective '{name}'. "
                    f"Define it under objective.features first."
                )
        for name, value in cfg.objective.target_values.items():
            if name not in feature_names:
                raise ValueError(
                    f"Unknown target_values key '{name}'. "
                    f"Define it under objective.features first."
                )
            if not isinstance(value, (int, float)):
                raise ValueError(f"objective.target_values['{name}'] must be numeric")
        n_obj = len(objective_names) + (1 if cfg.objective.include_rmse_in_multi else 0)
        n_obj = max(1, n_obj)
        if len(cfg.optimization.directions) not in (1, n_obj):
            raise ValueError(
                "optimization.directions length must be 1 or match objective count "
                f"({n_obj}) for objective_type=multi"
            )
    if cfg.cae.max_retries < 1:
        raise ValueError("cae.max_retries must be >= 1")
    if cfg.cae.solver_progress_stall_sec < 1:
        raise ValueError("cae.solver_progress_stall_sec must be >= 1")
    if cfg.cae.solver_log_poll_sec <= 0:
        raise ValueError("cae.solver_log_poll_sec must be > 0")
    if cfg.cae.solver_hard_timeout_sec is not None and cfg.cae.solver_hard_timeout_sec < 1:
        raise ValueError("cae.solver_hard_timeout_sec must be >= 1 when set")
    if not cfg.cae.solver_error_markers:
        raise ValueError("cae.solver_error_markers must not be empty")
