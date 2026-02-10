"""
Proto4 Configuration

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
    timeout_sec: int = 600
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
    objective_type: str = "single"
    directions: list[str] = field(default_factory=lambda: ["minimize"])


@dataclass
class ObjectiveSpec:
    type: str = "rmse"
    weights: dict[str, float] = field(default_factory=lambda: {"rmse": 1.0})
    features: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    constraints: dict[str, dict[str, float]] = field(default_factory=dict)
    step_output_dir: str = "input/step"
    step_filename_template: str = "proto4_trial_{trial_id}.step"
    timeout_sec: int = 300


@dataclass
class Proto4Config:
    """Aggregated configuration for a Proto4 run."""

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
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    optimizer_config_path: Path,
    limits_path: Path,
) -> Proto4Config:
    """Load and validate both YAML files into a single Proto4Config."""

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
        objective_type=opt_raw.get("objective_type", "single"),
    )

    # --- Objective ---
    obj_raw = raw_opt.get("objective", {})
    objective = ObjectiveSpec(
        type=obj_raw.get("type", "rmse"),
        weights=obj_raw.get("weights", {"rmse": 1.0}),
        features=obj_raw.get("features", {}),
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
        constraints=constraints_raw,
        step_output_dir=fc_raw.get("step_output_dir", "input/step"),
        step_filename_template=fc_raw.get(
            "step_filename_template", "proto4_trial_{trial_id}.step"
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
    cae = CaeSpec(
        stroke_range_min=float(sr.get("min", 0.0)),
        stroke_range_max=float(sr.get("max", 0.5)),
        timeout_sec=int(cae_raw.get("timeout_sec", 600)),
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
            bounds.append(BoundsSpec(name=name, min=float(spec["min"]), max=float(spec["max"])))

    cfg = Proto4Config(
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
    logger.info("Proto4 config loaded: %d design variables", len(cfg.bounds))
    return cfg


def _validate(cfg: Proto4Config) -> None:
    """Raise on obviously invalid configuration."""
    if not cfg.bounds:
        raise ValueError("No design variables defined in freecad.constraints")
    for b in cfg.bounds:
        if b.min >= b.max:
            raise ValueError(f"Invalid bounds for {b.name}: min={b.min} >= max={b.max}")
    if cfg.optimization.max_trials < 1:
        raise ValueError("max_trials must be >= 1")
