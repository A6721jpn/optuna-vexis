"""Tests for proto4-codex config logging/CAE extensions."""

from pathlib import Path

import pytest
import yaml

from proto4_codex.config import load_config


def test_logging_and_cae_fields_loaded(tmp_path: Path):
    opt = tmp_path / "opt.yaml"
    lim = tmp_path / "lim.yaml"

    opt.write_text(yaml.safe_dump({
        "optimization": {},
        "objective": {},
        "paths": {},
        "logging": {"level": "DEBUG", "output_dir": "output/logs"},
    }))
    lim.write_text(yaml.safe_dump({
        "freecad": {"constraints": {"x": {"min": 0.0, "max": 1.0}}},
        "cae": {
            "stroke_range": {"min": 0.0, "max": 0.5},
            "solver_progress_stall_sec": 180,
            "solver_log_poll_sec": 0.5,
            "solver_error_markers": ["ERROR TERMINATION", "FATAL ERROR"],
            "stream_stdout": True,
            "stdout_log_dir": "output/logs/vexis",
            "stdout_console_level": "WARNING",
        },
    }))

    cfg = load_config(opt, lim)
    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.output_dir == "output/logs"
    assert cfg.cae.stream_stdout is True
    assert cfg.cae.solver_progress_stall_sec == 180
    assert cfg.cae.solver_log_poll_sec == 0.5
    assert cfg.cae.solver_error_markers == ["error termination", "fatal error"]
    assert cfg.cae.stdout_log_dir == "output/logs/vexis"
    assert cfg.cae.stdout_console_level == "WARNING"
    assert cfg.optimization.enable_dimension_discretization is True
    assert cfg.optimization.non_angle_step == 0.01
    assert cfg.optimization.angle_step == 0.001
    assert cfg.optimization.angle_name_token == "ANGLE"


def test_multi_objective_config_loaded_for_feature_only_mode(tmp_path: Path):
    opt = tmp_path / "opt_multi.yaml"
    lim = tmp_path / "lim_multi.yaml"

    opt.write_text(yaml.safe_dump({
        "optimization": {"objective_type": "multi"},
        "objective": {
            "type": "multi",
            "include_rmse_in_multi": False,
            "multi_objectives_use_error": False,
            "multi_objectives": ["peak_position", "peak_force"],
            "features": {
                "peak_position": {"type": "peak_position", "column": "force"},
                "peak_force": {"type": "local_max", "column": "force"},
            },
        },
        "paths": {},
    }))
    lim.write_text(yaml.safe_dump({
        "freecad": {"constraints": {"x": {"min": 0.0, "max": 1.0}}},
    }))

    cfg = load_config(opt, lim)
    assert cfg.optimization.objective_type == "multi"
    assert cfg.objective.include_rmse_in_multi is False
    assert cfg.objective.multi_objectives_use_error is False
    assert cfg.objective.multi_objectives == ["peak_position", "peak_force"]


def test_multi_objective_unknown_target_is_rejected(tmp_path: Path):
    opt = tmp_path / "opt_multi_invalid.yaml"
    lim = tmp_path / "lim_multi_invalid.yaml"

    opt.write_text(yaml.safe_dump({
        "optimization": {"objective_type": "multi"},
        "objective": {
            "type": "multi",
            "include_rmse_in_multi": False,
            "multi_objectives": ["peak_position", "unknown_feature"],
            "features": {
                "peak_position": {"type": "peak_position", "column": "force"},
                "peak_force": {"type": "local_max", "column": "force"},
            },
        },
        "paths": {},
    }))
    lim.write_text(yaml.safe_dump({
        "freecad": {"constraints": {"x": {"min": 0.0, "max": 1.0}}},
    }))

    with pytest.raises(ValueError, match="Unknown multi objective 'unknown_feature'"):
        load_config(opt, lim)


def test_multi_objective_direction_length_is_validated(tmp_path: Path):
    opt = tmp_path / "opt_multi_directions.yaml"
    lim = tmp_path / "lim_multi_directions.yaml"

    opt.write_text(yaml.safe_dump({
        "optimization": {
            "objective_type": "multi",
            "directions": ["maximize", "maximize", "minimize"],
        },
        "objective": {
            "type": "multi",
            "include_rmse_in_multi": False,
            "multi_objectives": ["click_ratio", "peak_force"],
            "features": {
                "click_ratio": {"type": "click_ratio", "column": "force"},
                "peak_force": {"type": "peak_force", "column": "force"},
            },
        },
        "paths": {},
    }))
    lim.write_text(yaml.safe_dump({
        "freecad": {"constraints": {"x": {"min": 0.0, "max": 1.0}}},
    }))

    with pytest.raises(ValueError, match="optimization.directions length must be 1 or match objective count"):
        load_config(opt, lim)
