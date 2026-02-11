"""Tests for proto4-codex config logging/CAE extensions."""

import yaml
from pathlib import Path

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
