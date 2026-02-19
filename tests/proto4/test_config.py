"""Tests for proto4-claude config loading and validation."""

import pytest
import yaml
from pathlib import Path

from proto4_claude.config import Proto4Config, load_config


class TestLoadConfig:
    def test_loads_valid_config(self, proto4_config: Proto4Config):
        assert proto4_config.optimization.sampler == "TPE"
        assert proto4_config.optimization.max_trials == 20
        assert proto4_config.optimization.seed == 42
        assert len(proto4_config.bounds) == 2

    def test_bounds_parsed(self, proto4_config: Proto4Config):
        names = {b.name for b in proto4_config.bounds}
        assert names == {"width", "height"}
        for b in proto4_config.bounds:
            if b.name == "width":
                assert b.min == 10.0
                assert b.max == 20.0

    def test_cad_gate_defaults(self, proto4_config: Proto4Config):
        assert proto4_config.cad_gate.enabled is True
        assert proto4_config.cad_gate.threshold == 0.5
        assert proto4_config.cad_gate.rejection_max_retries == 30

    def test_penalty_spec(self, proto4_config: Proto4Config):
        assert proto4_config.penalty.base_penalty == 50.0
        assert proto4_config.penalty.failure_weights["cae_fail"] == 0.6

    def test_cae_spec(self, proto4_config: Proto4Config):
        assert proto4_config.cae.stroke_range_min == 0.0
        assert proto4_config.cae.stroke_range_max == 0.5
        assert proto4_config.cae.max_retries == 1


class TestValidation:
    def test_missing_config_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml", tmp_path / "also_missing.yaml")

    def test_empty_constraints_rejected(self, tmp_path: Path):
        opt_cfg = tmp_path / "opt.yaml"
        lim_cfg = tmp_path / "lim.yaml"
        opt_cfg.write_text(yaml.dump({"optimization": {}, "paths": {}}))
        lim_cfg.write_text(yaml.dump({"freecad": {"constraints": {}}}))
        with pytest.raises(ValueError, match="No design variables"):
            load_config(opt_cfg, lim_cfg)

    def test_inverted_bounds_rejected(self, tmp_path: Path):
        opt_cfg = tmp_path / "opt.yaml"
        lim_cfg = tmp_path / "lim.yaml"
        opt_cfg.write_text(yaml.dump({"optimization": {}, "paths": {}}))
        lim_cfg.write_text(yaml.dump({
            "freecad": {"constraints": {"x": {"min": 20.0, "max": 5.0}}},
        }))
        with pytest.raises(ValueError, match="Invalid bounds"):
            load_config(opt_cfg, lim_cfg)
