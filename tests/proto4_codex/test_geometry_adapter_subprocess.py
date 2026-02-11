"""Tests for proto4-codex GeometryAdapter subprocess execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from proto4_codex.config import FreecadSpec
from proto4_codex.geometry_adapter import GeometryAdapter, GeometryError
from proto4_codex.types import DesignPoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_adapter(tmp_path: Path) -> GeometryAdapter:
    spec = FreecadSpec(
        fcstd_path=str(tmp_path / "input" / "model.FCStd"),
        sketch_name="Sketch",
        surface_name="Face",
        surface_label="SURFACE",
        constraints={"A": {"min": 0.9, "max": 1.1}},
        step_output_dir=str(tmp_path / "input" / "step"),
        step_filename_template="proto4_trial_{trial_id}.step",
        timeout_sec=30,
    )
    return GeometryAdapter(spec, PROJECT_ROOT)


def test_generate_step_runs_freecad_worker_subprocess(tmp_path: Path, monkeypatch) -> None:
    adapter = _build_adapter(tmp_path)
    (tmp_path / "input").mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "step").mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "model.FCStd").write_text("dummy fcstd", encoding="utf-8")

    fake_fc_python = tmp_path / "fcbin" / "python.exe"
    fake_fc_python.parent.mkdir(parents=True, exist_ok=True)
    fake_fc_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("FREECAD_PYTHON", str(fake_fc_python))

    called = {}

    def _fake_run(cmd, cwd, stdout, stderr, text, timeout, check):
        called["cmd"] = cmd
        step_idx = cmd.index("--step-path") + 1
        step_path = Path(cmd[step_idx])
        step_path.parent.mkdir(parents=True, exist_ok=True)
        step_path.write_text("step", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    point = DesignPoint(trial_id=7, params={"A": 1.0})
    with patch("proto4_codex.geometry_adapter.subprocess.run", side_effect=_fake_run):
        step_path = adapter.generate_step(point)

    assert step_path.exists()
    assert called["cmd"][0] == str(fake_fc_python)
    assert "freecad_worker.py" in called["cmd"][1]
    assert "--params-json" in called["cmd"]


def test_generate_step_raises_on_worker_failure(tmp_path: Path, monkeypatch) -> None:
    adapter = _build_adapter(tmp_path)
    (tmp_path / "input").mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "step").mkdir(parents=True, exist_ok=True)
    (tmp_path / "input" / "model.FCStd").write_text("dummy fcstd", encoding="utf-8")

    fake_fc_python = tmp_path / "fcbin" / "python.exe"
    fake_fc_python.parent.mkdir(parents=True, exist_ok=True)
    fake_fc_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("FREECAD_PYTHON", str(fake_fc_python))

    def _fake_run(cmd, cwd, stdout, stderr, text, timeout, check):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")

    point = DesignPoint(trial_id=9, params={"A": 1.0})
    with patch("proto4_codex.geometry_adapter.subprocess.run", side_effect=_fake_run):
        with pytest.raises(GeometryError, match="worker failed"):
            adapter.generate_step(point)
