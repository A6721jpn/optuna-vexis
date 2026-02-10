"""Tests for proto4-codex harness config patching."""

from pathlib import Path

from scripts.proto4_harness import _patch_limits_for_harness


def test_patch_limits_adds_stdout_fields(tmp_path: Path):
    limits = tmp_path / "limits.yaml"
    limits.write_text("cae:\n  timeout_sec: 10\n")

    class DummyCtx:
        logs_dir = tmp_path / "logs"
        artifacts_dir = tmp_path

    patched = _patch_limits_for_harness(limits, DummyCtx())
    content = patched.read_text(encoding="utf-8")
    assert "stream_stdout" in content
    assert "stdout_log_dir" in content
    assert "stdout_console_level" in content
