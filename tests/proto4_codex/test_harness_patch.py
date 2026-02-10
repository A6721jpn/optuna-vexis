"""Tests for proto4-codex harness config patching."""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
HARNESS = ROOT / "scripts" / "proto4_harness.py"

spec = importlib.util.spec_from_file_location("proto4_harness", str(HARNESS))
mod = importlib.util.module_from_spec(spec)
sys.modules["proto4_harness"] = mod
spec.loader.exec_module(mod)


def test_patch_limits_adds_stdout_fields(tmp_path: Path):
    limits = tmp_path / "limits.yaml"
    limits.write_text("cae:\n  timeout_sec: 10\n")

    class DummyCtx:
        logs_dir = tmp_path / "logs"
        artifacts_dir = tmp_path

    patched = mod._patch_limits_for_harness(limits, DummyCtx())
    content = patched.read_text(encoding="utf-8")
    assert "stream_stdout" in content
    assert "stdout_log_dir" in content
    assert "stdout_console_level" in content
