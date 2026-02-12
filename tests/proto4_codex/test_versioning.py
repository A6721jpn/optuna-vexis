"""Tests for production version metadata helpers."""

from __future__ import annotations

from pathlib import Path

from proto4_codex.versioning import get_version_info


def test_get_version_info_has_required_keys() -> None:
    info = get_version_info(Path(__file__).resolve().parent.parent.parent)
    assert info["line"] == "Production"
    assert info["version"] == "1.0.0"
    assert info["baseline"] == "proto4"
    assert "git_commit" in info
    assert "git_branch" in info
    assert "git_dirty" in info

