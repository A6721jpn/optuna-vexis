"""Readiness tests for `src/proto4-codex` before bug-fix implementation.

These tests encode expected behavior from `doc/proto4.md` for known gaps,
and are intended to fail until the implementation is fixed.
"""

from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_feasibility_attr_name_matches_spec() -> None:
    """Spec requires `proto4_feasibility_violation` user_attr key."""
    content = _read("src/proto4-codex/search_space.py")
    assert 'FEASIBILITY_ATTR = "proto4_feasibility_violation"' in content


def test_sampler_enforces_rejection_in_independent_path() -> None:
    """Independent sampling path should include feasibility rejection logic."""
    content = _read("src/proto4-codex/search_space.py")

    # Guard against current deferred behavior comment.
    assert "Rejection is" not in content or "deferred to the objective orchestrator" not in content

    # Expect presence of some feasibility check in sample_independent path.
    assert "def sample_independent(" in content
    assert "_predict_fn" in content


def test_dry_run_does_not_mark_cad_infeasible() -> None:
    """Dry-run should not classify trial as CAD infeasible."""
    content = _read("src/proto4-codex/objective.py")
    dry_run_block = "if dry_run:" in content
    assert dry_run_block
    assert "record.outcome = TrialOutcome.CAD_INFEASIBLE" not in content.split("if dry_run:", 1)[1][:300]

