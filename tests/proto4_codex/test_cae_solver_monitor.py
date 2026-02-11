"""Tests for solver-log monitoring helpers in proto4-codex CAE evaluator."""

from proto4_codex.cae_evaluator import (
    CaeEvaluator,
    _detect_solver_error_marker,
    _extract_solver_progress,
)


def test_extract_solver_progress_ratio() -> None:
    line = "Solver Time:  59%|#####8    | 5.86834/10.0 [01:03<00:44]"
    ratio = _extract_solver_progress(line)
    assert ratio is not None
    assert abs(ratio - 0.586834) < 1e-6


def test_extract_solver_progress_none_without_solver_marker() -> None:
    assert _extract_solver_progress("Info: Meshing order 2") is None


def test_solver_stall_reason_format() -> None:
    reason = CaeEvaluator._format_solver_progress_stall_reason(0.5, 180)
    assert reason == "solver_progress_stalled_at_50.0pct_for_180s"


def test_solver_start_reason_format() -> None:
    reason = CaeEvaluator._format_solver_start_reason(120)
    assert reason == "solver_log_not_started_for_120s"


def test_detect_solver_error_marker() -> None:
    marker = _detect_solver_error_marker(
        "Error Termination in nonlinear solver",
        ("error termination", "fatal error"),
    )
    assert marker == "error termination"


def test_solver_error_reason_format() -> None:
    reason = CaeEvaluator._format_solver_error_reason("fatal error")
    assert reason == "solver_error_marker:fatal_error"


def test_progress_stall_ignored_after_solver_reaches_100pct() -> None:
    stalled = CaeEvaluator._is_solver_progress_stalled(
        progress=1.0,
        elapsed_since_progress_sec=999.0,
        stall_sec=180,
    )
    assert stalled is False
