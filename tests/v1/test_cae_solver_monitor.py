from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _ensure_v1_package_loaded() -> None:
    if "v1" in sys.modules:
        return
    project_root = Path(__file__).resolve().parents[2]
    pkg_dir = project_root / "src" / "v1"
    spec = importlib.util.spec_from_file_location(
        "v1",
        str(pkg_dir / "__init__.py"),
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load v1 package for tests")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1"] = mod
    spec.loader.exec_module(mod)


_ensure_v1_package_loaded()

from v1.cae_evaluator import CaeEvaluator, _extract_solver_progress  # noqa: E402


def test_extract_solver_progress_ratio() -> None:
    line = "Solver Time:  59%|#####8    | 5.86834/10.0 [01:03<00:44]"
    ratio = _extract_solver_progress(line)
    assert ratio is not None
    assert abs(ratio - 0.586834) < 1e-6


def test_progress_stall_resets_on_recent_log_activity() -> None:
    stalled = CaeEvaluator._is_solver_progress_stalled(
        progress=0.4,
        elapsed_since_progress_sec=120.0,
        stall_sec=60,
        log_activity_age_sec=5.0,
        reset_on_log_activity=True,
    )
    assert stalled is False


def test_progress_stall_triggers_without_recent_log_activity() -> None:
    stalled = CaeEvaluator._is_solver_progress_stalled(
        progress=0.4,
        elapsed_since_progress_sec=120.0,
        stall_sec=60,
        log_activity_age_sec=120.0,
        reset_on_log_activity=True,
    )
    assert stalled is True


def test_progress_stall_ignores_log_activity_when_disabled() -> None:
    stalled = CaeEvaluator._is_solver_progress_stalled(
        progress=0.4,
        elapsed_since_progress_sec=120.0,
        stall_sec=60,
        log_activity_age_sec=1.0,
        reset_on_log_activity=False,
    )
    assert stalled is True


def test_solver_hard_timeout_reason_format() -> None:
    reason = CaeEvaluator._format_solver_hard_timeout_reason(0.349, 3600)
    assert reason == "solver_hard_timeout_at_34.9pct_for_3600s"

