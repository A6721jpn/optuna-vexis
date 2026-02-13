"""
v1.0 version metadata helpers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

PRODUCT_NAME = "optuna-for-vexis"
PRODUCT_LINE = "Production"
PRODUCT_VERSION = "1.0.0"
BASELINE = "v1.0"


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def get_version_info(project_root: Path) -> dict[str, Any]:
    """Return runtime version info with optional git metadata."""
    commit = _git_output(project_root, "rev-parse", "--short", "HEAD")
    branch = _git_output(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output(project_root, "status", "--porcelain")
    dirty = bool(status) if status is not None else None
    return {
        "product": PRODUCT_NAME,
        "line": PRODUCT_LINE,
        "version": PRODUCT_VERSION,
        "baseline": BASELINE,
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
    }
