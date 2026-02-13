"""
v1.0 Persistence

Per-trial artifact and metadata persistence for traceability and
reproducibility.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import TrialRecord

logger = logging.getLogger(__name__)


class TrialPersistence:
    """Save and query trial records as JSON artefacts."""

    def __init__(self, result_dir: Path) -> None:
        self._result_dir = result_dir
        self._trials_dir = result_dir / "trials"
        self._trials_dir.mkdir(parents=True, exist_ok=True)

    def save_trial(self, record: TrialRecord) -> Path:
        """Write a TrialRecord to ``<trials_dir>/trial_<id>/trial_info.json``."""
        trial_dir = self._trials_dir / f"trial_{record.trial_id}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        data = record.to_dict()
        data["saved_at"] = datetime.now().isoformat()

        out_path = trial_dir / "trial_info.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.debug("Trial %d persisted to %s", record.trial_id, out_path)
        return out_path

    def save_run_config(self, config_dict: dict[str, Any]) -> Path:
        """Snapshot the full configuration used for this run."""
        out_path = self._result_dir / "run_config_snapshot.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Run config snapshot saved: %s", out_path)
        return out_path

    def save_summary(self, summary: dict[str, Any]) -> Path:
        """Write the final optimization summary."""
        out_path = self._result_dir / "summary_v1_0.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Summary saved: %s", out_path)
        return out_path

    def load_trial(self, trial_id: int) -> dict[str, Any]:
        """Load a single trial record."""
        path = self._trials_dir / f"trial_{trial_id}" / "trial_info.json"
        if not path.exists():
            raise FileNotFoundError(f"Trial record not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
