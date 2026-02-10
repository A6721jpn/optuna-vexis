"""
Proto4 CAE Evaluator

Wraps VEXIS subprocess execution, result CSV loading, curve processing,
and metric extraction into a single coherent unit.

Consolidates proto3 vexis_runner + result_loader + curve_processor.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate, signal as sp_signal

from .config import CaeSpec, ObjectiveSpec
from .types import CaeResult, CaeStatus, DesignPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curve helpers (from proto3 curve_processor / result_loader)
# ---------------------------------------------------------------------------

_DISP_CANDIDATES = ["Stroke", "stroke", "Displacement", "displacement", "x", "X"]
_FORCE_CANDIDATES = [
    "Reaction_Force", "reaction_force", "Adjusted force", "adjusted_force",
    "Force", "force", "Reaction", "reaction", "y", "Y", "Fz", "fz",
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_curve(csv_path: Path) -> pd.DataFrame:
    """Load a CSV into a normalised ``displacement`` / ``force`` DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Result CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    disp_col = _find_col(df, _DISP_CANDIDATES)
    force_col = _find_col(df, _FORCE_CANDIDATES)

    if disp_col is None or force_col is None:
        if len(df.columns) >= 2:
            disp_col, force_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(f"Insufficient columns in {csv_path}")

    return pd.DataFrame({
        "displacement": pd.to_numeric(df[disp_col], errors="coerce"),
        "force": pd.to_numeric(df[force_col], errors="coerce"),
    }).dropna().reset_index(drop=True)


def extract_range(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    mask = (df["displacement"] >= lo) & (df["displacement"] <= hi)
    return df[mask].copy().reset_index(drop=True)


def split_cycle(df: pd.DataFrame):
    if df.empty:
        return df, None
    max_idx = df["displacement"].idxmax()
    loading = df.iloc[: max_idx + 1].copy().reset_index(drop=True)
    unloading = None
    if max_idx < len(df) - 1:
        unloading = df.iloc[max_idx:].copy().reset_index(drop=True)
        if len(unloading) < 2:
            unloading = None
    return loading, unloading


def calculate_rmse(result: pd.DataFrame, target: pd.DataFrame, n_points: int = 100) -> float:
    disp_min = max(result["displacement"].min(), target["displacement"].min())
    disp_max = min(result["displacement"].max(), target["displacement"].max())
    if disp_min >= disp_max:
        return float("inf")

    common_disp = np.linspace(disp_min, disp_max, n_points)

    res_interp = interpolate.interp1d(
        result["displacement"].values, result["force"].values,
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    tgt_interp = interpolate.interp1d(
        target["displacement"].values, target["force"].values,
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    return float(np.sqrt(np.mean((res_interp(common_disp) - tgt_interp(common_disp)) ** 2)))


def extract_features(df: pd.DataFrame, feature_config: dict) -> dict[str, float]:
    """Extract features from a curve DataFrame following feature_config."""
    features: dict[str, float] = {}
    for name, cfg in feature_config.items():
        feat_type = cfg.get("type", "max")
        column = cfg.get("column", "force")
        if column not in df.columns:
            continue

        if feat_type == "max":
            features[name] = float(df[column].max())
        elif feat_type == "min":
            features[name] = float(df[column].min())
        elif feat_type == "mean":
            features[name] = float(df[column].mean())
        elif feat_type == "slope":
            r = cfg.get("range", [0.0, 0.5])
            sub = df[(df["displacement"] >= r[0]) & (df["displacement"] <= r[1])]
            if len(sub) >= 2:
                slope, _ = np.polyfit(sub["displacement"].values, sub[column].values, 1)
                features[name] = float(slope)
            else:
                features[name] = 0.0
        elif feat_type == "peak_position":
            features[name] = float(df.loc[df[column].idxmax(), "displacement"])
        elif feat_type == "value_at":
            at = cfg.get("at", 0.5)
            f = interpolate.interp1d(
                df["displacement"].values, df[column].values,
                kind="linear", bounds_error=False, fill_value="extrapolate",
            )
            features[name] = float(f(at))
        elif feat_type == "local_max":
            y = df[column].values
            prominence = cfg.get("prominence", 0.05 * (y.max() - y.min()))
            peaks, _ = sp_signal.find_peaks(y, prominence=prominence)
            features[name] = float(y[peaks].max()) if len(peaks) else float(y.max())

    return features


# ---------------------------------------------------------------------------
# VEXIS subprocess runner (from proto3 vexis_runner)
# ---------------------------------------------------------------------------

class CaeEvaluator:
    """Run VEXIS, load result CSV, and compute metrics."""

    def __init__(
        self,
        vexis_path: Path,
        cae_spec: CaeSpec,
        obj_spec: ObjectiveSpec,
        target_curve: pd.DataFrame,
        target_features: dict[str, float],
    ) -> None:
        self._vexis_root = vexis_path
        self._cae_spec = cae_spec
        self._obj_spec = obj_spec
        self._target_curve = target_curve
        self._target_features = target_features
        self._current_proc: Optional[subprocess.Popen] = None
        self._stop_requested = False

        if not self._vexis_root.exists():
            raise FileNotFoundError(f"VEXIS not found: {self._vexis_root}")

        self._vexis_input = self._vexis_root / "input"
        self._vexis_results = self._vexis_root / "results"
        self._vexis_main = self._vexis_root / "main.py"

        # Pre-split target curve
        self._tgt_load, self._tgt_unload = split_cycle(target_curve)
        if self._tgt_load is None or len(self._tgt_load) < 2:
            raise ValueError("Target curve has no valid loading segment")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, step_path: Path, point: DesignPoint) -> CaeResult:
        """Run full CAE evaluation pipeline with retry."""
        job_name = f"proto4_trial_{point.trial_id}"
        retries = self._cae_spec.max_retries

        for attempt in range(1, retries + 1):
            t0 = time.time()
            result = self._single_run(step_path, job_name)
            elapsed = time.time() - t0
            result.runtime_sec = elapsed

            if result.status == CaeStatus.SUCCESS:
                return result

            if attempt < retries:
                logger.warning("CAE attempt %d/%d failed; retrying", attempt, retries)

        return result  # last failed attempt

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _single_run(self, step_path: Path, job_name: str) -> CaeResult:
        """Execute one VEXIS run and compute metrics."""
        # Copy STEP to VEXIS input
        self._vexis_input.mkdir(parents=True, exist_ok=True)
        target_step = self._vexis_input / f"{job_name}.step"
        shutil.copy2(step_path, target_step)

        # Run VEXIS subprocess
        result_csv = self._run_subprocess(job_name)
        if result_csv is None:
            return CaeResult(status=CaeStatus.FAIL)

        # Load and process result
        try:
            result_curve = load_curve(result_csv)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to load result: %s", exc)
            return CaeResult(status=CaeStatus.FAIL)

        trimmed = extract_range(
            result_curve,
            self._cae_spec.stroke_range_min,
            self._cae_spec.stroke_range_max,
        )

        # Compute metrics
        metrics = self._compute_metrics(trimmed)
        if metrics is None:
            return CaeResult(status=CaeStatus.FAIL)

        return CaeResult(
            status=CaeStatus.SUCCESS,
            metrics=metrics,
            artifact_paths=[str(result_csv)],
        )

    def _compute_metrics(self, result_curve: pd.DataFrame) -> Optional[dict[str, float]]:
        res_load, res_unload = split_cycle(result_curve)
        if res_load is None or len(res_load) < 2:
            logger.error("CAE result has no loading segment")
            return None

        metrics: dict[str, float] = {}
        rmse_load = calculate_rmse(res_load, self._tgt_load)
        metrics["rmse_loading"] = rmse_load

        # Determine unloading RMSE
        if self._tgt_unload is not None and res_unload is not None:
            rmse_unload = calculate_rmse(res_unload, self._tgt_unload)
            metrics["rmse_unloading"] = rmse_unload
            metrics["rmse"] = (rmse_load + rmse_unload) / 2.0
        else:
            metrics["rmse"] = rmse_load

        # Feature-based metrics
        feat_cfg = self._obj_spec.features
        if feat_cfg:
            result_feats = extract_features(result_curve, feat_cfg)
            for name, target_val in self._target_features.items():
                actual = result_feats.get(name)
                if actual is not None and abs(target_val) > 1e-10:
                    metrics[f"{name}_error"] = abs(actual - target_val) / abs(target_val)
                else:
                    metrics[f"{name}_error"] = 1.0

        # Weighted score
        weights = self._obj_spec.weights
        ws = 0.0
        for key, val in metrics.items():
            if key in ("rmse_loading", "rmse_unloading"):
                w = weights.get(key, 0.0)
            else:
                base = key.replace("_error", "")
                w = weights.get(base, weights.get(key, 0.0))
            ws += w * val
        metrics["weighted_score"] = ws

        return metrics

    def _run_subprocess(self, job_name: str) -> Optional[Path]:
        self._stop_requested = False
        project_root = self._vexis_root.parent
        venv_py = project_root / ".venv" / "Scripts" / "python.exe"
        if not venv_py.exists():
            venv_py = project_root / ".venv" / "bin" / "python"
        python_cmd = str(venv_py) if venv_py.exists() else "python"

        cmd = [python_cmd, str(self._vexis_main)]
        logger.info("VEXIS run: %s", " ".join(cmd))

        error_detected = False
        try:
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            env = os.environ.copy()
            env.update({
                "QT_QPA_PLATFORM": "offscreen",
                "DISPLAY": "",
                "VTK_DEFAULT_OPENGL_WINDOW": "vtkOSOpenGLRenderWindow",
                "PYVISTA_OFF_SCREEN": "true",
            })

            proc = subprocess.Popen(
                cmd,
                cwd=str(self._vexis_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=creation_flags,
                env=env,
            )
            self._current_proc = proc

            timeout = self._cae_spec.timeout_sec
            if timeout and timeout > 0:
                def _watcher():
                    t0 = time.time()
                    while proc.poll() is None:
                        if self._stop_requested or time.time() - t0 > timeout:
                            self._terminate(proc)
                            return
                        time.sleep(1)
                threading.Thread(target=_watcher, daemon=True).start()

            for line in proc.stdout:
                if self._stop_requested:
                    break
                
                low = line.lower()
                if "error termination" in low or "fatal error" in low:
                    error_detected = True
                    logger.error("VEXIS error detected: %s", line.rstrip())
                
                # Always log INFO to show progress in console
                logger.info(line.rstrip())

            proc.wait()

            if self._stop_requested or proc.returncode != 0:
                return None

        except Exception as exc:
            logger.error("VEXIS execution error: %s", exc)
            return None
        finally:
            # Ensure process is killed if still running (e.g. on exception)
            if self._current_proc:
                self._terminate(self._current_proc)
            self._current_proc = None

        result_csv = self._vexis_results / f"{job_name}_result.csv"
        for _ in range(10):
            if result_csv.exists():
                break
            time.sleep(0.5)

        return result_csv if result_csv.exists() else None

    def _terminate(self, proc: subprocess.Popen, timeout: int = 10) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception as exc:
            logger.error("Process termination error: %s", exc)

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._current_proc:
            self._terminate(self._current_proc)
