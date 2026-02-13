"""
v1.0 CAE Evaluator

Wraps VEXIS subprocess execution, result CSV loading, curve processing,
and metric extraction into a single coherent unit.

Consolidates proto3 vexis_runner + result_loader + curve_processor.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
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
_SOLVER_PROGRESS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _extract_solver_progress(line: str) -> Optional[float]:
    if "solver time" not in line.lower():
        return None
    matches = _SOLVER_PROGRESS_RE.findall(line)
    if not matches:
        return None
    cur_text, total_text = matches[-1]
    try:
        cur = float(cur_text)
        total = float(total_text)
    except ValueError:
        return None
    if total <= 0:
        return None
    ratio = cur / total
    return min(1.0, max(0.0, ratio))


def _detect_solver_error_marker(line: str, markers: tuple[str, ...]) -> Optional[str]:
    low = line.lower()
    for marker in markers:
        if marker and marker in low:
            return marker
    return None


def _first_peak_and_next_bottom(
    df: pd.DataFrame,
    column: str,
    *,
    prominence: Optional[float] = None,
) -> Optional[tuple[float, float]]:
    """Find the first local maximum and the next local minimum after it.

    Returns:
        (first_peak_force, next_bottom_force) or None when input is invalid.
    """
    if column not in df.columns or len(df) < 2:
        return None

    y = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if len(y) < 2:
        return None

    y_range = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(y_range):
        return None
    if prominence is None:
        prominence = max(1e-10, y_range * 0.02)

    peaks, _ = sp_signal.find_peaks(y, prominence=prominence)
    if len(peaks):
        peak_idx = int(peaks[0])
    else:
        peak_idx = int(np.argmax(y))

    if peak_idx >= len(y) - 1:
        return float(y[peak_idx]), float(y[peak_idx])

    tail = y[peak_idx + 1:]
    if len(tail) == 0:
        return float(y[peak_idx]), float(y[peak_idx])

    bottoms, _ = sp_signal.find_peaks(-tail, prominence=max(1e-10, prominence * 0.5))
    if len(bottoms):
        bottom_idx = peak_idx + 1 + int(bottoms[0])
    else:
        bottom_idx = peak_idx + 1 + int(np.argmin(tail))

    return float(y[peak_idx]), float(y[bottom_idx])


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
        elif feat_type in ("peak_force", "first_peak_force"):
            peak_bottom = _first_peak_and_next_bottom(
                df,
                column,
                prominence=cfg.get("prominence"),
            )
            features[name] = peak_bottom[0] if peak_bottom else float(df[column].max())
        elif feat_type in ("next_bottom_force", "bottom_force"):
            peak_bottom = _first_peak_and_next_bottom(
                df,
                column,
                prominence=cfg.get("prominence"),
            )
            features[name] = peak_bottom[1] if peak_bottom else float(df[column].min())
        elif feat_type == "click_ratio":
            peak_bottom = _first_peak_and_next_bottom(
                df,
                column,
                prominence=cfg.get("prominence"),
            )
            if peak_bottom is None:
                features[name] = 0.0
                continue
            peak_force, bottom_force = peak_bottom
            denom = abs(peak_force)
            features[name] = (peak_force - bottom_force) / denom if denom > 1e-10 else 0.0

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
        self._stdout_logger = None
        self._stdout_log_dir: Optional[Path] = None
        self._last_failure_reason: Optional[str] = None

        if not self._vexis_root.exists():
            raise FileNotFoundError(f"VEXIS not found: {self._vexis_root}")

        self._vexis_input = self._vexis_root / "input"
        self._vexis_results = self._vexis_root / "results"
        self._vexis_main = self._vexis_root / "main.py"
        self._step_stash_root = self._vexis_input / ".optuna_step_stash"

        if self._cae_spec.stdout_log_dir:
            self._stdout_log_dir = Path(self._cae_spec.stdout_log_dir)
            self._stdout_log_dir.mkdir(parents=True, exist_ok=True)

        # Pre-split target curve
        self._tgt_load, self._tgt_unload = split_cycle(target_curve)
        if self._tgt_load is None or len(self._tgt_load) < 2:
            raise ValueError("Target curve has no valid loading segment")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, step_path: Path, point: DesignPoint) -> CaeResult:
        """Run full CAE evaluation pipeline with retry."""
        job_name = f"v1_0_trial_{point.trial_id}"
        retries = max(1, self._cae_spec.max_retries)
        result = CaeResult(status=CaeStatus.FAIL, failure_reason="cae_not_started")

        for attempt in range(1, retries + 1):
            t0 = time.time()
            result = self._single_run(step_path, job_name)
            elapsed = time.time() - t0
            result.runtime_sec = elapsed

            if result.status == CaeStatus.SUCCESS:
                return result

            if attempt < retries:
                logger.warning(
                    "CAE attempt %d/%d failed (%s); retrying",
                    attempt,
                    retries,
                    result.failure_reason or "unknown",
                )

        return result  # last failed attempt

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _single_run(self, step_path: Path, job_name: str) -> CaeResult:
        """Execute one VEXIS run and compute metrics."""
        self._last_failure_reason = None
        target_step: Optional[Path] = None
        stash_dir: Optional[Path] = None

        try:
            # Keep only the current trial STEP in vexis/input so VEXIS processes
            # exactly one STEP per trial without changing VEXIS code.
            target_step, stash_dir = self._prepare_single_step_input(step_path, job_name)

            # Ensure stale CSV from a previous run cannot be reused as success.
            stale_csv = self._vexis_results / f"{job_name}_result.csv"
            if stale_csv.exists():
                try:
                    stale_csv.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove stale result CSV %s: %s", stale_csv, exc)

            # Run VEXIS subprocess
            result_csv = self._run_subprocess(job_name)
            if result_csv is None:
                return CaeResult(
                    status=CaeStatus.FAIL,
                    failure_reason=self._last_failure_reason or "vexis_subprocess_failed",
                )

            # Load and process result
            try:
                result_curve = load_curve(result_csv)
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Failed to load result: %s", exc)
                return CaeResult(
                    status=CaeStatus.FAIL,
                    failure_reason=f"result_load_failed:{exc.__class__.__name__}",
                )

            trimmed = extract_range(
                result_curve,
                self._cae_spec.stroke_range_min,
                self._cae_spec.stroke_range_max,
            )

            # Compute metrics
            metrics = self._compute_metrics(trimmed)
            if metrics is None:
                return CaeResult(
                    status=CaeStatus.FAIL,
                    failure_reason="metric_computation_failed",
                )

            return CaeResult(
                status=CaeStatus.SUCCESS,
                metrics=metrics,
                artifact_paths=[str(result_csv)],
            )
        finally:
            if target_step is not None and target_step.exists():
                try:
                    target_step.unlink()
                except OSError as exc:
                    logger.warning("Failed to remove trial STEP %s: %s", target_step, exc)
            if stash_dir is not None:
                self._restore_stashed_steps(stash_dir)

    def _prepare_single_step_input(self, step_path: Path, job_name: str) -> tuple[Path, Path]:
        """Move existing STEP files aside and place only the current trial STEP."""
        self._vexis_input.mkdir(parents=True, exist_ok=True)
        self._recover_orphaned_step_stash()

        stash_dir = self._step_stash_root / f"{job_name}_{int(time.time() * 1000)}"
        stash_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        try:
            for existing_step in sorted(self._vexis_input.glob("*.step")):
                existing_step.replace(stash_dir / existing_step.name)
                moved += 1

            target_step = self._vexis_input / f"{job_name}.step"
            if target_step.exists():
                target_step.unlink()
            shutil.copy2(step_path, target_step)
        except Exception:
            self._restore_stashed_steps(stash_dir)
            raise

        logger.info(
            "Isolated VEXIS input for %s: moved %d STEP(s), active=%s",
            job_name,
            moved,
            target_step.name,
        )
        return target_step, stash_dir

    def _recover_orphaned_step_stash(self) -> None:
        """Restore STEP files from a previous interrupted run, if any."""
        if not self._step_stash_root.exists():
            return
        for stash_dir in sorted(p for p in self._step_stash_root.iterdir() if p.is_dir()):
            logger.warning("Recovering orphaned STEP stash: %s", stash_dir)
            self._restore_stashed_steps(stash_dir)

    def _restore_stashed_steps(self, stash_dir: Path) -> None:
        """Move stashed STEP files back into vexis/input."""
        if not stash_dir.exists():
            return

        for stashed_step in sorted(stash_dir.glob("*.step")):
            dest = self._vexis_input / stashed_step.name
            if dest.exists():
                conflict_dest = self._vexis_input / (
                    f"{stashed_step.stem}.restored_conflict_{int(time.time() * 1000)}{stashed_step.suffix}"
                )
                logger.warning(
                    "STEP restore conflict: %s exists, restored as %s",
                    dest.name,
                    conflict_dest.name,
                )
                stashed_step.replace(conflict_dest)
            else:
                stashed_step.replace(dest)

        try:
            stash_dir.rmdir()
        except OSError:
            pass

        if self._step_stash_root.exists():
            try:
                self._step_stash_root.rmdir()
            except OSError:
                pass

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
            for name, actual in result_feats.items():
                metrics[name] = float(actual)
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
            elif key.endswith("_error"):
                base = key.replace("_error", "")
                w = weights.get(base, weights.get(key, 0.0))
            else:
                # Raw feature metrics are not part of the legacy weighted objective
                # unless explicitly configured by exact key.
                w = weights.get(key, 0.0)
            ws += w * val
        metrics["weighted_score"] = ws

        return metrics

    def _run_subprocess(self, job_name: str) -> Optional[Path]:
        self._stop_requested = False
        self._last_failure_reason = None
        project_root = self._vexis_root.parent
        venv_py = project_root / ".venv" / "Scripts" / "python.exe"
        if not venv_py.exists():
            venv_py = project_root / ".venv" / "bin" / "python"
        python_cmd = str(venv_py) if venv_py.exists() else "python"

        cmd = [python_cmd, str(self._vexis_main)]
        logger.info("VEXIS run: %s", " ".join(cmd))

        error_markers = tuple(m.strip().lower() for m in self._cae_spec.solver_error_markers if m.strip())
        solver_marker_reason: Optional[str] = None
        solver_stalled_reason: Optional[str] = None
        stdout_fh = None
        try:
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            env = os.environ.copy()
            env.update({
                "QT_QPA_PLATFORM": "offscreen",
                "DISPLAY": "",
                "VTK_DEFAULT_OPENGL_WINDOW": "vtkOSOpenGLRenderWindow",
                "PYVISTA_OFF_SCREEN": "true",
                "VEXIS_NONINTERACTIVE": "1",
            })

            if self._stdout_log_dir:
                self._stdout_log_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = self._stdout_log_dir / f"{job_name}_vexis.log"
                stdout_fh = open(stdout_path, "w", encoding="utf-8")

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

            stall_sec = max(1, int(self._cae_spec.solver_progress_stall_sec))
            poll_sec = max(0.1, float(self._cae_spec.solver_log_poll_sec))
            start_ts = time.time()
            solver_log_seen = False
            last_solver_progress: Optional[float] = None
            last_solver_progress_ts = start_ts
            line_queue: queue.Queue[Optional[str]] = queue.Queue()

            def _reader() -> None:
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        line_queue.put(line.rstrip("\n"))
                finally:
                    line_queue.put(None)

            threading.Thread(target=_reader, daemon=True).start()
            poll_token = object()
            reader_done = False

            while True:
                if self._stop_requested:
                    self._terminate(proc)
                    break

                now = time.time()
                try:
                    line_item = line_queue.get(timeout=poll_sec)
                except queue.Empty:
                    line_item = poll_token

                if line_item is None:
                    reader_done = True
                elif line_item is not poll_token:
                    msg = str(line_item)
                    low = msg.lower()
                    matched_marker = _detect_solver_error_marker(low, error_markers)
                    if matched_marker and solver_marker_reason is None:
                        solver_marker_reason = self._format_solver_error_reason(matched_marker)
                        logger.warning(
                            "VEXIS solver error marker detected (%s): %s",
                            matched_marker,
                            msg.rstrip(),
                        )

                    progress = _extract_solver_progress(msg)
                    if progress is not None:
                        solver_log_seen = True
                        if last_solver_progress is None or progress > last_solver_progress + 1e-8:
                            last_solver_progress = progress
                            last_solver_progress_ts = now

                    logger.debug(msg)
                    if stdout_fh:
                        stdout_fh.write(msg + "\n")
                    if self._cae_spec.stream_stdout:
                        level_name = self._cae_spec.stdout_console_level.upper()
                        level = getattr(logging, level_name, logging.INFO)
                        if logger.isEnabledFor(level):
                            logger.log(level, "VEXIS: %s", msg)

                    if matched_marker and proc.poll() is None:
                        self._terminate(proc)
                        break

                if proc.poll() is None and not solver_log_seen and (now - start_ts) > stall_sec:
                    solver_stalled_reason = self._format_solver_start_reason(stall_sec)
                    logger.warning(
                        "VEXIS solver log did not start (%s); terminating process",
                        solver_stalled_reason,
                    )
                    self._terminate(proc)
                    break

                progress_age = now - last_solver_progress_ts
                if (
                    proc.poll() is None
                    and solver_log_seen
                    and self._is_solver_progress_stalled(
                        last_solver_progress,
                        progress_age,
                        stall_sec,
                    )
                ):
                    solver_stalled_reason = self._format_solver_progress_stall_reason(
                        last_solver_progress,
                        stall_sec,
                    )
                    logger.warning(
                        "VEXIS solver progress stalled (%s); terminating process",
                        solver_stalled_reason,
                    )
                    self._terminate(proc)
                    break

                if proc.poll() is not None and reader_done and line_queue.empty():
                    break

            proc.wait()

            if self._stop_requested:
                self._last_failure_reason = "stop_requested"
                return None
            if solver_marker_reason:
                self._last_failure_reason = solver_marker_reason
                return None
            if solver_stalled_reason:
                self._last_failure_reason = solver_stalled_reason
                return None
            if proc.returncode != 0:
                self._last_failure_reason = f"process_exit_{proc.returncode}"
                return None

        except Exception as exc:
            logger.error("VEXIS execution error: %s", exc)
            self._last_failure_reason = f"execution_error:{exc.__class__.__name__}"
            return None
        finally:
            if stdout_fh:
                stdout_fh.close()
            self._current_proc = None

        result_csv = self._vexis_results / f"{job_name}_result.csv"
        for _ in range(10):
            if result_csv.exists():
                break
            time.sleep(0.5)

        if result_csv.exists():
            return result_csv

        self._last_failure_reason = "result_csv_missing"
        return None

    @staticmethod
    def _format_solver_start_reason(stall_sec: int) -> str:
        return f"solver_log_not_started_for_{stall_sec}s"

    @staticmethod
    def _format_solver_progress_stall_reason(progress: Optional[float], stall_sec: int) -> str:
        if progress is None:
            return f"solver_log_stalled_for_{stall_sec}s"
        return f"solver_progress_stalled_at_{progress * 100.0:.1f}pct_for_{stall_sec}s"

    @staticmethod
    def _format_solver_error_reason(marker: str) -> str:
        normalized = marker.strip().lower().replace(" ", "_")
        return f"solver_error_marker:{normalized}"

    @staticmethod
    def _is_solver_progress_stalled(
        progress: Optional[float],
        elapsed_since_progress_sec: float,
        stall_sec: int,
    ) -> bool:
        if elapsed_since_progress_sec <= stall_sec:
            return False
        if progress is not None and progress >= 1.0 - 1e-6:
            # After reported 100%, post-processing can continue without solver ticks.
            return False
        return True

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
