"""
Proto3 Curve Processor Module
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate


logger = logging.getLogger(__name__)


class CurveProcessor:
    """Target curve processing utilities."""

    STROKE_COL = "Stroke"
    FORCE_COL = "Adjusted force"

    ALT_STROKE_COLS = ["stroke", "Displacement", "displacement", "x", "X"]
    ALT_FORCE_COLS = ["adjusted_force", "Force", "force", "Reaction_Force", "y", "Y"]

    def __init__(self) -> None:
        self._target_curve: Optional[pd.DataFrame] = None
        self._fitted_poly: Optional[np.poly1d] = None

    def load_target_curve(self, csv_path: str | Path) -> pd.DataFrame:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Target curve not found: {csv_path}")

        df = pd.read_csv(csv_path)

        stroke_col = self._find_column(df, [self.STROKE_COL] + self.ALT_STROKE_COLS)
        force_col = self._find_column(df, [self.FORCE_COL] + self.ALT_FORCE_COLS)

        if stroke_col is None or force_col is None:
            if len(df.columns) >= 2:
                stroke_col = df.columns[0]
                force_col = df.columns[1]
            else:
                raise ValueError(f"CSV columns insufficient: {csv_path}")

        result = pd.DataFrame({
            "displacement": pd.to_numeric(df[stroke_col], errors="coerce"),
            "force": pd.to_numeric(df[force_col], errors="coerce"),
        }).dropna().reset_index(drop=True)

        self._target_curve = result
        return result

    def split_cycle(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        df_cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        return None

    def extract_range(self, df: pd.DataFrame, min_stroke: float, max_stroke: float) -> pd.DataFrame:
        mask = (df["displacement"] >= min_stroke) & (df["displacement"] <= max_stroke)
        return df[mask].copy().reset_index(drop=True)

    def resample_curve(self, df: pd.DataFrame, num_points: int = 100) -> pd.DataFrame:
        if len(df) < 2:
            raise ValueError("Not enough points to resample")

        is_monotonic = df["displacement"].is_monotonic_increasing
        if is_monotonic:
            interp_func = interpolate.interp1d(
                df["displacement"].values,
                df["force"].values,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            x_new = np.linspace(df["displacement"].min(), df["displacement"].max(), num_points)
            y_new = interp_func(x_new)
            return pd.DataFrame({"displacement": x_new, "force": y_new})

        load, unload = self.split_cycle(df)
        if load is None:
            raise ValueError("Loading segment not found")

        total_len = len(df)
        n_load = max(2, int(num_points * len(load) / total_len))

        interp_load = interpolate.interp1d(
            load["displacement"].values,
            load["force"].values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        x_load = np.linspace(load["displacement"].min(), load["displacement"].max(), n_load)
        y_load = interp_load(x_load)

        if unload is not None and len(unload) > 1:
            n_unload = max(2, num_points - n_load)
            un_disp = unload["displacement"].values
            un_force = unload["force"].values
            idx_sort = np.argsort(un_disp)
            interp_unload = interpolate.interp1d(
                un_disp[idx_sort],
                un_force[idx_sort],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            x_unload = np.linspace(unload["displacement"].max(), unload["displacement"].min(), n_unload)
            y_unload = interp_unload(x_unload)
            x_final = np.concatenate([x_load, x_unload])
            y_final = np.concatenate([y_load, y_unload])
        else:
            x_final = x_load
            y_final = y_load

        return pd.DataFrame({"displacement": x_final, "force": y_final})

    def process_target_curve(
        self,
        csv_path: str | Path,
        cae_stroke_range: tuple[float, float],
        use_polynomial: bool = False,
        polynomial_degree: int = 5,
        num_points: int = 100,
    ) -> pd.DataFrame:
        df = self.load_target_curve(csv_path)
        result = self.extract_range(df, cae_stroke_range[0], cae_stroke_range[1])
        if len(result) != num_points:
            result = self.resample_curve(result, num_points)
        return result


def load_and_process_target(
    csv_path: str | Path,
    stroke_min: float,
    stroke_max: float,
    use_polynomial: bool = False,
) -> pd.DataFrame:
    processor = CurveProcessor()
    return processor.process_target_curve(
        csv_path,
        (stroke_min, stroke_max),
        use_polynomial=use_polynomial,
    )
