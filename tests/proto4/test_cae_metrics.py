"""Tests for CAE metric computation with real CSV data.

Tests curve loading, column normalization, RMSE, feature extraction,
and cycle splitting — without running VEXIS.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from proto4_claude.cae_evaluator import (
    calculate_rmse,
    extract_features,
    extract_range,
    load_curve,
    split_cycle,
)


class TestLoadCurve:
    def test_standard_columns(self, target_csv: Path):
        df = load_curve(target_csv)
        assert "displacement" in df.columns
        assert "force" in df.columns
        assert len(df) > 0

    def test_vexis_column_names(self, result_csv_vexis_format: Path):
        """VEXIS uses Stroke/Reaction_Force — should be normalised."""
        df = load_curve(result_csv_vexis_format)
        assert "displacement" in df.columns
        assert "force" in df.columns

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_curve(tmp_path / "nonexistent.csv")


class TestExtractRange:
    def test_filters_within_range(self, target_curve: pd.DataFrame):
        trimmed = extract_range(target_curve, 0.1, 0.3)
        assert trimmed["displacement"].min() >= 0.1
        assert trimmed["displacement"].max() <= 0.3

    def test_empty_if_no_overlap(self, target_curve: pd.DataFrame):
        trimmed = extract_range(target_curve, 10.0, 20.0)
        assert len(trimmed) == 0


class TestSplitCycle:
    def test_with_unloading(self, target_curve: pd.DataFrame):
        load, unload = split_cycle(target_curve)
        assert load is not None
        assert unload is not None
        assert len(load) > 0
        assert len(unload) > 0
        # Loading should end at max displacement
        assert load["displacement"].iloc[-1] == target_curve["displacement"].max()

    def test_loading_only(self, target_curve_loading_only: pd.DataFrame):
        load, unload = split_cycle(target_curve_loading_only)
        assert load is not None
        assert unload is None

    def test_empty_dataframe(self):
        df = pd.DataFrame({"displacement": [], "force": []})
        load, unload = split_cycle(df)
        assert len(load) == 0
        assert unload is None


class TestCalculateRMSE:
    """RMSE tests use loading-only (monotonic) curves because interp1d
    requires monotonic x values."""

    @pytest.fixture
    def mono_target(self) -> pd.DataFrame:
        d = np.linspace(0.0, 0.5, 50)
        return pd.DataFrame({"displacement": d, "force": 100.0 * (d / 0.5) ** 1.5})

    @pytest.fixture
    def mono_good(self) -> pd.DataFrame:
        d = np.linspace(0.0, 0.5, 50)
        return pd.DataFrame({"displacement": d, "force": 102.0 * (d / 0.5) ** 1.5})

    @pytest.fixture
    def mono_bad(self) -> pd.DataFrame:
        d = np.linspace(0.0, 0.5, 50)
        return pd.DataFrame({"displacement": d, "force": 200.0 * (d / 0.5) ** 1.5})

    def test_identical_curves_zero_rmse(self, mono_target: pd.DataFrame):
        rmse = calculate_rmse(mono_target, mono_target)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_similar_curves_small_rmse(
        self, mono_target: pd.DataFrame, mono_good: pd.DataFrame,
    ):
        rmse = calculate_rmse(mono_good, mono_target)
        assert rmse < 10.0

    def test_different_curves_large_rmse(
        self, mono_target: pd.DataFrame, mono_good: pd.DataFrame, mono_bad: pd.DataFrame,
    ):
        rmse_bad = calculate_rmse(mono_bad, mono_target)
        rmse_good = calculate_rmse(mono_good, mono_target)
        assert rmse_bad > rmse_good

    def test_no_overlap_returns_inf(self):
        a = pd.DataFrame({"displacement": [0.0, 0.1], "force": [1.0, 2.0]})
        b = pd.DataFrame({"displacement": [10.0, 10.1], "force": [1.0, 2.0]})
        assert calculate_rmse(a, b) == float("inf")


class TestExtractFeatures:
    @pytest.fixture
    def simple_curve(self) -> pd.DataFrame:
        d = np.linspace(0.0, 1.0, 100)
        f = 50.0 * d ** 2 + 10.0 * np.sin(5 * d)
        return pd.DataFrame({"displacement": d, "force": f})

    def test_max(self, simple_curve: pd.DataFrame):
        cfg = {"peak": {"type": "max", "column": "force"}}
        feats = extract_features(simple_curve, cfg)
        assert "peak" in feats
        assert feats["peak"] == pytest.approx(simple_curve["force"].max())

    def test_min(self, simple_curve: pd.DataFrame):
        cfg = {"trough": {"type": "min", "column": "force"}}
        feats = extract_features(simple_curve, cfg)
        assert feats["trough"] == pytest.approx(simple_curve["force"].min())

    def test_slope(self, simple_curve: pd.DataFrame):
        cfg = {"stiffness": {"type": "slope", "column": "force", "range": [0.0, 0.3]}}
        feats = extract_features(simple_curve, cfg)
        assert "stiffness" in feats
        # Slope should be positive for this curve
        assert feats["stiffness"] > 0

    def test_peak_position(self, simple_curve: pd.DataFrame):
        cfg = {"pp": {"type": "peak_position", "column": "force"}}
        feats = extract_features(simple_curve, cfg)
        # Peak force is near displacement=1.0 for quadratic
        assert feats["pp"] > 0.5

    def test_value_at(self, simple_curve: pd.DataFrame):
        cfg = {"val": {"type": "value_at", "column": "force", "at": 0.5}}
        feats = extract_features(simple_curve, cfg)
        # f(0.5) ≈ 50*0.25 + 10*sin(2.5) ≈ 12.5 + 5.98 ≈ 18.5
        assert feats["val"] == pytest.approx(18.5, abs=1.0)

    def test_empty_config_returns_empty(self, simple_curve: pd.DataFrame):
        assert extract_features(simple_curve, {}) == {}
