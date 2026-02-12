"""Tests for click-ratio and first-peak feature extraction."""

from __future__ import annotations

import pandas as pd

from proto4_codex.cae_evaluator import extract_features


def test_extract_click_ratio_and_peak_force_features() -> None:
    curve = pd.DataFrame({
        "displacement": [0.0, 0.1, 0.2, 0.3, 0.4],
        "force": [0.0, 10.0, 5.0, 8.0, 7.0],
    })
    cfg = {
        "click_ratio": {"type": "click_ratio", "column": "force"},
        "peak_force": {"type": "peak_force", "column": "force"},
        "next_bottom_force": {"type": "next_bottom_force", "column": "force"},
    }

    feats = extract_features(curve, cfg)
    assert abs(feats["peak_force"] - 10.0) < 1e-9
    assert abs(feats["next_bottom_force"] - 5.0) < 1e-9
    assert abs(feats["click_ratio"] - 0.5) < 1e-9

