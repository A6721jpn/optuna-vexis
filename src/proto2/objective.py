"""
Proto2 Objective Module

最適化目的関数の計算
（Proto1から流用）
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate


logger = logging.getLogger(__name__)


def calculate_rmse(
    result: pd.DataFrame,
    target: pd.DataFrame,
    n_points: int = 100
) -> float:
    """
    2つのカーブ間のRMSEを計算
    
    補間により同一の変位点で比較
    """
    # 共通の変位範囲を決定
    disp_min = max(result["displacement"].min(), target["displacement"].min())
    disp_max = min(result["displacement"].max(), target["displacement"].max())
    
    if disp_min >= disp_max:
        logger.warning("カーブの変位範囲が重なっていません")
        return float("inf")
    
    # 共通グリッド作成
    common_disp = np.linspace(disp_min, disp_max, n_points)
    
    # 結果カーブを補間
    result_interp = interpolate.interp1d(
        result["displacement"].values,
        result["force"].values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    result_force = result_interp(common_disp)
    
    # ターゲットカーブを補間
    target_interp = interpolate.interp1d(
        target["displacement"].values,
        target["force"].values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate"
    )
    target_force = target_interp(common_disp)
    
    # RMSE計算
    rmse = np.sqrt(np.mean((result_force - target_force) ** 2))
    
    logger.debug(f"RMSE: {rmse:.6f}")
    return float(rmse)


def calculate_relative_error(actual: float, target: float) -> float:
    """相対誤差を計算"""
    if abs(target) < 1e-10:
        return abs(actual)
    return abs(actual - target) / abs(target)


def calculate_feature_errors(
    result_features: dict[str, float],
    target_features: dict[str, float]
) -> dict[str, float]:
    """特徴量の誤差を計算"""
    errors = {}
    
    for name, target_val in target_features.items():
        if name in result_features:
            actual_val = result_features[name]
            errors[f"{name}_error"] = calculate_relative_error(actual_val, target_val)
        else:
            logger.warning(f"特徴量が結果にありません: {name}")
            errors[f"{name}_error"] = 1.0
    
    return errors


def calculate_objectives(
    result: pd.DataFrame,
    target: pd.DataFrame,
    result_features: dict[str, float],
    target_features: dict[str, float],
    config: dict
) -> dict[str, float]:
    """多目的最適化用の目的関数値を計算"""
    objectives = {}
    
    # RMSE
    rmse = calculate_rmse(result, target)
    objectives["rmse"] = rmse
    
    # 特徴量誤差
    feature_errors = calculate_feature_errors(result_features, target_features)
    objectives.update(feature_errors)
    
    # 重み付き合計スコア
    weights = config.get("weights", {"rmse": 1.0})
    weighted_score = 0.0
    
    for obj_name, obj_val in objectives.items():
        base_name = obj_name.replace("_error", "")
        weight = weights.get(base_name, weights.get(obj_name, 0.0))
        weighted_score += weight * obj_val
    
    objectives["weighted_score"] = weighted_score
    
    logger.info(f"目的関数値: RMSE={rmse:.6f}, weighted_score={weighted_score:.6f}")
    
    return objectives


def is_converged(
    objectives: dict[str, float],
    threshold: float,
    target_key: str = "rmse"
) -> bool:
    """収束判定"""
    if target_key not in objectives:
        return False
    
    return objectives[target_key] <= threshold


class ObjectiveCalculator:
    """目的関数計算を管理するクラス"""
    
    def __init__(self, target_curve: pd.DataFrame, target_features: dict[str, float], config: dict):
        self.target_curve = target_curve
        self.target_features = target_features
        self.config = config
        self.history: list[dict] = []
    
    def evaluate(
        self,
        result_curve: pd.DataFrame,
        result_features: dict[str, float]
    ) -> dict[str, float]:
        """目的関数を評価"""
        objectives = calculate_objectives(
            result_curve,
            self.target_curve,
            result_features,
            self.target_features,
            self.config
        )
        
        self.history.append(objectives.copy())
        return objectives
    
    def get_best_trial(self, key: str = "rmse") -> Optional[dict]:
        """最良試行を取得"""
        if not self.history:
            return None
        
        return min(self.history, key=lambda x: x.get(key, float("inf")))
    
    def get_convergence_status(self, threshold: float, key: str = "rmse") -> bool:
        """現在の収束状態を取得"""
        if not self.history:
            return False
        
        best = self.get_best_trial(key)
        if best is None:
            return False
        
        return best.get(key, float("inf")) <= threshold
