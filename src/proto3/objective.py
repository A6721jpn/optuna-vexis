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

from .curve_processor import CurveProcessor


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


class FatalOptimizationError(Exception):
    """最適化を継続不可能な致命的なエラー"""
    pass

class ObjectiveCalculator:
    """目的関数計算を管理するクラス"""
    
    def __init__(self, target_curve: pd.DataFrame, target_features: dict[str, float], config: dict):
        self.config = config
        self.history: list[dict] = []
        
        # ターゲットカーブの分割と前処理
        self.processor = CurveProcessor()
        
        # ターゲットカーブを行き/帰りに分割
        self.target_load, self.target_unload = self.processor.split_cycle(target_curve)
        
        if self.target_load is None or len(self.target_load) < 2:
            raise ValueError("ターゲットデータに行き（Loading）パートが含まれていません。")

        unload_len = len(self.target_unload) if self.target_unload is not None else 0
        logger.info(f"ターゲットカーブ分割: Loading={len(self.target_load)}点, Unloading={unload_len}点")
        
        self.target_features = target_features

    def evaluate(
        self,
        result_curve: pd.DataFrame,
        result_features: dict[str, float]
    ) -> dict[str, float]:
        """目的関数を評価"""
        
        # 結果カーブを分割
        res_load, res_unload = self.processor.split_cycle(result_curve)
        
        # --- ケース4: CAEに行きカーブがない ---
        if res_load is None or len(res_load) < 2:
            logger.error("重大なエラー: CAE結果に行き（Loading）カーブが存在しません。")
            raise FatalOptimizationError("CAE結果に行きカーブがありません。最適化を強制終了します。")
            
        objectives = {}
        
        # 行き（Loading）のRMSE計算
        rmse_load = calculate_rmse(res_load, self.target_load)
        objectives["rmse_loading"] = rmse_load
        
        rmse_unload = 0.0
        total_rmse = rmse_load # Default fallback
        
        # --- ケース判定と処理 ---
        
        # ケース1: 両方に行きと帰りがある
        if self.target_unload is not None and res_unload is not None:
            rmse_unload = calculate_rmse(res_unload, self.target_unload)
            objectives["rmse_unloading"] = rmse_unload
            
            # 総合RMSE (単純平均)
            total_rmse = (rmse_load + rmse_unload) / 2.0
            logger.debug(f"ケース1: Target(L+U), Result(L+U), RMSE_L={rmse_load:.6f}, RMSE_U={rmse_unload:.6f}")
            
        # ケース2: ターゲットは行きのみ、CAEは行きと帰り
        elif self.target_unload is None and res_unload is not None:
            logger.debug("ケース2: Target(L), Result(L+U) -> 行きのみ評価（CAEの帰りは無視）")
            total_rmse = rmse_load
            
        # ケース3: ターゲットは行きと帰り、CAEは行きのみ
        elif self.target_unload is not None and res_unload is None:
            logger.debug("ケース3: Target(L+U), Result(L) -> 行きのみ評価（ターゲットの帰りは無視）")
            total_rmse = rmse_load
            
        # ケース: 両方行きのみ
        else:
            logger.debug("ケースその他: Target(L), Result(L) -> 行きのみ評価")
            total_rmse = rmse_load

        objectives["rmse"] = total_rmse
        
        # 特徴量誤差
        feature_errors = calculate_feature_errors(result_features, self.target_features)
        objectives.update(feature_errors)
        
        # 重み付き合計スコア
        weights = self.config.get("weights", {"rmse": 1.0})
        weighted_score = 0.0
        
        for obj_name, obj_val in objectives.items():
            # rmse_loading/unloading は個別の重みがなければ無視（総合rmseに含まれるため）
            if obj_name in ["rmse_loading", "rmse_unloading"]:
                weight = weights.get(obj_name, 0.0)
            else:
                base_name = obj_name.replace("_error", "")
                weight = weights.get(base_name, weights.get(obj_name, 0.0))
                
            weighted_score += weight * obj_val
        
        objectives["weighted_score"] = weighted_score
        
        logger.info(f"目的関数値: RMSE(Total)={total_rmse:.6f} (Load={rmse_load:.6f}, Unload={rmse_unload:.6f})")
        
        self.history.append(objectives.copy())
        return objectives
    
    def get_best_trial(self, key: str = "rmse") -> Optional[dict]:
        """最良試行を取得"""
        if not self.history:
            return None
        
        # nanやinfを除外
        valid_history = [h for h in self.history if np.isfinite(h.get(key, float("inf")))]
        if not valid_history:
            return None
            
        return min(valid_history, key=lambda x: x.get(key, float("inf")))
    
    def get_convergence_status(self, threshold: float, key: str = "rmse") -> bool:
        """現在の収束状態を取得"""
        best = self.get_best_trial(key)
        if best is None:
            return False
        
        return best.get(key, float("inf")) <= threshold
