"""
Proto2 Result Loader Module

CAE解析結果CSVの読込と特徴量抽出
（Proto1から流用）
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate, signal


logger = logging.getLogger(__name__)


class ResultLoader:
    """
    VEXISの解析結果CSVを読込・解析するクラス
    """
    
    # VEXISの結果CSVカラム名
    DISPLACEMENT_COL = "Stroke"
    FORCE_COL = "Reaction_Force"
    
    # 代替カラム名
    ALT_DISPLACEMENT_COLS = ["stroke", "Displacement", "displacement", "x", "X"]
    ALT_FORCE_COLS = ["reaction_force", "Force", "force", "Reaction", "reaction", "y", "Y", "Fz", "fz"]
    
    def __init__(self):
        self._cache: dict[str, pd.DataFrame] = {}
    
    def load_curve(self, csv_path: str | Path) -> pd.DataFrame:
        """
        反力-変位カーブを読込
        
        Args:
            csv_path: CSVファイルのパス
        
        Returns:
            DataFrame with columns: ['displacement', 'force']
        """
        csv_path = Path(csv_path)
        
        # キャッシュチェック（Proto2では毎回新しい結果なのでスキップ可）
        # cache_key = str(csv_path)
        # if cache_key in self._cache:
        #     return self._cache[cache_key].copy()
        
        if not csv_path.exists():
            raise FileNotFoundError(f"結果CSVが見つかりません: {csv_path}")
        
        logger.info(f"結果CSV読込: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # カラム名の正規化
        disp_col = self._find_column(df, [self.DISPLACEMENT_COL] + self.ALT_DISPLACEMENT_COLS)
        force_col = self._find_column(df, [self.FORCE_COL] + self.ALT_FORCE_COLS)
        
        if disp_col is None or force_col is None:
            logger.warning(f"カラム名が不明。最初の2カラムを使用: {df.columns.tolist()}")
            if len(df.columns) >= 2:
                disp_col = df.columns[0]
                force_col = df.columns[1]
            else:
                raise ValueError(f"CSVのカラム数が不足: {csv_path}")
        
        result = pd.DataFrame({
            "displacement": pd.to_numeric(df[disp_col], errors="coerce"),
            "force": pd.to_numeric(df[force_col], errors="coerce")
        })
        
        result = result.dropna().reset_index(drop=True)
        
        # 重複平均化とソートを削除（往復分離のため）
        # result = result.sort_values("displacement").reset_index(drop=True)
        
        logger.debug(f"データ点数: {len(result)}")
        
        return result
    
    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """候補リストからカラムを探す（大文字小文字無視）"""
        df_cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        return None
    
    def load_target(self, csv_path: str | Path) -> pd.DataFrame:
        """ターゲットカーブを読込（load_curveと同じ形式）"""
        return self.load_curve(csv_path)
    
    def interpolate_to_common_grid(
        self,
        df: pd.DataFrame,
        target_displacements: np.ndarray
    ) -> pd.DataFrame:
        """
        指定した変位点に補間
        """
        interp_func = interpolate.interp1d(
            df["displacement"].values,
            df["force"].values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        interpolated_force = interp_func(target_displacements)
        
        return pd.DataFrame({
            "displacement": target_displacements,
            "force": interpolated_force
        })
    
    def extract_features(self, df: pd.DataFrame, config: dict) -> dict[str, float]:
        """
        カーブから特徴量を抽出
        """
        features = {}
        
        for name, feat_config in config.items():
            feat_type = feat_config.get("type", "max")
            column = feat_config.get("column", "force")
            
            if column not in df.columns:
                logger.warning(f"カラムが見つかりません: {column}")
                continue
            
            if feat_type == "max":
                features[name] = float(df[column].max())
            
            elif feat_type == "min":
                features[name] = float(df[column].min())
            
            elif feat_type == "mean":
                features[name] = float(df[column].mean())
            
            elif feat_type == "slope":
                range_vals = feat_config.get("range", [0.0, 0.5])
                mask = (df["displacement"] >= range_vals[0]) & (df["displacement"] <= range_vals[1])
                subset = df[mask]
                
                if len(subset) >= 2:
                    x = subset["displacement"].values
                    y = subset[column].values
                    slope, _ = np.polyfit(x, y, 1)
                    features[name] = float(slope)
                else:
                    features[name] = 0.0
            
            elif feat_type == "peak_position":
                idx = df[column].idxmax()
                features[name] = float(df.loc[idx, "displacement"])
            
            elif feat_type == "value_at":
                target_disp = feat_config.get("at", 0.5)
                interp_func = interpolate.interp1d(
                    df["displacement"].values,
                    df[column].values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                features[name] = float(interp_func(target_disp))
            
            elif feat_type == "local_max":
                # 極大値（ピーク）検出
                y = df[column].values
                # ノイズ対策: 突出度(prominence)の閾値を設定可能に
                prominence = feat_config.get("prominence", 0.05 * (y.max() - y.min()))
                peaks, properties = signal.find_peaks(y, prominence=prominence)
                
                if len(peaks) > 0:
                    # 複数のピークがある場合は最大のものを使用（設定で変更可能にしても良い）
                    # features[name] = float(y[peaks].max())
                    
                    # 最初に見つかった有意なピーク、または最大のピークを選択
                    # ゴム特性の場合は「降伏点」的な最初のピークが重要な場合もあるが
                    # ここでは最大のピークを採用する（最もロバスト）
                    features[name] = float(y[peaks].max())
                else:
                    logger.warning(f"ピークが見つかりませんでした: {name} (prominence={prominence:.4f})")
                    features[name] = float(y.max()) # 代替として単純最大値
            
            else:
                logger.warning(f"未対応の特徴量タイプ: {feat_type}")
        
        logger.debug(f"抽出された特徴量: {features}")
        return features
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()


def load_result_curve(csv_path: str | Path) -> pd.DataFrame:
    """結果カーブを読込む便利関数"""
    loader = ResultLoader()
    return loader.load_curve(csv_path)
