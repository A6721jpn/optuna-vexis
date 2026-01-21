"""
Proto2 Curve Processor Module

ターゲットカーブの読込・多項式近似・範囲抽出
- 実機データ（ノイズ含む）の処理
- CAE解析範囲への切り出し
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate


logger = logging.getLogger(__name__)


class CurveProcessor:
    """
    ターゲットカーブを処理するクラス
    
    実機データ（ノイズ含む）を多項式近似し、
    CAE解析範囲に切り出す
    """
    
    # ターゲットカーブのカラム名
    STROKE_COL = "Stroke"
    FORCE_COL = "Adjusted force"
    
    # 代替カラム名
    ALT_STROKE_COLS = ["stroke", "Displacement", "displacement", "x", "X"]
    ALT_FORCE_COLS = ["adjusted_force", "Force", "force", "Reaction_Force", "y", "Y"]
    
    def __init__(self):
        self._target_curve: Optional[pd.DataFrame] = None
        self._fitted_poly: Optional[np.poly1d] = None
    
    def load_target_curve(self, csv_path: str | Path) -> pd.DataFrame:
        """
        ターゲットカーブ読込
        
        Args:
            csv_path: CSVファイルパス
            
        Returns:
            DataFrame with columns: ['displacement', 'force']
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"ターゲットカーブが見つかりません: {csv_path}")
        
        logger.info(f"ターゲットカーブ読込: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # カラム名を検索
        stroke_col = self._find_column(df, [self.STROKE_COL] + self.ALT_STROKE_COLS)
        force_col = self._find_column(df, [self.FORCE_COL] + self.ALT_FORCE_COLS)
        
        if stroke_col is None or force_col is None:
            logger.warning(f"カラム名が不明。最初の2カラムを使用: {df.columns.tolist()}")
            if len(df.columns) >= 2:
                stroke_col = df.columns[0]
                force_col = df.columns[1]
            else:
                raise ValueError(f"CSVのカラム数が不足: {csv_path}")
        
        # 正規化
        result = pd.DataFrame({
            "displacement": pd.to_numeric(df[stroke_col], errors="coerce"),
            "force": pd.to_numeric(df[force_col], errors="coerce")
        })
        
        result = result.dropna().sort_values("displacement").reset_index(drop=True)
        
        logger.info(f"カーブ読込完了: {len(result)}点, 範囲=[{result['displacement'].min():.3f}, {result['displacement'].max():.3f}]")
        
        self._target_curve = result
        return result
    
    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """候補リストからカラムを探す"""
        df_cols_lower = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        return None
    
    def fit_polynomial(
        self,
        df: pd.DataFrame,
        degree: int = 5,
        stroke_range: Optional[tuple[float, float]] = None
    ) -> np.poly1d:
        """
        多項式近似（ノイズ除去）
        
        Args:
            df: Stroke, Force列を持つDataFrame
            degree: 多項式の次数
            stroke_range: 近似対象の範囲 (min, max)
        
        Returns:
            近似多項式
        """
        # 範囲抽出
        if stroke_range:
            mask = (df["displacement"] >= stroke_range[0]) & (df["displacement"] <= stroke_range[1])
            df_fit = df[mask]
        else:
            df_fit = df
        
        if len(df_fit) < degree + 1:
            raise ValueError(f"データ点数が不足しています: {len(df_fit)} < {degree + 1}")
        
        x = df_fit["displacement"].values
        y = df_fit["force"].values
        
        # 多項式フィッティング
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        
        # フィッティング品質
        y_pred = poly(x)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        logger.info(f"多項式近似完了: degree={degree}, RMSE={rmse:.6f}, R²={r2:.4f}")
        
        self._fitted_poly = poly
        return poly
    
    def get_fitted_curve(
        self,
        stroke_range: tuple[float, float],
        num_points: int = 100
    ) -> pd.DataFrame:
        """
        近似多項式からカーブを生成
        
        Args:
            stroke_range: (min, max)
            num_points: 生成点数
        
        Returns:
            近似カーブのDataFrame
        """
        if self._fitted_poly is None:
            raise ValueError("多項式近似が実行されていません。fit_polynomial()を先に呼んでください。")
        
        x = np.linspace(stroke_range[0], stroke_range[1], num_points)
        y = self._fitted_poly(x)
        
        return pd.DataFrame({
            "displacement": x,
            "force": y
        })
    
    def extract_range(
        self,
        df: pd.DataFrame,
        min_stroke: float,
        max_stroke: float
    ) -> pd.DataFrame:
        """
        CAE解析範囲のみ抽出
        
        Args:
            df: 入力カーブ
            min_stroke: 最小Stroke
            max_stroke: 最大Stroke
        
        Returns:
            範囲抽出後のDataFrame
        """
        mask = (df["displacement"] >= min_stroke) & (df["displacement"] <= max_stroke)
        result = df[mask].copy().reset_index(drop=True)
        
        logger.info(f"範囲抽出: [{min_stroke:.3f}, {max_stroke:.3f}] -> {len(result)}点")
        return result
    
    def resample_curve(
        self,
        df: pd.DataFrame,
        num_points: int = 100
    ) -> pd.DataFrame:
        """
        指定点数にリサンプリング
        
        Args:
            df: 入力カーブ
            num_points: 出力点数
        
        Returns:
            リサンプリング後のDataFrame
        """
        if len(df) < 2:
            raise ValueError("データ点数が不足しています")
        
        # 補間関数作成
        interp_func = interpolate.interp1d(
            df["displacement"].values,
            df["force"].values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        # 新しいグリッド
        x_new = np.linspace(
            df["displacement"].min(),
            df["displacement"].max(),
            num_points
        )
        y_new = interp_func(x_new)
        
        return pd.DataFrame({
            "displacement": x_new,
            "force": y_new
        })
    
    def process_target_curve(
        self,
        csv_path: str | Path,
        cae_stroke_range: tuple[float, float],
        use_polynomial: bool = False,
        polynomial_degree: int = 5,
        num_points: int = 100
    ) -> pd.DataFrame:
        """
        ターゲットカーブを一括処理
        
        Args:
            csv_path: CSVファイルパス
            cae_stroke_range: CAE解析範囲 (min, max)
            use_polynomial: 多項式近似を使用するか
            polynomial_degree: 多項式次数
            num_points: 出力点数
        
        Returns:
            処理済みのターゲットカーブ
        """
        # 読込
        df = self.load_target_curve(csv_path)
        
        if use_polynomial:
            # 多項式フィッティング
            self.fit_polynomial(df, polynomial_degree, cae_stroke_range)
            # 近似カーブを生成
            result = self.get_fitted_curve(cae_stroke_range, num_points)
        else:
            # 範囲抽出のみ
            result = self.extract_range(df, cae_stroke_range[0], cae_stroke_range[1])
            # リサンプリング
            if len(result) != num_points:
                result = self.resample_curve(result, num_points)
        
        logger.info(f"ターゲットカーブ処理完了: {len(result)}点")
        return result


def load_and_process_target(
    csv_path: str | Path,
    stroke_min: float,
    stroke_max: float,
    use_polynomial: bool = False
) -> pd.DataFrame:
    """ターゲットカーブを読込・処理する便利関数"""
    processor = CurveProcessor()
    return processor.process_target_curve(
        csv_path,
        (stroke_min, stroke_max),
        use_polynomial=use_polynomial
    )
