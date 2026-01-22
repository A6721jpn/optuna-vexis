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
        
        result = result.dropna().reset_index(drop=True)
        
        # 重複する変位点の平均化処理は削除（往復分離のため）
        # if result["displacement"].duplicated().any():
        #     logger.debug("重複する変位点を平均化します")
        #     result = result.groupby("displacement", as_index=False).mean()
        
        # 変位順ソートはしない（往復順序を保持するため）
        # result = result.sort_values("displacement").reset_index(drop=True)
        
        logger.info(f"カーブ読込完了: {len(result)}点, 範囲=[{result['displacement'].min():.3f}, {result['displacement'].max():.3f}]")
        
        self._target_curve = result
        return result

    def split_cycle(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        カーブを行き（Loading）と帰り（Unloading）に分割
        
        Args:
            df: 変位と荷重を含むDataFrame
        
        Returns:
            (loading_df, unloading_df): 行きと帰りのDataFrame
            帰りがない場合はunloading_dfはNone
        """
        if df.empty:
            return df, None
            
        # 最大変位のインデックスを取得
        # idxmax()は最初の最大値を返す
        max_idx = df["displacement"].idxmax()
        
        # 行き（Loading）: 最初から最大変位まで
        loading = df.iloc[:max_idx+1].copy().reset_index(drop=True)
        
        # 帰り（Unloading）: 最大変位から最後または変位0まで
        # データが往復を含み、かつ最大変位以降にデータがある場合
        unloading = None
        if max_idx < len(df) - 1:
            unloading = df.iloc[max_idx:].copy().reset_index(drop=True)
            
            # 帰りのデータが極端に少ない、または変位が減少していない場合はノイズとみなして除外も検討できるが
            # ここでは単純に存在有無で判定
            if len(unloading) < 2:
                unloading = None
        
        return loading, unloading
    
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
        
        # 単調増加（ヒステリシスなし）かチェック
        is_monotonic = df["displacement"].is_monotonic_increasing
        
        if is_monotonic:
            # 既存の単純リサンプリング
            interp_func = interpolate.interp1d(
                df["displacement"].values,
                df["force"].values,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
            
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
        else:
            # ヒステリシス（往復）あり -> 分離してリサンプリング
            logger.debug("ヒステリシス検知: 行きと帰りを分離してリサンプリングします")
            load, unload = self.split_cycle(df)
            
            if load is None:
                 raise ValueError("リサンプリング失敗: Loadingパートが見つかりません")

            # 点数を分配 (Loading/Unloadingの長さに応じて、または単純に分割)
            # ここでは各フェーズのデータ点数比率で配分
            total_len = len(df)
            n_load = max(2, int(num_points * len(load) / total_len))
            
            # Loadingリサンプリング
            # 行き：0 -> Max
            interp_load = interpolate.interp1d(
                load["displacement"].values,
                load["force"].values,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate"
            )
            x_load = np.linspace(
                load["displacement"].min(),
                load["displacement"].max(),
                n_load
            )
            y_load = interp_load(x_load)
            
            # Unloadingリサンプリング (存在する場合)
            if unload is not None and len(unload) > 1:
                n_unload = max(2, num_points - n_load)
                
                # 帰り：Max -> 0 (データ順序はそのまま使用)
                # interp1dはxがソートされている必要があるため、sortして補間関数作成
                # ただしUnloadingは変位が減少していくので、一度sortしてinterp作成し、
                # 評価点を逆順(Max -> 0)に生成する
                
                un_disp = unload["displacement"].values
                un_force = unload["force"].values
                
                # sort for interp1d
                idx_sort = np.argsort(un_disp)
                interp_unload = interpolate.interp1d(
                    un_disp[idx_sort],
                    un_force[idx_sort],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                # Evaluation points: Max -> Min (to preserve order)
                x_unload = np.linspace(
                    unload["displacement"].max(),
                    unload["displacement"].min(), # Descending
                    n_unload
                )
                y_unload = interp_unload(x_unload)
                
                # 接合 (x_loadの最後とx_unloadの最初は共にMax付近だが、念のためそのまま結合)
                x_final = np.concatenate([x_load, x_unload])
                y_final = np.concatenate([y_load, y_unload])
            else:
                x_final = x_load
                y_final = y_load
            
            return pd.DataFrame({
                "displacement": x_final,
                "force": y_final
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
