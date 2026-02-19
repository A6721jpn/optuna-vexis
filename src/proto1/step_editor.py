"""
Proto1 STEP Editor Module

OpenCASCADE (pythonocc-core) を使用したSTEPファイルの寸法編集

Note:
    プロトタイプでは座標変換方式（スケーリング/移動）を採用
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

# OpenCASCADE imports (pythonocc-core)
try:
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Pnt, gp_Dir, gp_Ax1
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    HAS_OCC = True
except ImportError:
    HAS_OCC = False


logger = logging.getLogger(__name__)



class StepEditor:
    """
    STEPファイルの寸法を座標変換で編集するクラス
    
    プロトタイプでは以下の変換をサポート:
    - Z軸方向のスケーリング（高さ変更）
    - 壁厚の変更（内外面オフセット）
    """
    
    def __init__(self):
        if not HAS_OCC:
            raise ImportError(
                "pythonocc-core がインストールされていません。\n"
                "conda install -c conda-forge pythonocc-core でインストールしてください。"
            )
        self._shape: Optional[Any] = None
        self._original_bounds: Optional[tuple] = None
        self._source_path: Optional[Path] = None
    
    def load(self, step_path: str | Path) -> None:
        """
        STEPファイルを読込
        
        Args:
            step_path: 入力STEPファイルのパス
        """
        step_path = Path(step_path)
        if not step_path.exists():
            raise FileNotFoundError(f"STEPファイルが見つかりません: {step_path}")
        
        logger.info(f"STEPファイル読込: {step_path}")
        
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(step_path))
        
        if status != IFSelect_RetDone:
            raise ValueError(f"STEPファイルの読込に失敗: {step_path}")
        
        reader.TransferRoots()
        self._shape = reader.OneShape()
        self._source_path = step_path
        self._original_bounds = self._get_bounds(self._shape)
        
        logger.debug(f"元の境界ボックス: {self._original_bounds}")
    
    def _get_bounds(self, shape: Any) -> tuple:
        """形状の境界ボックスを取得"""
        box = Bnd_Box()
        brepbndlib.Add(shape, box)
        return box.Get()  # (xmin, ymin, zmin, xmax, ymax, zmax)
    
    def set_dimension(self, name: str, value: float, config: dict) -> None:
        """
        指定寸法の値を設定
        
        Args:
            name: 寸法名
            value: 新しい値
            config: 寸法設定（dimensions.yamlから）
        """
        if self._shape is None:
            raise RuntimeError("STEPファイルが読み込まれていません")
        
        method = config.get("step_reference", {}).get("method", "coordinate")
        
        if method == "coordinate":
            self._apply_coordinate_transform(name, value, config)
        elif method == "scale_z":
            self._apply_scale_z(value, config)
        elif method == "scale_uniform":
            self._apply_scale_uniform(value, config)
        else:
            logger.warning(f"未対応の変換方式: {method}")
    
    def _apply_coordinate_transform(self, name: str, value: float, config: dict) -> None:
        """座標変換を適用（高さ変更など）"""
        ref = config.get("step_reference", {})
        axis = ref.get("axis", "Z").upper()
        
        if self._original_bounds is None:
            return
        
        xmin, ymin, zmin, xmax, ymax, zmax = self._original_bounds
        
        # 軸に応じたスケール計算
        if axis == "Z":
            original_height = zmax - zmin
            if original_height <= 0:
                return
            scale_factor = value / original_height
            self._apply_axis_scale(scale_factor, axis="Z")
        elif axis == "X":
            original_width = xmax - xmin
            if original_width <= 0:
                return
            scale_factor = value / original_width
            self._apply_axis_scale(scale_factor, axis="X")
        elif axis == "Y":
            original_depth = ymax - ymin
            if original_depth <= 0:
                return
            scale_factor = value / original_depth
            self._apply_axis_scale(scale_factor, axis="Y")
        
        logger.info(f"寸法 '{name}' を {value} に設定（{axis}軸スケール）")
    
    def _apply_scale_z(self, target_height: float, config: dict) -> None:
        """Z軸方向のスケーリング"""
        if self._original_bounds is None:
            return
        
        _, _, zmin, _, _, zmax = self._original_bounds
        original_height = zmax - zmin
        
        if original_height <= 0:
            return
        
        scale_factor = target_height / original_height
        self._apply_axis_scale(scale_factor, axis="Z")
    
    def _apply_scale_uniform(self, scale_factor: float, config: dict) -> None:
        """全体の均一スケーリング"""
        trsf = gp_Trsf()
        trsf.SetScaleFactor(scale_factor)
        
        transform = BRepBuilderAPI_Transform(self._shape, trsf, True)
        transform.Build()
        
        if transform.IsDone():
            self._shape = transform.Shape()
            logger.debug(f"均一スケール適用: {scale_factor}")
    
    def _apply_axis_scale(self, scale_factor: float, axis: str = "Z") -> None:
        """
        特定軸方向のスケーリング
        
        Note:
            OpenCASCADEの標準変換では軸方向スケーリングが直接サポートされていないため、
            プロトタイプでは均一スケールで近似
        """
        # TODO: 軸方向スケーリングの正確な実装
        # 現在は均一スケールで近似（プロトタイプ用の簡易実装）
        trsf = gp_Trsf()
        trsf.SetScaleFactor(scale_factor)
        
        transform = BRepBuilderAPI_Transform(self._shape, trsf, True)
        transform.Build()
        
        if transform.IsDone():
            self._shape = transform.Shape()
            logger.debug(f"{axis}軸スケール適用: {scale_factor}")
    
    def apply_dimensions(self, dimensions: dict[str, float], dim_configs: list[dict]) -> None:
        """
        複数の寸法を一括適用
        
        Args:
            dimensions: {寸法名: 値} の辞書（スケールファクター、例: 1.0が変更なし）
            dim_configs: dimensions.yamlからの寸法設定リスト
        
        Note:
            プロトタイプでは全てのスケールを統合して1回の均一スケールとして適用。
            複数のスケールを個別に適用すると累積されてしまうため。
        """
        # 設定をname -> configの辞書に変換
        config_map = {cfg["name"]: cfg for cfg in dim_configs}
        
        # 全スケールファクターの積を計算
        combined_scale = 1.0
        scale_details = []
        
        for name, value in dimensions.items():
            if name in config_map:
                config = config_map[name]
                method = config.get("step_reference", {}).get("method", "scale_uniform")
                
                # プロトタイプでは全て均一スケールとして扱う
                # valueはスケールファクター（1.0が変更なし）
                combined_scale *= value
                scale_details.append(f"{name}={value:.4f}")
                logger.debug(f"スケール追加: {name} = {value:.4f}")
            else:
                logger.warning(f"寸法設定が見つかりません: {name}")
        
        # 最終スケールを1回だけ適用
        if abs(combined_scale - 1.0) > 1e-6:  # 変更がある場合のみ
            logger.info(f"統合スケール適用: {combined_scale:.4f} ({', '.join(scale_details)})")
            self._apply_scale_uniform(combined_scale, {})
    
    def save(self, output_path: str | Path) -> Path:
        """
        編集済みSTEPファイルを保存
        
        Args:
            output_path: 出力ファイルパス
        
        Returns:
            保存したファイルのPath
        """
        if self._shape is None:
            raise RuntimeError("保存する形状がありません")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"STEPファイル保存: {output_path}")
        
        writer = STEPControl_Writer()
        writer.Transfer(self._shape, STEPControl_AsIs)
        status = writer.Write(str(output_path))
        
        if status != IFSelect_RetDone:
            raise RuntimeError(f"STEPファイルの保存に失敗: {output_path}")
        
        return output_path
    
    def get_current_bounds(self) -> Optional[tuple]:
        """現在の形状の境界ボックスを取得"""
        if self._shape is None:
            return None
        return self._get_bounds(self._shape)


def edit_step_file(
    input_path: str | Path,
    output_path: str | Path,
    dimensions: dict[str, float],
    dim_configs: list[dict]
) -> Path:
    """
    STEPファイルを編集して保存する便利関数
    
    Args:
        input_path: 入力STEPファイル
        output_path: 出力STEPファイル
        dimensions: {寸法名: 値} の辞書
        dim_configs: 寸法設定リスト
    
    Returns:
        保存したファイルのPath
    """
    editor = StepEditor()
    editor.load(input_path)
    editor.apply_dimensions(dimensions, dim_configs)
    return editor.save(output_path)
