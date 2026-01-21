"""
Proto2 Material Editor Module

OGDEN材料モデル係数の編集とVEXIS設定更新
- material.yamlへの試行材料追記
- config.yamlのmaterial_name更新
"""

import logging
from pathlib import Path
from typing import Optional
import copy

import yaml


logger = logging.getLogger(__name__)


class MaterialEditor:
    """
    OGDEN材料モデル係数を編集するクラス
    
    VEXISのmaterial.yamlに試行用材料を追記し、
    config.yamlのmaterial_name参照を更新する
    """
    
    def __init__(
        self,
        material_yaml_path: str | Path,
        config_yaml_path: str | Path
    ):
        """
        Args:
            material_yaml_path: vexis/config/material.yamlのパス
            config_yaml_path: vexis/config/config.yamlのパス
        """
        self.material_yaml_path = Path(material_yaml_path)
        self.config_yaml_path = Path(config_yaml_path)
        
        if not self.material_yaml_path.exists():
            raise FileNotFoundError(f"material.yamlが見つかりません: {self.material_yaml_path}")
        if not self.config_yaml_path.exists():
            raise FileNotFoundError(f"config.yamlが見つかりません: {self.config_yaml_path}")
        
        # 元のファイル内容をバックアップ
        self._original_material = self._load_yaml(self.material_yaml_path)
        self._original_config = self._load_yaml(self.config_yaml_path)
        
        # 追加した試行材料名のリスト
        self._added_materials: list[str] = []
        
        logger.info(f"MaterialEditor初期化: {self.material_yaml_path}")
    
    def _load_yaml(self, path: Path) -> dict:
        """YAMLファイルを読込"""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _save_yaml(self, data: dict, path: Path) -> None:
        """YAMLファイルに保存（コメント保持なし）"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def load_base_material(self, material_name: str) -> dict:
        """
        基準材料の係数を読込
        
        Args:
            material_name: 材料名（例: "Ogden_Rubber_v1"）
        
        Returns:
            材料パラメータの辞書
        """
        materials = self._original_material.get("materials", {})
        
        if material_name not in materials:
            raise ValueError(f"材料が見つかりません: {material_name}")
        
        material = materials[material_name]
        logger.info(f"基準材料読込: {material_name}")
        
        return copy.deepcopy(material)
    
    def get_ogden_params(self, material_name: str) -> dict:
        """
        OGDEN係数のみ抽出
        
        Returns:
            {
                "c": [...], "m": [...],
                "t": [...], "g": [...]
            }
        """
        material = self.load_base_material(material_name)
        params = material.get("parameters", {})
        
        elastic = params.get("elastic", {})
        visco = params.get("visco", {})
        
        return {
            "c": list(elastic.get("c", [])),
            "m": list(elastic.get("m", [])),
            "t": list(visco.get("t", [])),
            "g": list(visco.get("g", []))
        }
    
    def calculate_bounds(
        self,
        base_params: dict,
        range_percent: float,
        min_nonzero: float = 0.001,
        mode: str = "all"
    ) -> dict:
        """
        係数の上下限を計算
        
        Args:
            base_params: 基準パラメータ {"c": [...], "m": [...], ...}
            range_percent: 変動範囲（%）例: 60 -> ±60%
            min_nonzero: 0でない係数の最小値
            mode: "elastic_only" | "visco_only" | "all"
        
        Returns:
            {
                "c": [(min, max), (min, max), ...],
                "m": [(min, max), ...],
                "t": [(min, max), ...],
                "g": [(min, max), ...]
            }
        """
        bounds = {}
        ratio_low = 1.0 - (range_percent / 100.0)
        ratio_high = 1.0 + (range_percent / 100.0)
        
        # 対象係数を決定
        if mode == "elastic_only":
            target_keys = ["c", "m"]
        elif mode == "visco_only":
            target_keys = ["t", "g"]
        else:  # all
            target_keys = ["c", "m", "t", "g"]
        
        for key in ["c", "m", "t", "g"]:
            values = base_params.get(key, [])
            key_bounds = []
            
            for val in values:
                if key in target_keys and abs(val) > 1e-10:
                    # 範囲計算
                    low = val * ratio_low
                    high = val * ratio_high
                    
                    # 順序保証（負の値の場合）
                    if low > high:
                        low, high = high, low
                    
                    # 最小値制約
                    if abs(low) < min_nonzero and low != 0:
                        low = min_nonzero if val > 0 else -min_nonzero
                    
                    key_bounds.append((low, high))
                else:
                    # 0または対象外の場合は固定（範囲なし）
                    key_bounds.append(None)
            
            bounds[key] = key_bounds
        
        logger.info(f"係数範囲計算完了: mode={mode}, range=±{range_percent}%")
        return bounds
    
    def add_trial_material(
        self,
        trial_id: int,
        params: dict,
        base_material: str
    ) -> str:
        """
        試行用の材料をmaterial.yamlに追記
        
        Args:
            trial_id: 試行ID
            params: {"c": [...], "m": [...], "t": [...], "g": [...]}
            base_material: ベースとなる材料名
        
        Returns:
            追加した材料名（例: "Proto2_Trial_001"）
        """
        # 現在のmaterial.yaml読込
        current_data = self._load_yaml(self.material_yaml_path)
        materials = current_data.get("materials", {})
        
        # ベース材料をコピー
        if base_material not in materials:
            raise ValueError(f"ベース材料が見つかりません: {base_material}")
        
        new_material = copy.deepcopy(materials[base_material])
        
        # 係数を更新
        elastic = new_material.get("parameters", {}).get("elastic", {})
        visco = new_material.get("parameters", {}).get("visco", {})
        
        if "c" in params:
            elastic["c"] = params["c"]
        if "m" in params:
            elastic["m"] = params["m"]
        if "t" in params:
            visco["t"] = params["t"]
        if "g" in params:
            visco["g"] = params["g"]
        
        # 新しい材料名
        new_name = f"Proto2_Trial_{trial_id:03d}"
        
        # 追加
        materials[new_name] = new_material
        current_data["materials"] = materials
        
        # 保存
        self._save_yaml(current_data, self.material_yaml_path)
        self._added_materials.append(new_name)
        
        logger.info(f"試行材料追加: {new_name}")
        return new_name
    
    def update_config_material(self, material_name: str) -> None:
        """
        config.yamlのmaterial_nameを更新
        
        Args:
            material_name: 使用する材料名
        """
        config = self._load_yaml(self.config_yaml_path)
        
        if "analysis" not in config:
            config["analysis"] = {}
        
        config["analysis"]["material_name"] = material_name
        
        self._save_yaml(config, self.config_yaml_path)
        logger.info(f"config.yaml更新: material_name={material_name}")
    
    def cleanup_trial_materials(self) -> None:
        """試行用材料をmaterial.yamlから削除"""
        if not self._added_materials:
            return
        
        current_data = self._load_yaml(self.material_yaml_path)
        materials = current_data.get("materials", {})
        
        for name in self._added_materials:
            if name in materials:
                del materials[name]
                logger.debug(f"試行材料削除: {name}")
        
        current_data["materials"] = materials
        self._save_yaml(current_data, self.material_yaml_path)
        
        logger.info(f"試行材料クリーンアップ完了: {len(self._added_materials)}件削除")
        self._added_materials.clear()
    
    def restore_original_config(self) -> None:
        """config.yamlを元の状態に復元"""
        self._save_yaml(self._original_config, self.config_yaml_path)
        logger.info("config.yamlを復元しました")
    
    def get_added_materials(self) -> list[str]:
        """追加した試行材料のリストを取得"""
        return self._added_materials.copy()


def save_optimized_material(
    output_path: str | Path,
    params: dict,
    base_material_data: dict,
    metadata: dict
) -> None:
    """
    最適化結果をYAMLファイルに出力
    
    Args:
        output_path: 出力ファイルパス
        params: 最適化されたパラメータ
        base_material_data: ベース材料のデータ
        metadata: メタデータ（trial数など）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 材料データを更新
    optimized = copy.deepcopy(base_material_data)
    elastic = optimized.get("parameters", {}).get("elastic", {})
    visco = optimized.get("parameters", {}).get("visco", {})
    
    if "c" in params:
        elastic["c"] = params["c"]
    if "m" in params:
        elastic["m"] = params["m"]
    if "t" in params:
        visco["t"] = params["t"]
    if "g" in params:
        visco["g"] = params["g"]
    
    # 出力データ構造
    output_data = {
        "Proto2_Optimized": optimized,
        "metadata": metadata
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        # ヘッダーコメント
        f.write("# Proto2 最適化結果 - OGDEN材料モデル\n")
        f.write(f"# Generated: {metadata.get('end_time', 'N/A')}\n\n")
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    logger.info(f"最適化結果を保存: {output_path}")
