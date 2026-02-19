"""
修正後のSTEP編集機能をテスト
"""
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.proto1.step_editor import StepEditor

# OpenCASCADE imports
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib


def get_bounds(shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()


if __name__ == "__main__":
    input_step = project_root / "input" / "example_1.stp"
    output_step = project_root / "temp" / "test_fixed_scale.step"
    
    print("=" * 60)
    print("修正後のSTEP編集機能テスト")
    print("=" * 60)
    
    # テストパラメータ（scale = 1.0で変更なし）
    test_params = {
        "height_scale": 1.0,
        "overall_scale": 1.0,
        "radial_scale": 1.0,
    }
    
    dim_configs = [
        {"name": "height_scale", "step_reference": {"method": "scale_z"}},
        {"name": "overall_scale", "step_reference": {"method": "scale_uniform"}},
        {"name": "radial_scale", "step_reference": {"method": "coordinate", "axis": "Y"}},
    ]
    
    # STEP編集
    editor = StepEditor()
    editor.load(str(input_step))
    
    # 元の境界
    orig_bounds = get_bounds(editor._shape)
    print(f"\n元の境界:")
    print(f"  Y: [{orig_bounds[1]:.4f}, {orig_bounds[4]:.4f}] → 奥行: {orig_bounds[4]-orig_bounds[1]:.4f}")
    print(f"  Z: [{orig_bounds[2]:.4f}, {orig_bounds[5]:.4f}] → 高さ: {orig_bounds[5]-orig_bounds[2]:.4f}")
    
    # スケール適用
    editor.apply_dimensions(test_params, dim_configs)
    
    # 編集後の境界
    new_bounds = get_bounds(editor._shape)
    print(f"\n編集後の境界 (scale=1.0):")
    print(f"  Y: [{new_bounds[1]:.4f}, {new_bounds[4]:.4f}] → 奥行: {new_bounds[4]-new_bounds[1]:.4f}")
    print(f"  Z: [{new_bounds[2]:.4f}, {new_bounds[5]:.4f}] → 高さ: {new_bounds[5]-new_bounds[2]:.4f}")
    
    # 比較
    orig_height = orig_bounds[5] - orig_bounds[2]
    new_height = new_bounds[5] - new_bounds[2]
    print(f"\n高さ比率: {new_height/orig_height:.4f} (1.0000が正解)")
    
    # 保存
    editor.save(str(output_step))
    print(f"\nテストファイル保存: {output_step}")
    
    # 0.9倍テスト
    print("\n" + "-" * 60)
    print("0.9倍テスト")
    
    test_params_09 = {
        "height_scale": 0.9,
        "overall_scale": 1.0,
        "radial_scale": 1.0,
    }
    
    editor2 = StepEditor()
    editor2.load(str(input_step))
    editor2.apply_dimensions(test_params_09, dim_configs)
    
    new_bounds_09 = get_bounds(editor2._shape)
    new_height_09 = new_bounds_09[5] - new_bounds_09[2]
    print(f"高さ比率: {new_height_09/orig_height:.4f} (0.9000が正解)")
