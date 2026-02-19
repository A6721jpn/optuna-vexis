import sys
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent / "src"))

from proto2.optimizer import Optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_autosampler_integration():
    # テスト用の設定
    bounds = {
        "c": [(0.1, 1.0)],
        "m": [(1.0, 10.0)],
        "t": [(0.01, 1.0)],
        "g": [(0.1, 0.9)]
    }
    
    config = {
        "sampler": "AUTO",
        "seed": 42,
        "n_startup_trials": 5,
        "max_trials": 10,
        "directions": ["minimize"]
    }
    
    # Optimizer初期化
    optimizer = Optimizer(bounds=bounds, config=config, mode="all")
    
    # 目的関数（ダミー）
    def dummy_objective(params):
        # 単純な2次関数
        c0 = params["c"][0]
        m0 = params["m"][0]
        return (c0 - 0.5)**2 + (m0 - 5.0)**2

    # ベースパラメータ（固定値用）
    base_params = {
        "c": [0.0],
        "m": [0.0],
        "t": [0.0],
        "g": [0.0]
    }

    # 最適化実行
    logger.info("Starting optimization with AutoSampler...")
    study = optimizer.run_optimization(
        objective_func=dummy_objective,
        base_params=base_params,
        n_trials=10
    )
    
    # 結果確認
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    if len(study.trials) == 10:
        print("Integration test PASSED")
    else:
        print(f"Integration test FAILED: Expected 10 trials, got {len(study.trials)}")

if __name__ == "__main__":
    test_autosampler_integration()
