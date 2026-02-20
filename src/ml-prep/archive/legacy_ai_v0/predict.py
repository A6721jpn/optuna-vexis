"""
AI-v0 推論モジュール

ソフトウェア組込み用のシンプルな推論API。

使用例:
    from predict import SafetyPredictor
    
    predictor = SafetyPredictor("artifacts_optuna_100")
    sample = [1.0] * 20  # 全て基準値
    print(predictor.is_safe(sample))  # True or False
"""
from __future__ import annotations

from pathlib import Path
from typing import Union, List, Sequence

import numpy as np
import joblib


class SafetyPredictor:
    """CAD成立性を予測するモデル"""
    
    FEATURE_NAMES = [
        "CROWN-D-L", "CROWN-D-H", "CROWN-W", "PUSHER-D-H", "PUSHER-D-L",
        "TIP-D", "STROKE-OUT", "STROKE-CENTER", "FOOT-W", "FOOT-MID",
        "SHOULDER-ANGLE-OUT", "SHOULDER-ANGLE-IN", "TOP-T", "TOP-DROP",
        "FOOT-IN", "DIAMETER", "HEIGHT", "TIP-DROP", "SHOUDER-T", "FOOT-OUT"
    ]
    
    def __init__(self, model_dir: Union[str, Path]):
        """
        Args:
            model_dir: model.joblib, scaler.joblibがあるディレクトリ
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        model_path = model_dir / "model.joblib"
        scaler_path = model_dir / "scaler.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self._n_features = len(self.FEATURE_NAMES)
    
    def _validate_input(self, features: np.ndarray) -> np.ndarray:
        """入力を検証してN×20の配列に整形"""
        features = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if features.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {features.shape[1]}"
            )
        return features
    
    def predict(self, features: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """
        Args:
            features: (N, 20) の寸法比率配列、または (20,) の1サンプル
        Returns:
            (N,) の予測ラベル (0=unsafe, 1=safe)
        """
        features = self._validate_input(features)
        scaled = self.scaler.transform(features)
        return self.model.predict(scaled)
    
    def predict_proba(self, features: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """
        Args:
            features: (N, 20) の寸法比率配列、または (20,) の1サンプル
        Returns:
            (N,) のsafe確率 [0.0 ~ 1.0]
        """
        features = self._validate_input(features)
        scaled = self.scaler.transform(features)
        return self.model.predict_proba(scaled)[:, 1]
    
    def is_safe(
        self, 
        features: Union[np.ndarray, Sequence[float]], 
        threshold: float = 0.5
    ) -> Union[bool, np.ndarray]:
        """
        Args:
            features: (N, 20) の寸法比率配列、または (20,) の1サンプル
            threshold: safe判定の閾値（デフォルト0.5）
        Returns:
            単一サンプルの場合: bool
            複数サンプルの場合: (N,) のbool配列
        """
        proba = self.predict_proba(features)
        result = proba >= threshold
        return result[0] if len(result) == 1 else result


def get_default_predictor() -> SafetyPredictor:
    """デフォルトモデルでSafetyPredictorを取得"""
    default_dir = Path(__file__).parent / "artifacts_optuna_100"
    return SafetyPredictor(default_dir)


# 動作確認用
if __name__ == "__main__":
    print("Loading model...")
    predictor = get_default_predictor()
    
    # テスト1: 全て基準値（safe期待）
    sample_normal = np.ones(20)
    prob = predictor.predict_proba(sample_normal)[0]
    print(f"Normal sample: safe_prob={prob:.3f}, is_safe={predictor.is_safe(sample_normal)}")
    
    # テスト2: バッチ処理
    samples = np.random.uniform(0.8, 1.2, size=(10, 20))
    probs = predictor.predict_proba(samples)
    print(f"Batch (10 samples): mean_prob={probs.mean():.3f}")
    
    print("All tests passed!")
