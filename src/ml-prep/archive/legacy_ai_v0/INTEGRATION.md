# AI-v0 組込み開発マニュアル

本モデルをソフトウェアに組み込むための手順書。

---

## 1. モデルファイル

### 必要なファイル（3点セット）

| ファイル        | 説明                     | サイズ目安 |
| --------------- | ------------------------ | ---------- |
| `model.joblib`  | 学習済みMLPモデル        | ~500KB     |
| `scaler.joblib` | 特徴量標準化器           | ~5KB       |
| `metrics.json`  | モデル性能情報（参考用） | ~1KB       |

**配置場所**: `src/ai-v0/artifacts_optuna_100/`

---

## 2. 依存関係

### 最小構成
```
scikit-learn>=1.0.0
joblib>=1.0.0
numpy>=1.20.0
```

### requirements.txt
```
scikit-learn==1.8.0
joblib==1.5.3
numpy==2.4.1
```

---

## 3. 推論コード

### 基本的な使い方

```python
import joblib
import numpy as np
from pathlib import Path

# モデルのロード
model_dir = Path("src/ai-v0/artifacts_optuna_100")
model = joblib.load(model_dir / "model.joblib")
scaler = joblib.load(model_dir / "scaler.joblib")

# 入力データ（20次元の寸法比率ベクトル）
# 各次元は基準値に対する比率（例: 1.0 = 基準値、1.1 = 10%増）
features = np.array([[
    1.0,   # CROWN-D-L
    1.05,  # CROWN-D-H
    0.98,  # CROWN-W
    1.02,  # PUSHER-D-H
    1.0,   # PUSHER-D-L
    1.0,   # TIP-D
    1.0,   # STROKE-OUT
    1.0,   # STROKE-CENTER
    1.0,   # FOOT-W
    1.0,   # FOOT-MID
    1.0,   # SHOULDER-ANGLE-OUT
    1.0,   # SHOULDER-ANGLE-IN
    1.0,   # TOP-T
    1.0,   # TOP-DROP
    1.0,   # FOOT-IN
    1.0,   # DIAMETER
    1.0,   # HEIGHT
    1.0,   # TIP-DROP
    1.0,   # SHOUDER-T
    1.0,   # FOOT-OUT
]])

# スケーリング
features_scaled = scaler.transform(features)

# 推論
prediction = model.predict(features_scaled)  # 0 or 1
probability = model.predict_proba(features_scaled)[:, 1]  # safe確率

print(f"Prediction: {'safe' if prediction[0] == 1 else 'unsafe'}")
print(f"Safe probability: {probability[0]:.3f}")
```

---

## 4. 入出力仕様

### 入力

| 項目 | 値                                          |
| ---- | ------------------------------------------- |
| 型   | `np.ndarray` (float64)                      |
| 形状 | `(N, 20)` - N件のサンプル                   |
| 範囲 | 各次元 0.78 ~ 1.22 を想定（学習データ範囲） |

### 特徴量の順序（20次元）

```python
FEATURE_NAMES = [
    "CROWN-D-L", "CROWN-D-H", "CROWN-W", "PUSHER-D-H", "PUSHER-D-L",
    "TIP-D", "STROKE-OUT", "STROKE-CENTER", "FOOT-W", "FOOT-MID",
    "SHOULDER-ANGLE-OUT", "SHOULDER-ANGLE-IN", "TOP-T", "TOP-DROP",
    "FOOT-IN", "DIAMETER", "HEIGHT", "TIP-DROP", "SHOUDER-T", "FOOT-OUT"
]
```

### 出力

| メソッド                 | 戻り値              | 説明                               |
| ------------------------ | ------------------- | ---------------------------------- |
| `model.predict(X)`       | `np.ndarray[int]`   | 0=unsafe, 1=safe                   |
| `model.predict_proba(X)` | `np.ndarray[float]` | [:, 0]=unsafe確率, [:, 1]=safe確率 |

---

## 5. 推論クラス（推奨実装）

```python
"""AI-v0 推論ラッパークラス"""
from pathlib import Path
from typing import Union, List
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
        self.model = joblib.load(model_dir / "model.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Args:
            features: (N, 20) の寸法比率配列
        Returns:
            (N,) の予測ラベル (0=unsafe, 1=safe)
        """
        features = np.atleast_2d(features)
        scaled = self.scaler.transform(features)
        return self.model.predict(scaled)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Args:
            features: (N, 20) の寸法比率配列
        Returns:
            (N,) のsafe確率
        """
        features = np.atleast_2d(features)
        scaled = self.scaler.transform(features)
        return self.model.predict_proba(scaled)[:, 1]
    
    def is_safe(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Args:
            features: (N, 20) の寸法比率配列
            threshold: safe判定の閾値（デフォルト0.5）
        Returns:
            (N,) のbool配列
        """
        proba = self.predict_proba(features)
        return proba >= threshold


# 使用例
if __name__ == "__main__":
    predictor = SafetyPredictor("src/ai-v0/artifacts_optuna_100")
    
    # 単一サンプル
    sample = np.ones(20)  # 全て基準値
    print(f"Safe: {predictor.is_safe(sample)[0]}")
    print(f"Probability: {predictor.predict_proba(sample)[0]:.3f}")
```

---

## 6. 性能仕様

| 項目         | 値                      |
| ------------ | ----------------------- |
| 推論時間     | < 1ms / サンプル        |
| メモリ使用量 | ~50MB（モデルロード後） |
| F1スコア     | 0.862                   |
| ROC-AUC      | 0.870                   |
| Accuracy     | 83.1%                   |

### 注意事項

- 入力値が学習データ範囲（0.78~1.22）を大きく外れる場合、精度保証なし
- `threshold`を下げる（例: 0.4）とUnsafe Recallが向上（保守的判定）
- バッチ処理推奨（100件以上をまとめて推論すると効率的）

---

## 7. モデル更新時の対応

新しいモデルに入れ替える場合:

1. `model.joblib`と`scaler.joblib`を差し替え
2. 特徴量の順序が変わっていないか確認（`FEATURE_NAMES`）
3. 簡単な動作確認テスト実行

```python
# 動作確認テスト
def test_model_load():
    predictor = SafetyPredictor("path/to/new/model")
    sample = np.ones(20)
    proba = predictor.predict_proba(sample)
    assert 0 <= proba[0] <= 1, "出力が確率範囲外"
    print(f"Model OK: safe_prob={proba[0]:.3f}")
```
