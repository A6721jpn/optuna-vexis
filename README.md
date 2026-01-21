# Optuna for VEXIS (Proto2)

Optunaを用いたOGDEN材料モデル係数の自動最適化システム。
VEXIS (CAEソルバー) と連携し、ターゲットカーブ（実験値）に対するRMSEが最小となるパラメータを探索します。

## 主な機能

- **係数最適化**: Optuna (TPE/NSGA-II) による効率的な探索
- **自動離散化**: CAEソフトの有効数字に合わせた係数丸め処理（`step`パラメータ対応）
- **結果可視化**: 最適化推移、最良カーブ比較、パレートフロントの自動プロット
- **学習管理**: データベースによる中断/再開 (`--resume`)、バックアップ機能

## ディレクトリ構成

```
.
├── config/                 # 設定ファイル
│   └── optimizer_config.yaml
├── input/                  # 入力ファイル (STEP, Target CSV)
├── output/                 # 出力結果 (Log, Plots, DB, XML/YAML)
├── src/
│   └── proto2/            # ソースコード
│       ├── main.py        # エントリポイント
│       ├── optimizer.py   # Optunaラッパー
│       ├── visualizer.py  # 可視化モジュール
│       └── ...
└── vexis/                  # CAEソルバー (Submodule)
```

## 必要要件

- Windows OS
- Python 3.11+
- 依存ライブラリ: `optuna`, `pandas`, `matplotlib`, `pyyaml`, `scipy` など

## セットアップ

1.  依存ライブラリのインストール
    ```bash
    pip install -r src/proto2/requirements.txt
    ```

2.  VEXISサブモジュールの準備
    ```bash
    git submodule update --init --recursive
    # VEXIS側のセットアップが必要な場合は実施してください
    ```

## 使い方

### 1. 設定の編集

`config/optimizer_config.yaml` で最適化パラメータを調整します。

```yaml
optimization:
  max_trials: 30              # 試行回数
  discretization_step: 0.0001 # 係数の離散化ステップ（小数点以下4桁）
  objective_type: "single"    # single(RMSE) または multi(多目的)
```

### 2. 最適化の実行

プロジェクトルートから以下のコマンドを実行します。

```bash
# 新規実行（既存DBがある場合はバックアップされます）
python -m src.proto2.main --config config/optimizer_config.yaml

# 継続実行（既存DBの続きから再開）
python -m src.proto2.main --config config/optimizer_config.yaml --resume
```

### 3. 結果の確認

`output/` ディレクトリに結果が出力されます。

- **`output/plots/`**:
    - `optimization_history.png`: 最適化の推移グラフ
    - `best_result_comparison.png`: ターゲット定義(点線)と最良結果(赤線)の比較
- **`optimized_material.yaml`**: 最適化されたOGDEN係数定義
- **`summary_proto2.json`**: 実行結果サマリ
- **`optuna_study_proto2.db`**: 学習データベース
