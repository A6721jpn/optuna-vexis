# CAD Gate 学習手順書（v2 / 相対拘束対応）

## 目的

`v2` の FreeCAD 相対拘束アルゴリズム（修復ソルバ込み）を反映したラベルで、CAD gate MLモデルを再学習する。

## 前提

- 実行ディレクトリ: リポジトリルート
- `config/v2_limitations.yaml` の `freecad.relative_constraints` が有効
- FreeCAD 実行環境が有効
- 学習用 Python に以下を導入

```bash
python -m pip install numpy scikit-learn optuna joblib
```

## 最短手順（推奨）

1. 1コマンドでデータ生成 + 学習

```bash
python src/ml-prep/scripts/retrain_cad_gate.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml \
  --samples 2000 \
  --trials 120 \
  --timeout-sec 1800
```

CPUを強く使う例（16論理スレッド想定）:

```bash
python src/ml-prep/scripts/retrain_cad_gate.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml \
  --samples 5000 \
  --dataset-workers 6 \
  --trials 120 \
  --timeout-sec 5400 \
  --cv-n-jobs 4 \
  --optuna-n-jobs 4 \
  --tree-n-jobs 1 \
  --blas-threads 1
```

2. 学習済みモデル出力を確認

- `src/ml-prep/models/cad_gate_model/model.joblib`
- `src/ml-prep/models/cad_gate_model/scaler.joblib`
- `src/ml-prep/models/cad_gate_model/metadata.json`

3. `v2` 実行で新モデルを利用

```bash
python scripts/run_v2.py --config config/optimizer_config.yaml --limits config/v2_limitations.yaml
```

## 分割手順（詳細制御したい場合）

1. データ生成

```bash
python src/ml-prep/scripts/generate_cad_gate_dataset.py \
  --config config/optimizer_config.yaml \
  --limits config/v2_limitations.yaml \
  --samples 3000 \
  --sampler lhs \
  --seed 42 \
  --output src/ml-prep/data/cad_gate_dataset.csv
```

2. 学習

```bash
python src/ml-prep/scripts/train_cad_gate_automl.py \
  --dataset src/ml-prep/data/cad_gate_dataset.csv \
  --model-dir src/ml-prep/models/cad_gate_model \
  --trials 200 \
  --timeout-sec 3600 \
  --seed 42
```

## 主要パラメータの目安

- `--samples`
  - まずは `1000-3000` を推奨
- `--trials`
  - まずは `80-200` を推奨
- `--timeout-sec`
  - 試行数が多い場合は `1800-7200`

## 失敗時の確認ポイント

1. 依存不足

```bash
python -m pip install numpy scikit-learn optuna joblib
```

2. FreeCAD 実行失敗

- `config/v2_limitations.yaml` の `freecad.fcstd_path`、`sketch_name` を確認
- FreeCAD 実行環境（`FREECAD_BIN` / `FREECAD_PYTHON`）を確認

3. クラス不均衡（全件 feasible / infeasible）

- `samples` を増やす
- 相対拘束ルールや FreeCAD 制約設定を見直す

## 補足

- 旧MLスクリプトは `src/ml-prep/archive/` に隔離済み。
- `v2` の CAD gate 既定参照先は `src/ml-prep/models/cad_gate_model`。

## 5k/10k/20k の性能比較CSVを作る

現在のモデル（想定: 5k 学習済み）を 1 行目に記録し、その後 10k / 20k を再学習して同じ CSV に追記する。

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\sweep_cad_gate_sample_sizes.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --sizes 10000,20000 --dataset-workers 6 --trials 120 --timeout-sec 5400 --cv-n-jobs 4 --optuna-n-jobs 4 --tree-n-jobs 1 --blas-threads 1 --csv-output src\ml-prep\reports\cad_gate_sample_size_sweep.csv
```

出力:

- `src/ml-prep/reports/cad_gate_sample_size_sweep.csv`

主な列:

- `roc_auc`, `pr_auc`, `f1_default_0_5`
- `recommended_threshold`
- `fpr_t_0_44`, `fnr_t_0_44`
- `fpr_t_0_5`, `fnr_t_0_5`
- `fpr_t_recommended`, `fnr_t_recommended`

## Active Learningを1サイクル実行（+2000件）

既存の学習データセットからモデルを使って不確実点を選び、FreeCADラベルを付与して +2000 行を追記する。

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\active_learn_cad_gate_cycle.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --dataset src\ml-prep\data\cad_gate_dataset.csv --add-samples 2000 --pool-size 20000 --dataset-workers 6
```

追記後に同時再学習まで行う場合:

```powershell
.\.venv\Scripts\python.exe src\ml-prep\scripts\active_learn_cad_gate_cycle.py --config config\optimizer_config.yaml --limits config\v2_limitations.yaml --dataset src\ml-prep\data\cad_gate_dataset.csv --add-samples 2000 --pool-size 20000 --dataset-workers 6 --retrain --trials 120 --timeout-sec 5400 --cv-n-jobs 4 --optuna-n-jobs 4 --tree-n-jobs 1 --blas-threads 1
```

出力:

- 更新後データセット: `src/ml-prep/data/cad_gate_dataset.csv`
- サイクル要約: `src/ml-prep/data/cad_gate_dataset.active_learn.summary.json`
