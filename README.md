# Optuna for VEXIS

FreeCAD の寸法を変えながら STEP を作成し、VEXIS の解析結果を Optuna で探索するためのツールです。

## 目的

- 目標に近い寸法条件を探す
- クリック率、ピーク荷重、RMSE などを評価する
- 試行ごとの寸法、結果、失敗理由を残す

## できること

- FreeCAD モデルから trial ごとの STEP を作る
- VEXIS を実行して結果カーブを評価する
- 単目的 / 多目的の Optuna 探索を回す
- 既存 study DB から続きを回す
- レポート、グラフ、trial ログを出力する

## 準備

必要な入力:

- `input/model.FCStd`
- `input/target_curve_generated.csv`
- `config/optimizer_config.yaml`
- `config/v1_0_limitations.yaml`
- `input/cad_gate_model/`（CAD gate を使う場合）

初回のみ:

```bash
git submodule update --init --recursive
python -m pip install optuna pandas matplotlib pyyaml scipy joblib scikit-learn optunahub cmaes
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 実行

```bash
python scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml
```

試行数を指定する場合:

```bash
python scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml --max-trials 50
```

主なオプション:

- `--max-trials`: trial 数を指定
- `--dry-run`: 導線確認
- `--verbose`: 詳細ログ
- `--version`: バージョン表示

## 設定

- `config/optimizer_config.yaml`: trial 数、sampler、目的関数、入出力
- `config/v1_0_limitations.yaml`: 寸法範囲、CAD gate、VEXIS timeout、penalty

## 出力

- `output/report_v1_0.md`
- `output/summary_v1_0.json`
- `output/optuna_study_v1_0.db`
- `output/trials/trial_<id>/trial_info.json`
- `output/report_assets/`
- `output/logs/`
- `input/step/v1_0_trial_<id>.step`

## 比較用

```bash
python scripts/run_proto4_codex.py --config config/optimizer_config.yaml --limits config/proto4_limitations.yaml
```
