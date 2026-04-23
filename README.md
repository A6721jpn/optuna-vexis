# Optuna for VEXIS

FreeCAD の寸法を振って STEP を作り、VEXIS の解析結果を見ながら Optuna で条件を探すためのリポジトリです。

手で「寸法変更 → STEP 出力 → 解析 → 結果確認」を繰り返す代わりに、指定した範囲の中から目標に近い形状条件を探します。

## 目的

- VEXIS の解析結果が目標値に近づく寸法条件を探す
- クリック率、ピーク荷重、RMSE などを評価値として使う
- 試した寸法、解析結果、失敗理由をあとから追える形で残す
- CAD として成立しにくい候補をなるべく避け、無駄な解析を減らす

## できること

- `input/model.FCStd` の寸法を trial ごとに変更する
- FreeCAD から trial ごとの STEP を出力する
- VEXIS を実行し、結果カーブや特徴量を評価する
- Optuna で単目的または多目的の探索を回す
- 途中まで回した study DB を使って続きを回す
- 各 trial の結果、集計 JSON、Markdown レポート、グラフを出力する

## 使う前に用意するもの

- Windows
- Python 3.11 以上
- `input/model.FCStd`
- `input/target_curve_generated.csv`
- `config/optimizer_config.yaml`
- `config/v1_0_limitations.yaml`
- VEXIS submodule
- CAD gate を使う場合は `input/cad_gate_model/model.joblib` と `input/cad_gate_model/scaler.joblib`

submodule は最初に初期化してください。

```bash
git submodule update --init --recursive
```

このリポジトリには top-level の `requirements.txt` や `pyproject.toml` はありません。
実行に使う Python 環境へ必要なパッケージを入れてください。

```bash
python -m pip install optuna pandas matplotlib pyyaml scipy joblib scikit-learn optunahub cmaes
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`optimization.sampler` で `AUTO` を使わない場合、`optunahub`、`cmaes`、`torch` が不要なケースもあります。

## 実行

通常は v1 を使います。プロジェクトルートから実行してください。

```bash
python scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml
```

試行数だけ変えたい場合:

```bash
python scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml --max-trials 50
```

よく使うオプション:

- `--max-trials 50`: 今回の上限 trial 数を指定する
- `--dry-run`: 解析を本格的に回す前に導線を確認する
- `--verbose`: コンソールに詳しいログを出す
- `--version`: バージョンと git 情報を表示する

`output/optuna_study_v1_0.db` が残っている場合は、既存 study を読み込んで続きから動きます。
完全に新しく回したい場合は、必要な出力を退避してから既存 DB を消してください。

## 主に触る設定

`config/optimizer_config.yaml`

- trial 数
- sampler
- 単目的 / 多目的
- 目標値
- 評価に使う特徴量
- 入出力パス

`config/v1_0_limitations.yaml`

- 探索する寸法名と範囲
- CAD gate の有効 / 無効
- VEXIS の timeout と retry
- 失敗時の penalty

## 結果を見る場所

- `output/report_v1_0.md`: 実行結果のレポート
- `output/summary_v1_0.json`: 実行サマリ
- `output/optuna_study_v1_0.db`: Optuna study DB
- `output/trials/trial_<id>/trial_info.json`: trial ごとの詳細
- `output/report_assets/`: グラフ画像
- `output/logs/`: 実行ログ
- `output/logs/vexis/`: VEXIS のログ
- `input/step/v1_0_trial_<id>.step`: trial ごとの STEP

## 比較用の実行

必要なときだけ `proto4-codex` も実行できます。

```bash
python scripts/run_proto4_codex.py --config config/optimizer_config.yaml --limits config/proto4_limitations.yaml
```
