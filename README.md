# Optuna for VEXIS

FreeCAD で形状を更新し、STEP を生成し、VEXIS で CAE 評価し、その結果を Optuna で最適化するためのパイプラインです。
現行の実装ラインは `src/v1` で、目標カーブに対する RMSE や特徴量誤差を最小化する用途を想定しています。

## Current Status

- Production line: `src/v1`
- Comparison line: `src/proto4-codex`
- Other tracked lines: `src/proto4-claude`, `src/v2`, `src/v2-claude`, `src/ml-prep`
- Historical lines: `src/proto1`, `src/proto2`, `src/proto3`
- Main entrypoints:
  - `scripts/run_v1.py`
  - `scripts/run_proto4_codex.py`

追加の実験用スクリプトや設定ファイルもありますが、通常は `src/v1` を起点に見れば十分です。

## LLM Context Docs

Codex セッションでは、段階的に文脈を読むために以下を入口にします。

- `doc/llm_progressive_disclosure/README.md`

## What Production v1 Does

- FreeCAD モデルの寸法探索範囲を読み込み、Optuna の探索空間を構築する
- CAD feasibility gate により、実行前に CAD 不成立になりやすい点を避ける
- FreeCAD から STEP を生成し、VEXIS を起動して CAE を評価する
- RMSE または target feature error を目的関数として最適化する
- 試行ごとの JSON、study DB、集計サマリ、Markdown レポートを保存する

## Required Inputs And Environment

- Windows
- Python 3.11 以上
- Git submodule を初期化済みであること
- `input/model.FCStd`
- `input/target_curve_generated.csv`
- `config/optimizer_config.yaml`
- `config/v1_0_limitations.yaml`
- `cad_gate.enabled: true` の場合:
  - `input/cad_gate_model/model.joblib`
  - `input/cad_gate_model/scaler.joblib`

このリポジトリには、現時点で top-level の `requirements.txt` や `pyproject.toml` はありません。
そのため、実行に使う Python 環境へ必要パッケージを個別に入れてください。

最低限の目安:

- Baseline: `optuna`, `pandas`, `matplotlib`, `pyyaml`, `scipy`
- CAD gate 使用時: `joblib`, `scikit-learn`
- `optimization.sampler: "AUTO"` 使用時: `optunahub`, `cmaes`, `torch`

例:

```bash
git submodule update --init --recursive
python -m pip install optuna pandas matplotlib pyyaml scipy joblib scikit-learn optunahub cmaes
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Main Config Files

- `config/optimizer_config.yaml`
  - Optuna sampler
  - trial 数
  - objective 設定
  - logging / paths
- `config/v1_0_limitations.yaml`
  - FreeCAD 制約範囲
  - CAD gate 設定
  - CAE タイムアウト / リトライ
  - ペナルティ設定
- `config/proto4_limitations.yaml`
  - `proto4-codex` ライン用の制約設定

## Quick Start

Production v1.0 を root から起動します。

```bash
python scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml
```

よく使うオプション:

- `--max-trials 50`: config より優先して試行回数を上書き
- `--dry-run`: CAE 実行を伴う本番処理の前に導線確認
- `--verbose`: コンソールログを詳細化
- `--version`: 実行時の version / git 情報を JSON で表示

補足:

- v1 の study は `output/optuna_study_v1_0.db` に保存されます
- 既存 DB がある場合、runner は `load_if_exists=True` で study を再利用します
- そのため、旧 README にあった「resume 専用コマンド」は通常不要です

## Proto4 Comparison Line

比較用に `proto4-codex` ラインも実行できます。

```bash
python scripts/run_proto4_codex.py --config config/optimizer_config.yaml --limits config/proto4_limitations.yaml
```

## Outputs

Production v1 実行後の主な出力:

- `output/optuna_study_v1_0.db`
- `output/run_config_snapshot.json`
- `output/summary_v1_0.json`
- `output/report_v1_0.md`
- `output/report_assets/optimization_history.png` または `.svg`
- `output/report_assets/pareto_front_2d.png` または `.svg`
- `output/trials/trial_<id>/trial_info.json`
- `output/logs/`
- `output/logs/vexis/`
- `input/step/v1_0_trial_<id>.step`

## Repository Layout

代表的な構成は以下です。

```text
.
├── config/                     # Optimizer / limitation YAML
├── doc/                        # 設計メモ、runbook、progressive disclosure
├── input/                      # FCStd, target curve, CAD gate model, generated STEP
├── output/                     # Study DB, logs, reports, summaries
├── scripts/                    # Run scripts and support utilities
├── src/
│   ├── v1/                     # Production line
│   ├── proto4-codex/           # Comparison line
│   ├── proto4-claude/          # Older proto4 implementation
│   ├── v2/                     # Experimental line
│   ├── v2-claude/              # Alternative v2 line
│   ├── ml-prep/                # ML preparation utilities
│   └── proto1/proto2/proto3/   # Historical implementations
├── tests/                      # Current tests
├── vexis/                      # VEXIS submodule
└── cad-automaton/              # CAD-related submodule
```

## Versioning

`src/v1/versioning.py` の定義では、Production line は以下です。

- Product: `optuna-for-vexis`
- Line: `Production`
- Version: `1.0.0`
- Baseline: `v1.0`

確認コマンド:

```bash
python scripts/run_v1.py --version
```
