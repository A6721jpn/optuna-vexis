# Proto4 - CAD Feasibility-Gated Optimization

CAD実現性判定を組み込んだOptuna最適化パイプライン。

## 概要

Proto4は以下を統合します：
- **CAD Feasibility Gate**: MLモデルによるCAD破綻リスク予測
- **Optuna最適化**: パラメータ探索
- **Vexis CAE評価**: 実際のシミュレーション実行

## フロー

```
Optuna サンプリング → CAD Gate判定 → [Feasible] → Vexis CAE → 目的関数
                              → [Infeasible] → Penalty/Prune
```

## 使用方法

```bash
conda activate b123d
python -m src.proto4-claude.runner \
    --config config/optimizer_config.yaml \
    --limits config/proto4_limitations.yaml
```

## CAD Gate依存関係

`input/cad_gate_model` の `model.joblib` / `scaler.joblib` を使う場合、実行Pythonに
以下が必要です。

- `joblib`
- `scikit-learn`

```bash
python -m pip install joblib scikit-learn
```

## Sampler依存関係（AUTO使用時）

`optimization.sampler: "AUTO"` で OptunaHub AutoSampler を使う場合は、
追加で以下が必要です。

- `optunahub`
- `cmaes`
- `scipy`
- `torch`（CPU版可）

```bash
python -m pip install optunahub cmaes scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 指定可能サンプラー

- `AUTO`
- `GP`
- `TPE`
- `MOTPE` / `MO-TPE`
- `NSGAII`
- `NSGAIII`
- `RANDOM`
- `CMA-ES`

## 設定ファイル

| ファイル                         | 内容                                |
| -------------------------------- | ----------------------------------- |
| `config/optimizer_config.yaml`   | 最適化設定（sampler, max_trials等） |
| `config/proto4_limitations.yaml` | FreeCAD制約、CAD Gate、CAE設定      |

## モジュール構成

| ファイル              | 役割                         |
| --------------------- | ---------------------------- |
| `runner.py`           | CLIエントリポイント          |
| `config.py`           | 設定読み込み・バリデーション |
| `cad_gate.py`         | ML実現性予測                 |
| `objective.py`        | 目的関数オーケストレータ     |
| `cae_evaluator.py`    | Vexis実行                    |
| `geometry_adapter.py` | FreeCAD→STEP変換             |
| `constraints.py`      | 制約処理                     |
| `persistence.py`      | アーティファクト永続化       |

## テスト

```bash
pytest tests/proto4/ -v
```
