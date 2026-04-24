# L3 Module Deep Dive

## 1. 現行ライン

- 現行実装は `src/v1`
- `DesignPoint` は `physical_params` を持つ
- constraints domain は `physical` / `ratio` 両対応
- 目標値最適化は `objective.target_values` で扱う
- Summary は `summary_v1_0.json`

## 2. 仕様判断に使うテスト境界

- v1仕様:
  - `tests/v1/test_constraints_domain.py`
  - `tests/v1/test_physical_discretization.py`
  - `tests/v1/test_target_values_optimization.py`

## 3. 命名/importの注意点

- テスト境界は `tests/v1/` に集約する。

## 4. 高コスト依存の境界

- FreeCAD依存:
  - `src/v1/geometry_adapter.py`
  - `src/v1/freecad_worker.py`
- VEXIS/CAE依存:
  - `src/v1/cae_evaluator.py`
  - `vexis/`（サブモジュール）

## 5. 深掘り時の推奨読込順

1. 対象ラインの `runner.py`
2. 同ラインの `config.py` と `search_space.py`
3. `objective.py`（分岐仕様）
4. I/O境界（`geometry_adapter.py`, `cae_evaluator.py`）
5. 該当テスト

## 6. 非推奨（コンテキスト節約のため）

- 目的と無関係な proto を同時に読む
- `devlog/` を先に読む
- `debug_output*.txt` を初手で読む
