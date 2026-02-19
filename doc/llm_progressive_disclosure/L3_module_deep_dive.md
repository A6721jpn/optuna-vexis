# L3 Module Deep Dive (v1 vs proto4)

## 1. 実装ライン比較

| 観点 | `src/v1` | `src/proto4-codex` | `src/proto4-claude` |
| --- | --- | --- | --- |
| 位置づけ | 現行ライン | proto4系のCodex向け実装 | proto4旧系/参照 |
| `DesignPoint` | `physical_params` を持つ | `params` のみ | `params` のみ |
| constraints domain | `physical`/`ratio` 両対応 | ratio前提 | ratio前提 |
| target value最適化 | `objective.target_values` 対応 | 未対応 | 未対応 |
| feasibility attr | `v1_0_*` + 互換 `proto4_*` | `proto4_*` | `proto4_*` |
| summary名 | `summary_v1_0.json` | `summary_proto4.json` | `summary_proto4.json` |

## 2. 仕様判断に使うテスト境界

- v1仕様:
  - `tests/v1/test_constraints_domain.py`
  - `tests/v1/test_physical_discretization.py`
  - `tests/v1/test_target_values_optimization.py`
- proto4-codex仕様:
  - `tests/proto4_codex/test_search_space_sampler_gate.py`
  - `tests/proto4_codex/test_objective_error_handling.py`
  - `tests/proto4_codex/test_geometry_adapter_subprocess.py`
- proto4-claude互換:
  - `tests/proto4/` 一式（`conftest.py` で alias 読み込み）

## 3. 命名/importの注意点

- `src/proto4-codex` はディレクトリ名にハイフンを含むため、実行時に alias ローダを使う。
  - `scripts/run_proto4_codex.py`
  - `scripts/proto4_codex_alias.py`
  - `tests/proto4_codex/conftest.py`
- `src/proto4-claude` も同様にテスト側 alias がある。
  - `tests/proto4/conftest.py`

## 4. 高コスト依存の境界

- FreeCAD依存:
  - `src/v1/geometry_adapter.py`
  - `src/v1/freecad_worker.py`
  - `src/proto4-codex/geometry_adapter.py`
- VEXIS/CAE依存:
  - `src/v1/cae_evaluator.py`
  - `src/proto4-codex/cae_evaluator.py`
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
