# L1 Task Router (Open Only What You Need)

## タスク別の最短読込

| タスク | 最初に読むファイル | 次に読むファイル |
| --- | --- | --- |
| 実行入口/CLI引数を変える | `scripts/run_v1.py` or `scripts/run_proto4_codex.py` | `src/v1/runner.py` or `src/proto4-codex/runner.py` |
| サンプラー設定・離散化を変更する | `src/v1/search_space.py` | `src/v1/config.py`, `tests/v1/test_physical_discretization.py` |
| CADゲート判定を修正する | `src/v1/cad_gate.py` | `src/v1/objective.py`, `tests/proto4_codex/test_cad_gate_io.py` |
| FreeCAD STEP生成の不具合を追う | `src/v1/geometry_adapter.py` | `src/v1/freecad_worker.py`, `src/v1/freecad_engine.py` |
| CAE結果の読み取り/失敗検知を修正する | `src/v1/cae_evaluator.py` | `tests/proto4_codex/test_cae_solver_monitor.py`, `tests/proto4_codex/test_cae_logging.py` |
| ペナルティ/分岐ロジックを修正する | `src/v1/objective.py` | `src/v1/constraints.py`, `tests/proto4_codex/test_objective_error_handling.py` |
| 設定読み込みバリデーション修正 | `src/v1/config.py` | `tests/v1/test_constraints_domain.py`, `tests/v1/test_target_values_optimization.py` |
| 出力形式/レポート修正 | `src/v1/persistence.py` | `src/v1/reporting.py`, `tests/proto4_codex/test_reporting.py` |
| proto4系のE2E導線確認 | `scripts/proto4_harness.py` | `config/proto4_harness.yaml`, `doc/proto4_e2e.md` |

## いまは読まなくてよい場所

- `src/proto1`, `src/proto2`, `src/proto3`: 履歴実装
- `debug_*.py`, `debug_output*.txt`: 調査ログ
- `devlog/`: 日次メモ

## 迷ったときの優先順位

1. 実行中のライン（通常は `src/v1`）を優先
2. 設定 (`config/`) とテスト (`tests/`) で仕様を確定
3. それでも不明な時だけ `doc/` の設計資料を読む
