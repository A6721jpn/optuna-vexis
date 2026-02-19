# L0 Project Snapshot (Read First)

## 1. このリポジトリは何か

FreeCADで形状を更新してSTEPを生成し、VEXISでCAE評価し、Optunaで探索する最適化パイプラインです。

## 2. どこが現行か

- 現行実装: `src/v1`
- proto4系実装: `src/proto4-codex`
- proto4旧実装: `src/proto4-claude`
- 実行スクリプト:
  - `scripts/run_v1.py`
  - `scripts/run_proto4_codex.py`
  - `scripts/proto4_harness.py`（E2E確認）

## 3. パイプライン1行要約

`runner` -> `search_space` -> `objective` -> (`cad_gate` + `geometry_adapter` + `cae_evaluator`) -> `persistence` / `reporting`

## 4. 主要設定ファイル

- 最適化共通: `config/optimizer_config.yaml`
- v1制約: `config/v1_0_limitations.yaml`
- proto4制約: `config/proto4_limitations.yaml`
- harness: `config/proto4_harness.yaml`

## 5. 主要出力

- Study DB: `output/optuna_study_v1_0.db` / `output/optuna_study_proto4.db`
- Trial記録: `output/trials/trial_<id>/trial_info.json`
- Summary: `output/summary_v1_0.json` / `output/summary_proto4.json`
- レポート: `output/report_*.md`（実行設定次第）

## 6. 最短実行コマンド

```bash
python scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml
python scripts/run_proto4_codex.py --config config/optimizer_config.yaml --limits config/proto4_limitations.yaml
```

## 7. 次に読むべきもの

- タスク別に読むなら `L1_task_router.md`
- 実行フローを追うなら `L2_runtime_pipeline.md`
- v1/proto4差分を扱うなら `L3_module_deep_dive.md`
