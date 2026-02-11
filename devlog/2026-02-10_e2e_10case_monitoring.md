# 作業ログ: E2E 10ケース解析 + 失敗モニタリング (実行: 2026-02-11)

## 目的
- E2Eテストとして 10ケース解析を実行し、クラッシュ/失敗を詳細に監視・記録する。
- テスト用ジオメトリが無い場合は生成する前提で事前確認する。

## 事前確認
- 実施日時(ローカル): 2026-02-11 06:21 - 06:56
- 既存テストSTEP確認:
  - `input/step/harness_step.step`
  - `input/step/test_baseline.step`
  - `input/step/test_modified.step`
- 既存ジオメトリが存在したため、事前生成は不要。

## 実行内容

### A. Optuna E2E 10ケース (CAD gate ON)
- コマンド:
  - `python scripts/proto4_harness.py --optimizer temp/e2e10_optimizer_config_20260210.yaml --limits temp/e2e10_proto4_limitations_20260210.yaml --run-root output/harness/e2e10_20260210 --max-trials 10 --run-proto4`
- 実行ID: `20260211_062158`
- 出力:
  - `output/e2e_20260210_10cases/summary_proto4.json`
  - `output/logs/proto4_20260211_062220.log`
  - `output/e2e_20260210_10cases/trials/trial_*/trial_info.json`
- 結果:
  - 10/10 が `cad_infeasible`
  - CAD gate reason は全件 `ml_infeasible`
  - CAE到達件数 0
- Optuna user_attrs:
  - `proto4_failure_stage = cad_gate`
  - `proto4_failure_reason = ml_infeasible`
  - `proto4_feasibility_violation` は 0.9275 - 0.9959

### B. Optuna E2E 10ケース (CAD gate OFF: 下流観測用)
- コマンド:
  - `python scripts/proto4_harness.py --optimizer temp/e2e10_optimizer_config_20260210_gateoff.yaml --limits temp/e2e10_proto4_limitations_20260210_gateoff.yaml --run-root output/harness/e2e10_20260210_gateoff --max-trials 10 --run-proto4`
- 実行ID: `20260211_062351`
- 出力:
  - `output/e2e_20260210_10cases_gateoff/summary_proto4.json`
  - `output/logs/proto4_20260211_062401.log`
  - `output/e2e_20260210_10cases_gateoff/trials/trial_*/trial_info.json`
- 結果:
  - 10/10 が `cad_infeasible`
  - 失敗ステージは全件 `geometry_generation`
  - 主因: `FreeCAD error ... FreeCAD not found in any conda environment or fallback path`

### C. 既存STEPを用いた CAE 10連続実行 (VEXIS監視補完)
- 実行内容:
  - 既存STEP(3種)をローテーションし、`trial_id=1000..1009` で `CaeEvaluator.evaluate()` を実行
  - timeout は 180秒
- 出力:
  - `output/e2e_20260210_cae_batch/cae_batch_10cases_summary.json`
  - `output/logs/vexis/proto4_trial_1000_vexis.log` 〜 `output/logs/vexis/proto4_trial_1009_vexis.log`
- 結果:
  - SUCCESS: 0 / FAIL: 10
  - 失敗理由は全件 `timeout_after_180s`
  - 平均実行時間: 180.47 sec
  - 最終 Solver 進捗は 10% - 98% で分布

## 主要失敗シグネチャ

### 1) CAD生成段
- 全件: `geometry_generation` で失敗
- メッセージ: `FreeCAD not found in any conda environment or fallback path`

### 2) CAE段 (既存STEPテスト)
- 全件: `timeout_after_180s`
- VEXISログ傾向:
  - `singular node ... failed to assign to irregular vertex` 警告が多発
  - `Mesh generation error summary` が反復
  - `fatal error` / `error termination` / `traceback` は検出されず

## 環境診断メモ
- システム実行Python: `C:\Python314\python.exe`
- `C:/Program Files/FreeCAD 1.0/bin` は存在
- ただし system Python から `import FreeCAD` すると:
  - `ImportError: Module use of python311.dll conflicts with this version of Python`
- FreeCAD同梱Python (`C:/Program Files/FreeCAD 1.0/bin/python.exe`) では `import FreeCAD` 成功

## 将来バグフィックス向けの記録物
- 統合サマリ(JSON):
  - `output/e2e_20260210_monitoring/e2e_10case_failure_monitoring_summary.json`
- Optuna E2E成果物:
  - `output/e2e_20260210_10cases/`
  - `output/e2e_20260210_10cases_gateoff/`
- Harnessログ:
  - `output/harness/e2e10_20260210/20260211_062158/`
  - `output/harness/e2e10_20260210_gateoff/20260211_062351/`
- VEXIS生ログ:
  - `output/logs/vexis/proto4_trial_1000_vexis.log` - `output/logs/vexis/proto4_trial_1009_vexis.log`

## 次の技術アクション(提案)
1. FreeCAD実行をサブプロセス分離し、Python 3.11系(FreeCAD同梱Python)で geometry 生成する。
2. `geometry_generation` 失敗時に `proto4_failure_reason` をそのまま集計可能なダッシュボード化を追加する。
3. CAE timeout の再現性確認のため、`time_steps` / mesh密度 / timeout閾値をパラメトリックに切るスモーク設定を追加する。

---

## 追記: FreeCADサブプロセス化後の再検証 (実行: 2026-02-11 15:44 - 16:18)

### D. Optuna E2E 10ケース (CAD gate ON, subprocess版)
- コマンド:
  - `python scripts/proto4_harness.py --optimizer temp/e2e10_optimizer_config_20260211_subproc.yaml --limits temp/e2e10_proto4_limitations_20260210.yaml --run-root output/harness/e2e10_after_subproc_20260211 --max-trials 10 --run-proto4 --freecad-bin "C:/Program Files/FreeCAD 1.0/bin"`
- 実行ID: `20260211_154616`
- 出力:
  - `output/e2e_20260211_10cases_subproc/summary_proto4.json`
  - `output/e2e_20260211_10cases_subproc/optuna_study_proto4.db`
  - `output/harness/e2e10_after_subproc_20260211/20260211_154616/logs/proto4_run.log`
  - `output/logs/proto4_20260211_154624.log`
- 結果:
  - 10/10 が `cad_gate` で終了
  - `proto4_failure_reason` は全件 `ml_infeasible`
  - `Generated STEP for trial` は 0件 (CAD gateで停止するため)

### E. Optuna E2E 10ケース (CAD gate OFF, subprocess版)
- コマンド:
  - `python scripts/proto4_harness.py --optimizer temp/e2e10_optimizer_config_20260211_subproc_gateoff.yaml --limits temp/e2e10_proto4_limitations_20260210_gateoff.yaml --run-root output/harness/e2e10_after_subproc_gateoff_20260211 --max-trials 10 --run-proto4 --freecad-bin "C:/Program Files/FreeCAD 1.0/bin"`
- 実行ID: `20260211_154723`
- 出力:
  - `output/e2e_20260211_10cases_subproc_gateoff/summary_proto4.json`
  - `output/e2e_20260211_10cases_subproc_gateoff/optuna_study_proto4.db`
  - `output/harness/e2e10_after_subproc_gateoff_20260211/20260211_154723/logs/proto4_run.log`
  - `output/logs/proto4_20260211_154731.log`
- 結果:
  - `FreeCAD worker run`: 10件
  - `Generated STEP for trial`: 10件
  - `Geometry generation failed`: 0件
  - `Traceback`: 0件
  - `CAE failed (timeout_after_180s)`: 10件
  - VEXISログ( trial 0-9 集計 ):
    - `singular node`: 546件
    - `Mesh generation error summary`: 13件
    - `fatal error`: 0件
    - `error termination`: 0件
  - Optuna user_attrs 集計:
    - `proto4_failure_stage = cae_evaluation`: 10件
    - `proto4_failure_reason = timeout_after_180s`: 10件
  - 値は全 trial で `30.0` (ペナルティで継続)。

### F. 監視観点での結論
- FreeCADサブプロセス化により、以前の `geometry_generation` 失敗 (`FreeCAD not found` / Python ABI不整合) は解消。
- CAD/CAEが失敗する条件でも、Optuna trial は中断せず次 trial を生成し 10ケース完走。
- 残課題は CAE 側の安定化 (`timeout_after_180s` の改善) で、FreeCAD連携自体は継続実行可能な状態。
