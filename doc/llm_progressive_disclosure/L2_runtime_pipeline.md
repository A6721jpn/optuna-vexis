# L2 Runtime Pipeline (v1中心)

## 1. Bootstrapping

1. `scripts/run_v1.py` が `src/v1` を動的ロード。
2. `v1.runner.main()` がCLI引数と設定を処理。
3. `config/optimizer_config.yaml` + `config/v1_0_limitations.yaml` を `v1.config.load_config()` で統合。

主要ファイル:
- `scripts/run_v1.py`
- `src/v1/runner.py`
- `src/v1/config.py`

## 2. Search Space と Sampler

1. `v1.search_space.make_constraints_func()` が `trial.user_attrs` の実現性違反値を読む。
2. `v1.search_space.create_sampler()` が sampler を構築。
3. MLゲートが使えるとき `FeasibilityAwareSampler` でリジェクションサンプリング。
4. `suggest_design_point()` が trial から設計点を生成。

主要キー:
- `v1_0_feasibility_violation`（現行）
- `proto4_feasibility_violation`（互換）

主要ファイル:
- `src/v1/search_space.py`
- `src/v1/runner.py`

## 3. Objective 分岐

`v1.objective.ObjectiveOrchestrator` の分岐順:

1. ハード制約 (`check_hard_constraints`)
2. CAD feasibility (`CadGate.predict`)
3. STEP生成 (`GeometryAdapter.generate_step`)
4. CAE評価 (`CaeEvaluator.evaluate`)
5. メトリクスを目的値へ変換

失敗時はペナルティを返し、`trial.user_attrs` に失敗ステージ情報を記録:
- `v1_0_failure_stage`
- `v1_0_failure_reason`

主要ファイル:
- `src/v1/objective.py`
- `src/v1/constraints.py`
- `src/v1/types.py`

## 4. Geometry / CAE 実行

- FreeCAD操作は別プロセス（`src/v1/freecad_worker.py`）で実行。
- 実行Pythonは `FREECAD_PYTHON` / `FREECAD_BIN` / 既定パスの順で解決。
- CAEは `vexis/main.py` 実行とCSV解析を `v1.cae_evaluator` が担当。

主要ファイル:
- `src/v1/geometry_adapter.py`
- `src/v1/freecad_worker.py`
- `src/v1/freecad_engine.py`
- `src/v1/cae_evaluator.py`

## 5. Persistence / Reporting

- trialごとのJSON: `output/trials/trial_<id>/trial_info.json`
- run設定スナップショット: `output/run_config_snapshot.json`
- summary: `output/summary_v1_0.json`
- markdown report: `v1.reporting.generate_markdown_report()`

主要ファイル:
- `src/v1/persistence.py`
- `src/v1/reporting.py`

## 6. v1特有ポイント（proto4との差）

- `constraints_domain: physical|ratio` をサポート（runnerでratio変換）
- `DesignPoint.physical_params` を保持
- `objective.target_values` による `_error` 最適化をサポート

該当:
- `src/v1/config.py`
- `src/v1/runner.py`
- `src/v1/search_space.py`
- `src/v1/types.py`
