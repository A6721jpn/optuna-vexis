# v1.1 E2E実行手順（実設計ファイル最適化）

## 0. 目的
- Production v1.1 の最適化パイプラインで、実際の設計ファイル（FCStd）を使って E2E（FreeCAD -> STEP -> VEXIS/CAE -> Optuna）を実行する。

## 1. 前提
- 実行場所: リポジトリルート `C:\github_repo\optuna-for-vexis`
- Python仮想環境: `.venv`
- FreeCAD 1.0 が利用可能（`FREECAD_PYTHON` または既定パス）
- VEXIS が `vexis/` 配下で実行可能
- `input/model.FCStd` が最新設計を指している

## 2. 設定ファイル
- Optimizer: `config/optimizer_config.yaml`
- Limits: `config/v1_0_limitations.yaml`

v1.1 では Limits 側で以下を使える:
- `freecad.constraints_domain: "physical"` なら `constraints` の `min/max` を実寸で記述可能
- CAD gate を使う場合: `cad_gate.enabled: true`

## 3. 実行前チェック
1. 設計ファイルと拘束名を確認
- `freecad.fcstd_path`
- `freecad.sketch_name`
- `freecad.constraints` の20拘束名

2. 出力先の衝突を避ける
- `config/optimizer_config.yaml` の `paths.result_dir` を新規フォルダ名に変更
- 例: `output/e2e_v1_1_YYYYMMDD_HHMMSS`

3. 試行回数を決める
- 本番探索: `optimization.max_trials` を設定
- 単発確認: CLI の `-n` で上書き

## 4. 実行コマンド
最短実行（既定config使用）:

```powershell
.venv\Scripts\python.exe scripts/run_v1.py -n 10
```

明示実行（必要時のみ）:

```powershell
.venv\Scripts\python.exe scripts/run_v1.py --config config/optimizer_config.yaml --limits config/v1_0_limitations.yaml --max-trials 10
```

## 5. 実行中に見るログ
- メインログ: `output/.../logs/v1_0_*.log`
- VEXIS標準出力ログ: `output/logs/vexis/`

## 6. 結果確認
実行後、`paths.result_dir` 配下を確認:
- `report_v1_0.md`（サマリと各trial）
- `summary_v1_0.json`（開始/終了、試行数）
- `run_config_snapshot.json`（実行時設定スナップショット）
- `optuna_study_v1_0.db`（Study DB）

## 7. 再実行時の注意
- 同じ `result_dir` だと既存Studyを継続し、完了trial数によっては即スキップされる
- 完全新規で回す場合は `result_dir` を毎回変える

## 8. よくあるトラブル
- FreeCAD起動失敗:
  - `FREECAD_PYTHON` を正しい `python.exe` に設定
- CAD gateモデル読み込み失敗:
  - `cad_gate.model_path` と依存（joblib/scikit-learn）を確認
- CAE失敗が多い:
  - `constraints` 範囲を狭める、`cad_gate` を有効化、試行回数を増やす
