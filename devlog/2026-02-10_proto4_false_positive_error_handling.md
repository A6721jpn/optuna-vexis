# 作業ログ: CAD/CAEクラッシュ時の継続最適化設計 (2026-02-10)

## 背景
CAD gate の False-positive により CAD/CAE がクラッシュしても、Optuna が停止せず次の特徴量候補を生成し続けるように、失敗分類と trial 属性の設計を見直した。

## 実施内容

### 1. CAE失敗理由の構造化
- `src/proto4-codex/types.py`
  - `CaeResult` に `failure_reason` を追加。
  - 永続化 (`to_dict`) で `failure_reason` を保存可能にした。

### 2. CAE実行系の失敗分類を強化
- `src/proto4-codex/cae_evaluator.py`
  - 失敗時に `failure_reason` を返すように実装。
  - 主な分類:
    - `timeout_after_<N>s`
    - `process_exit_<code>`
    - `execution_error:<ExceptionClass>`
    - `result_csv_missing`
    - `fatal_error_output_and_result_missing`
    - `result_load_failed:<ExceptionClass>`
    - `metric_computation_failed`
  - trial再実行時の誤判定防止として、同一 `job_name` の stale CSV を実行前に削除。
  - `max_retries` が 0 以下でも最低1回は実行するガードを追加。

### 3. Objectiveの可行性学習信号を修正
- `src/proto4-codex/objective.py`
  - これまで: CAE実行前に `proto4_feasibility_violation=-1.0` を設定していたため、CAE失敗でも可行扱いに学習される可能性があった。
  - 修正後:
    - CAE成功時のみ `-1.0` (可行)
    - CAE失敗時は `failure_reason` に応じて `0.8〜1.0` の違反スコアを設定
  - 失敗の説明責務を強化:
    - `proto4_failure_stage`
    - `proto4_failure_reason`
    を `trial.user_attrs` に保存。

### 4. 回帰テストの追加
- `tests/proto4_codex/test_objective_error_handling.py` を新規追加。
  - ケース1: CAE失敗時に feasibility が正値で記録されること。
  - ケース2: 1 trial目失敗後も Optuna が 2 trial目へ進み、成功 trial を記録できること。

## テスト結果
- `pytest tests/proto4_codex -q` -> 7 passed
- `pytest tests/proto4/test_proto4_codex_readiness.py -q` -> 3 passed

## 効果
- CAD gate の False-positive が後段でクラッシュを起こしても、
  - trial は失敗として記録され、
  - サンプラに「危険領域」の学習信号が返り、
  - 最適化ループは継続して次の特徴量候補を生成できる。

## 残課題
- FreeCAD本体クラッシュなどで Python プロセス自体が異常終了するケースは、同一プロセス内 try/except では捕捉できない。
- 必要に応じて `GeometryAdapter` のサブプロセス分離を次段で検討する。
