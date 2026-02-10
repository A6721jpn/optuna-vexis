# Proto4 Codex 修正計画（準備のみ）

## 目的

`src/proto4-codex` の実装にはまだ手を入れず、`doc/proto4.md` の仕様と
レビュー結果に基づく修正計画・受け入れ条件・検証方針を定義する。

## 対象範囲

- 対象パッケージ: `src/proto4-codex`
- このフェーズでは実装の挙動は変更しない
- 追加するのは準備成果物のみ:
  - 修正計画ドキュメント（本ファイル）
  - 実装前は失敗する readiness テスト

## 次フェーズで修正する項目

1. Feasibility attr キーの不一致
- 現在: `cad_feasibility_violation`
- 仕様: `proto4_feasibility_violation`
- 影響: `constraints_func` が意図した user_attr を読めない

2. independent 経路での拒否サンプリングが無効
- `FeasibilityAwareSampler.sample_independent` が素通り委譲
- 影響: 独立サンプル主体のサンプラで早期排除が効かない

3. `dry_run` の意味づけが不正
- 現在: `CAD_INFEASIBLE` 扱いでペナルティ
- 影響: dry-run の診断と infeasible の区別が崩れる

## 実装計画（次フェーズ）

1. Feasibility attr の定数・参照箇所を仕様に揃える
2. independent 経路にも拒否サンプリングを実装
3. `dry_run` の扱いを `CAD_INFEASIBLE` から切り離す
4. 各修正に対応したユニットテストを追加/調整
5. 重点テスト後、proto4 の関連テストを実行

## 受け入れ条件

- `FEASIBILITY_ATTR` が `"proto4_feasibility_violation"` になる
- Feasibility-aware sampling が independent 経路でも判定する
- `dry_run` で `TrialOutcome.CAD_INFEASIBLE` を設定しない
- 新規 readiness テストが実装後に通る

## 検証方針

- 実装前: readiness テストは失敗する
- 実装後: readiness テストは成功する
- 回帰: `tests/proto4` の関連テストが引き続き通る
