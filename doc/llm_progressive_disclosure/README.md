# Codex Progressive Disclosure Guide

このフォルダは、次回以降のLLMセッションで「最小の読み込み」から開始し、必要時だけ情報を追加するための入口です。

## Start Protocol

1. まず `L0_project_snapshot.md` だけ読む。
2. 実施タスクを1つに絞る。
3. `L1_task_router.md` で必要ファイルだけ特定する。
4. それでも不足するときだけ `L2_runtime_pipeline.md`、`L3_module_deep_dive.md` を読む。
5. 無関係な履歴ドキュメントや古いprotoは読まない。

## Layer Map

| Layer | File | 目的 | 開くタイミング |
| --- | --- | --- | --- |
| L0 | `L0_project_snapshot.md` | 全体像を1枚で把握 | 毎回最初 |
| L1 | `L1_task_router.md` | タスク別の最短読込ルート | 作業対象を決める時 |
| L2 | `L2_runtime_pipeline.md` | 実行フローと責務境界 | バグ原因を切り分ける時 |
| L3 | `L3_module_deep_dive.md` | v1/proto4差分とテスト境界 | 仕様差分を扱う時 |

## Repo Default Scope

- 現行ライン: `src/v1`（実運用向け）
- 実行入口: `scripts/run_v1.py`

## Optional Entry-File Pattern

- Codex向け (`AGENTS.md`): 概要だけを短く書き、このフォルダへのリンクを置く。

## Best-Practice Rules Applied Here

- 最上位ファイルは短く保ち、詳細はリンク先で段階開示する。
- ドキュメントは用途ごとに分離し、重複を減らす。
- 1タスク1コンテキストで作業し、必要なファイルだけ読む。
- 「Issueのように具体的な依頼」でLLMに渡す。
- 長時間の調査は分離して行い、主コンテキストを圧迫しない。

## Source Research (checked: 2026-02-16)

- OpenAI Cookbook: Harnessing tools for agentic workflows  
  https://cookbook.openai.com/examples/partners/model_selection_guide/model_selection_guide
- OpenAI Cookbook: Codex Prompting Guide (Codex primer)  
  https://cookbook.openai.com/examples/gpt-5/codex_prompting_guide
- OpenAI for Business: How OpenAI uses Codex  
  https://openai.com/index/how-openai-uses-codex-to-improve-coding-velocity/
