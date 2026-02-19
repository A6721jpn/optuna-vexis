  ## Workspace write policy (mandatory)

  - Scope:
    - Writable: `C:\github_repo\optuna-for-vexis` and all descendants.
    - Read-only: any other directory under `C:\github_repo`.
  - Rules:
    - The agent may read files anywhere under `C:\github_repo`.
    - The agent must never create, modify, or delete files outside `C:\github_repo\optuna-for-vexis`.
    - If a task would require writing outside the writable scope, the agent must stop and ask the user first.

## UTF-8 display policy (mandatory)

- Always treat text files as UTF-8 unless the user explicitly requests another encoding.
- In PowerShell sessions, prefer UTF-8-safe commands:
  - Read: `Get-Content -Encoding UTF8`
  - Write: `Set-Content -Encoding UTF8`
  - Append: `Add-Content -Encoding UTF8`
- If console output is mojibake, switch the session to UTF-8 before displaying text:
  - `chcp 65001`
  - `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`
  - `$OutputEncoding = [System.Text.Encoding]::UTF8`
- Do not overwrite Japanese comments/strings only to avoid mojibake. Fix the display/output encoding path first.

## Codex progressive disclosure entrypoint (mandatory)

- Purpose:
  - This file is loaded every session. Use it to keep initial context small and still preserve architecture understanding.
- Entrypoint:
  - Start from `doc/llm_progressive_disclosure/README.md`.
- Layered loading rule:
  - Open `L0_project_snapshot.md` first.
  - Open only one additional layer based on the task:
    - Routing: `L1_task_router.md`
    - Runtime/dataflow debugging: `L2_runtime_pipeline.md`
    - Cross-line/module differences: `L3_module_deep_dive.md`
  - Do not bulk-read all layers by default.
- Token-saving read scope:
  - Default implementation line is `src/v1`.
  - Read only files explicitly pointed by L1/L2/L3 for the active task.
  - Avoid loading unrelated history/log areas unless required by the task:
    - `src/proto1`, `src/proto2`, `src/proto3`
    - `devlog/`
    - `debug_*.py`, `debug_output*.txt`
- Edit-time behavior:
  - Before code edits, identify the selected layer and target files.
  - Prefer narrow, task-scoped reads and edits over repository-wide scans.
