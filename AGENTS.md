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
