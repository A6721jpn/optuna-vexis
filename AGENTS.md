  ## Workspace write policy (mandatory)

  - Scope:
    - Writable: `C:\github_repo\optuna-for-vexis` and all descendants.
    - Read-only: any other directory under `C:\github_repo`.
  - Rules:
    - The agent may read files anywhere under `C:\github_repo`.
    - The agent must never create, modify, or delete files outside `C:\github_repo\optuna-for-vexis`.
    - If a task would require writing outside the writable scope, the agent must stop and ask the user first.