---
name: gemini-coding-servant
description: Delegate coding-only implementation tasks from Codex to Gemini CLI in headless mode while keeping Codex as the primary orchestrator. Use for fast draft patches, refactors, bug-fix proposals, and unit-test generation when interactive Gemini sessions are unnecessary.
---

# Gemini Coding Servant

Keep Codex as the owner of scope, quality bar, and final edits.
Use Gemini CLI only as a coding subroutine.

## Runtime Defaults

- Keep `gemini` resolved to `/usr/local/bin/gemini` only.
- Keep auth pinned to `oauth-personal` via `.gemini/settings.json`.

## Enforce Boundaries

Delegate only coding work:
- Implement or refactor code
- Propose unified diffs
- Draft or fix tests
- Suggest runnable commands

Do not delegate:
- Final architectural decisions
- Security or policy decisions
- Product prioritization
- Any non-coding communication to the user

## Build a Tight Handoff Prompt

Pass a compact, explicit brief. Keep it deterministic.

```text
You are a coding-only servant.
Do not call any tools.
Do not run shell commands.
Respond directly in plain text.
Return actionable code output for Codex.

Task:
<one concrete coding task>

Repository context:
<target files, relevant APIs, constraints>

Acceptance criteria:
<what must pass>

Output contract:
1) Summary (max 5 bullets)
2) Unified diff only (git-style, with file paths)
3) Test commands to run
4) Risks / assumptions (max 5 bullets)

Rules:
- Do not ask for broad redesign.
- If context is missing, ask at most 2 precise questions.
- Prefer minimal, reversible edits.
```

## Run Gemini in Headless Mode

Use the project wrapper so path and auth checks are always enforced.

```bash
PROMPT="$(cat <<'EOF'
<handoff prompt here>
EOF
)"

scripts/run_gemini_servant.sh --prompt "$PROMPT"
```

Use fully non-interactive execution only when tool approvals would block automation:

```bash
GEMINI_EXTRA_ARGS="--sandbox --approval-mode yolo" \
scripts/run_gemini_servant.sh --prompt "$PROMPT"
```

## Merge Discipline for Codex

After Gemini returns:
1. Validate that output follows the contract.
2. Apply only relevant parts of the diff.
3. Run project tests or linters.
4. Fix integration issues in Codex.
5. Report final result from Codex, not from Gemini.

## Failure Handling

If Gemini output is noisy or off-target:
1. Tighten scope to one file or one function.
2. Reduce acceptance criteria to objective checks.
3. Re-run with stricter output contract.
4. Fall back to direct Codex implementation when iteration cost exceeds benefit.
