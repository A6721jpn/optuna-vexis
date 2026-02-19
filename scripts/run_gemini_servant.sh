#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPECTED_GEMINI="/usr/local/bin/gemini"
SETTINGS_PATH="${PROJECT_ROOT}/.gemini/settings.json"

print_usage() {
  cat <<'EOF'
Usage:
  scripts/run_gemini_servant.sh --prompt "text"
  scripts/run_gemini_servant.sh --prompt-file path/to/prompt.txt
  scripts/run_gemini_servant.sh --check
  echo "task" | scripts/run_gemini_servant.sh

Environment:
  GEMINI_MODEL        Default: gemini-2.5-pro
  GEMINI_EXTRA_ARGS   Extra args appended to gemini command
EOF
}

sanitize_path() {
  local original_path="${PATH:-}"
  local -a parts cleaned
  IFS=':' read -r -a parts <<< "${original_path}"
  cleaned=()

  for part in "${parts[@]}"; do
    if [[ "${part}" =~ ^/mnt/c/Users/.*/AppData/Roaming/npm/?$ ]]; then
      continue
    fi
    if [[ -z "${part}" ]]; then
      continue
    fi
    cleaned+=("${part}")
  done

  PATH="/usr/local/bin"
  for part in "${cleaned[@]}"; do
    if [[ "${part}" == "/usr/local/bin" ]]; then
      continue
    fi
    PATH="${PATH}:${part}"
  done
  export PATH
}

check_auth_config() {
  if [[ ! -f "${SETTINGS_PATH}" ]]; then
    echo "ERROR: Missing ${SETTINGS_PATH}" >&2
    return 1
  fi

  local auth_type
  auth_type="$(python3 - <<'PY' "${SETTINGS_PATH}"
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
auth = data.get("security", {}).get("auth", {})
print(auth.get("selectedType", ""))
PY
)"

  if [[ "${auth_type}" != "oauth-personal" ]]; then
    echo "ERROR: selectedType must be oauth-personal in ${SETTINGS_PATH}" >&2
    return 1
  fi
}

build_prompt() {
  local user_prompt="$1"
  cat <<EOF
You are a coding-only servant.
Do not call any tools.
Do not run shell commands.
Respond directly in plain text.
Return actionable code output for Codex.

${user_prompt}
EOF
}

main() {
  local check_only=0
  local prompt_input=""
  local mode=""

  while (($# > 0)); do
    case "$1" in
      --prompt)
        mode="inline"
        shift
        if (($# == 0)); then
          echo "ERROR: --prompt requires a value" >&2
          return 1
        fi
        prompt_input="$1"
        ;;
      --prompt-file)
        mode="file"
        shift
        if (($# == 0)); then
          echo "ERROR: --prompt-file requires a file path" >&2
          return 1
        fi
        if [[ ! -f "$1" ]]; then
          echo "ERROR: Prompt file not found: $1" >&2
          return 1
        fi
        prompt_input="$(cat "$1")"
        ;;
      --check)
        check_only=1
        ;;
      -h|--help)
        print_usage
        return 0
        ;;
      *)
        echo "ERROR: Unknown option: $1" >&2
        print_usage
        return 1
        ;;
    esac
    shift || true
  done

  if [[ "${mode}" == "" && "${check_only}" -eq 0 ]]; then
    if [[ -t 0 ]]; then
      echo "ERROR: Provide --prompt, --prompt-file, or stdin" >&2
      print_usage
      return 1
    fi
    prompt_input="$(cat)"
  fi

  cd "${PROJECT_ROOT}"
  sanitize_path

  local gemini_bin
  gemini_bin="$(command -v gemini || true)"
  if [[ "${gemini_bin}" != "${EXPECTED_GEMINI}" ]]; then
    echo "ERROR: gemini resolves to '${gemini_bin}' (expected ${EXPECTED_GEMINI})" >&2
    return 1
  fi

  check_auth_config

  mapfile -t gemini_all < <(which -a gemini | awk '!seen[$0]++')
  if ((${#gemini_all[@]} != 1)); then
    echo "ERROR: gemini must resolve to a single binary. Current candidates:" >&2
    printf '  %s\n' "${gemini_all[@]}" >&2
    return 1
  fi

  if [[ "${check_only}" -eq 1 ]]; then
    echo "OK: gemini path and auth config are pinned."
    echo "gemini=${gemini_bin}"
    echo "settings=${SETTINGS_PATH}"
    return 0
  fi

  local final_prompt
  final_prompt="$(build_prompt "${prompt_input}")"

  local -a cmd
  cmd=(
    "${gemini_bin}"
    "--model" "${GEMINI_MODEL:-gemini-2.5-pro}"
    "--output-format" "text"
    "--prompt" "${final_prompt}"
  )

  if [[ -n "${GEMINI_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    cmd+=(${GEMINI_EXTRA_ARGS})
  fi

  "${cmd[@]}"
}

main "$@"
