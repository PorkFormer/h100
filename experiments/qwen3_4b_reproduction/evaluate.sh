#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${SETTINGS_FILE:-$ROOT/settings.env}"
source "${EVAL_SETTINGS_FILE:-$ROOT/eval_settings.env}"
set +a
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ -z "${VERL_ROOT:-}" ]]; then
  if [[ -d "$ROOT/verl/verl" ]]; then VERL_ROOT="$ROOT/verl"
  else VERL_ROOT="$(cd "$ROOT/../../verl" && pwd)"; fi
fi
export VERL_ROOT PYTHONPATH="$VERL_ROOT" PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON_BIN" "$ROOT/evaluate.py" "$@"
