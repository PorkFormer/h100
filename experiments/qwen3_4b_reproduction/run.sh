#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_FILE="${SETTINGS_FILE:-$ROOT/settings.env}"
if [[ ! -f "$SETTINGS_FILE" ]]; then
  echo "Missing settings: $SETTINGS_FILE" >&2; exit 2
fi
set -a
source "$SETTINGS_FILE"
set +a
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Run bash $ROOT/setup_env.sh first, or set PYTHON_BIN in settings.env." >&2
  exit 2
fi
if [[ -z "${VERL_ROOT:-}" ]]; then
  if [[ -d "$ROOT/verl/verl" ]]; then VERL_ROOT="$ROOT/verl"
  else VERL_ROOT="$(cd "$ROOT/../../verl" && pwd)"; fi
fi
export VERL_ROOT
export PYTHONPATH="$VERL_ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export RAY_USAGE_STATS_ENABLED=0
# Do not inherit a historical cluster, W&B identity, or machine-specific NIC binding.
unset RAY_ADDRESS WANDB_RUN_ID WANDB_RESUME WANDB_ENTITY WANDB_API_KEY WANDB_BASE_URL
unset NCCL_SOCKET_IFNAME NCCL_IB_HCA GLOO_SOCKET_IFNAME
export WANDB_MODE=offline
mkdir -p "$CACHE_ROOT"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export FLASHINFER_WORKSPACE_BASE="$CACHE_ROOT/flashinfer"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
exec "$PYTHON_BIN" "$ROOT/run.py" "$@"
