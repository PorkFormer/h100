#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$repo/verl"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=offline
exec python -u "$repo/verl/examples/pppo/checkpoint_every_step.py" "${1:-pipeline}"
