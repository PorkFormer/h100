#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$repo/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/tmp/pppo-matched-cache
export FLASHINFER_WORKSPACE_BASE=/tmp/pppo-matched-cache/flashinfer
exec python "$repo/experiments/pppo_ncbr_matched/launch.py" "$@"
