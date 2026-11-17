#!/usr/bin/env bash
# Explicit NCBR full-response 8K length penalty entry; no arguments starts the independent 300-step run.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
  echo 'Usage: bash run_ncbr_lp8k.sh [environment|config|check|smoke|train]' >&2
  exit 2
fi
ACTION="${1:-train}"
case "$ACTION" in
  environment|config|check|smoke|train) ;;
  *) echo "Unsupported action: $ACTION" >&2; exit 2 ;;
esac
exec bash "$ROOT/run.sh" "$ACTION" ncbr_lp8k
