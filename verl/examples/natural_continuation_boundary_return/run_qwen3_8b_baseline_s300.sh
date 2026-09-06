#!/usr/bin/env bash
set -euo pipefail
export NCBR_ARM=baseline NCBR_STAGE=formal_s300
exec "$(cd "$(dirname "$0")" && pwd)/run_qwen3_8b_profile_fsdp.sh" "$@"
