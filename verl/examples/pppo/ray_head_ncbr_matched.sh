#!/usr/bin/env bash
# Run on the already allocated NCBR head. Never stops an existing Ray service.
set -euo pipefail
[[ $(hostname -s) == p-kt-dgx-a100-03 ]] || { echo 'Expected allocated NCBR head p-kt-dgx-a100-03' >&2; exit 1; }
export RAY_USAGE_STATS_ENABLED=0
if timeout 15 ray status --address=10.8.191.127:6395; then
  echo 'Existing head retained. Run cluster_preflight.py to attest both nodes.'
  exit 0
fi
# Refuse to create a second head if any local Ray service owns the host.
if pgrep -x raylet >/dev/null || pgrep -x gcs_server >/dev/null; then
  echo 'Existing Ray processes detected but head is unavailable; inspect, do not automatically stop them.' >&2
  exit 1
fi
exec ray start --head --node-ip-address=10.8.191.127 --port=6395 \
  --num-cpus=240 --num-gpus=8 --resources='{"pppo_head":1}' \
  --object-store-memory=200000000000 --temp-dir=/tmp/rh16g \
  --object-manager-port=7011 --node-manager-port=7012 \
  --min-worker-port=22000 --max-worker-port=22999 \
  --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=52366 \
  --runtime-env-agent-port=52367 --metrics-export-port=52368 \
  --include-dashboard=false --disable-usage-stats --block
