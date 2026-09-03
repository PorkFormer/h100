#!/usr/bin/env bash
set -euo pipefail

node="${1:?usage: start_standalone_ray.sh A|B OBJECT_STORE_BYTES NUM_CPUS}"
object_store_bytes="${2:?missing object store bytes}"
num_cpus="${3:?missing Ray CPU count}"
ulimit -n 524288

case "${node}" in
  A)
    gcs=6397
    object_port=7111
    node_port=7112
    worker_min=22000
    worker_max=22511
    dashboard=8267
    metrics=9087
    ;;
  B)
    gcs=6398
    object_port=7211
    node_port=7212
    worker_min=23000
    worker_max=23511
    dashboard=8268
    metrics=9088
    ;;
  *)
    echo "node must be A or B" >&2
    exit 2
    ;;
esac
ray_tmp="${RAY_TMPDIR:?set a task-specific RAY_TMPDIR}"

if pgrep -f 'raylet|gcs_server|dashboard.py' >/dev/null; then
  echo "refusing to start: an existing Ray process is present" >&2
  exit 3
fi
mkdir -p "${ray_tmp}"
exec ray start --head \
  --port="${gcs}" \
  --object-manager-port="${object_port}" \
  --node-manager-port="${node_port}" \
  --min-worker-port="${worker_min}" \
  --max-worker-port="${worker_max}" \
  --dashboard-host=127.0.0.1 \
  --dashboard-port="${dashboard}" \
  --metrics-export-port="${metrics}" \
  --num-gpus=8 \
  --num-cpus="${num_cpus}" \
  --object-store-memory="${object_store_bytes}" \
  --temp-dir="${ray_tmp}" \
  --disable-usage-stats
