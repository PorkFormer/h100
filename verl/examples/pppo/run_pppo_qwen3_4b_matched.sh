#!/bin/bash
set -euo pipefail
host=$(hostname -s)
if [[ "$host" == th-hpc4-ln0 || "$host" == th-hpc4-ln1 || -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Submit through yhbatch/yhrun on HPC4; do not run training on login nodes.' >&2
  exit 1
fi
: "${1:?Pass a frozen resolved YAML config}"
python /fs1/home/zhushoufei/rlvr/experiments/pppo_4b_20261108/launch.py "$@"
