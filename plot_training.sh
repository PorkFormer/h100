#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$(hostname -s)" in
  th-hpc4-ln0|th-hpc4-ln1)
    if [[ ${1:---help} != --help && ${1:-} != -h ]]; then
      echo 'Run plotting in a CPU allocation: yhrun -N 1 -p debug -n 1 -c 4 bash plot_training.sh ...' >&2
      exit 2
    fi;;
esac
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
exec "${PLOT_PYTHON:-python3}" "$ROOT/analysis/training_plots/cli.py" "$@"
