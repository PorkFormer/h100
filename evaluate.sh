#!/usr/bin/env bash
# A single portable entrypoint; all evaluation semantics are frozen in Python.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CORE="$ROOT/analysis/offline_eval"
[[ ! -d "$ROOT/offline_eval" ]] || CORE="$ROOT/offline_eval"
args=()
while (($#)); do
  case "$1" in
    --container|--python|--apptainer|--squashfuse-bin|--bind)
      [[ $# -ge 2 ]] || { echo "Missing value: $1" >&2; exit 2; }
      case "$1" in
        --container) export EVAL_CONTAINER="$2";;
        --python) export PYTHON_BIN="$2";;
        --apptainer) export EVAL_APPTAINER="$2";;
        --squashfuse-bin) export EVAL_SQUASHFUSE_BIN="$2";;
        --bind) export EVAL_BINDS="${EVAL_BINDS:+$EVAL_BINDS,}$2";;
      esac
      shift 2;;
    *) args+=("$1"); shift;;
  esac
done
host="$(hostname -s)"
management="${EVAL_MANAGEMENT_PYTHON:-python3}"
case "$host" in
  th-hpc4-ln0|th-hpc4-ln1)
    # Site defaults are paths only, never hardware/protocol requirements.
    export EVAL_CONTAINER="${EVAL_CONTAINER:-/fs1/home/zhushoufei/rlvr/images/verl-vllm018-dev1.sif}"
    export EVAL_APPTAINER="${EVAL_APPTAINER:-/fs1/software/apptainer/1.3.6/bin/apptainer}"
    export EVAL_SQUASHFUSE_BIN="${EVAL_SQUASHFUSE_BIN:-/fs1/home/zhushoufei/rlvr/verify_20260916/sqf/bin}"
    management="${EVAL_MANAGEMENT_PYTHON:-/fs1/home/zhushoufei/miniconda3/bin/python3}"
    case "${args[0]:-}" in
      run|check|prepare-data|score|compare|_worker)
        echo 'Use submit for GPU evaluation, or yhrun in a CPU allocation for preparation/scoring.' >&2; exit 2;;
    esac;;
esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
# submit/status/help need only stdlib, including on machines without the ML environment.
case "${args[0]:-}" in
  submit|status|--help|-h|'') exec "$management" "$CORE/cli.py" "${args[@]}";;
esac
if [[ -n "${EVAL_CONTAINER:-}" ]]; then
  [[ -z "${EVAL_SQUASHFUSE_BIN:-}" ]] || export PATH="$EVAL_SQUASHFUSE_BIN:$PATH"
  ap="${EVAL_APPTAINER:-apptainer}"
  options=(exec --bind "$ROOT:$ROOT")
  if [[ -n "${EVAL_BINDS:-}" ]]; then options+=(--bind "$EVAL_BINDS")
  elif [[ -d /fs1 ]]; then options+=(--bind /fs1:/fs1); fi
  case "${args[0]}" in run|_worker) options+=(--nv);; esac
  exec "$ap" "${options[@]}" "$EVAL_CONTAINER" "${PYTHON_BIN:-python3}" "$CORE/cli.py" "${args[@]}"
fi
exec "${PYTHON_BIN:-python3}" "$CORE/cli.py" "${args[@]}"
