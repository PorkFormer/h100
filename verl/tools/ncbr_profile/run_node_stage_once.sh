#!/usr/bin/env bash
set -euo pipefail

# One-shot local controller: validate, start this node's standalone Ray, run one
# fixed stage, flush artifacts, tear down this Ray, attest no target remains.
node="${NCBR_NODE:?set NCBR_NODE to A or B}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
receipt_dir="${NCBR_RECEIPT_DIR:?set NCBR_RECEIPT_DIR}"
stage="${NCBR_STAGE:?set NCBR_STAGE}"
arm="${NCBR_ARM:?set NCBR_ARM}"
candidate="${NCBR_PROFILE_CANDIDATE:?set NCBR_PROFILE_CANDIDATE}"
object_store_bytes="${NCBR_OBJECT_STORE_BYTES:?set the cross-node minimum NCBR_OBJECT_STORE_BYTES}"
num_cpus="${NCBR_RAY_CPUS:?set the common cross-node NCBR_RAY_CPUS}"
if [[ ! "${object_store_bytes}" =~ ^[0-9]+$ || ! "${num_cpus}" =~ ^[0-9]+$ ]]; then
  echo "NCBR_OBJECT_STORE_BYTES and NCBR_RAY_CPUS must be positive integers" >&2
  exit 2
fi
task_key="${stage}_${candidate}_${arm}"
runtime_root="/tmp/qwen17-ncbr-${node}/${task_key}"
mkdir -p "${runtime_root}"
export RAY_ADDRESS="127.0.0.1:$([[ "${node}" == A ]] && printf 6397 || printf 6398)"
export RAY_TMPDIR="${runtime_root}/ray"
export XDG_CACHE_HOME="${runtime_root}/xdg-cache"
export XDG_CONFIG_HOME="${runtime_root}/xdg-config"
export FLASHINFER_WORKSPACE_BASE="${runtime_root}/flashinfer"
export PYTHONPYCACHEPREFIX="${runtime_root}/pycache"
export TORCH_EXTENSIONS_DIR="${runtime_root}/torch-extensions"
export NCBR_ARTIFACT_ROOT="${NCBR_ARTIFACT_ROOT:-/workspace/rl/h100/outputs/qwen3_1p7b_ncbr_profile/node_${node}}"
ulimit -n 524288
mkdir -p "${receipt_dir}"
preflight_json="${receipt_dir}/node_${node}_preflight.json"
preflight_args=(--node "${node}")
if [[ "${stage}" == profile || "${stage}" == acceptance || "${stage}" == formal_s300 ]]; then
  preflight_args+=(--require-wandb)
fi
python "${repo_root}/tools/ncbr_profile/preflight_node.py" "${preflight_args[@]}" >"${preflight_json}"
local_object_store_bytes="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["object_store_bytes"])' "${preflight_json}")"
local_num_cpus="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["ray_cpus"])' "${preflight_json}")"
if (( object_store_bytes < 32 * 1024 * 1024 * 1024 || object_store_bytes > local_object_store_bytes )); then
  echo "cross-node object-store selection is invalid for this node" >&2
  exit 3
fi
if (( num_cpus <= 0 || num_cpus > local_num_cpus )); then
  echo "cross-node Ray CPU selection is invalid for this node" >&2
  exit 3
fi

ray_started=0
cleanup() {
  local stage_exit_code="$?"
  local stop_exit_code="not_run"
  local teardown_exit_code="not_run"
  set +e
  if [[ "${ray_started}" == 1 ]]; then
    ray stop --force 2>&1 | tee "${receipt_dir}/node_${node}_${task_key}_ray_teardown_failure.log"
    stop_exit_code="${PIPESTATUS[0]}"
    python "${repo_root}/tools/ncbr_profile/verify_teardown.py" \
      --node "${node}" \
      --output "${receipt_dir}/node_${node}_${task_key}_teardown_failure.json"
    teardown_exit_code="$?"
  fi
  code_sha="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || printf unavailable)"
  manifest_sha="$(sha256sum "${NCBR_STAGE_MANIFEST}" 2>/dev/null | cut -d ' ' -f 1)"
  printf '{"schema_version":"ncbr-node-stage-receipt-v1","node":"%s","stage":"%s","arm":"%s","candidate":"%s","diagnostics":"%s","code_sha":"%s","manifest_sha256":"%s","stage_exit_code":%s,"ray_stop_exit_code":"%s","teardown_exit_code":"%s","status":"FAIL"}\n' \
    "${node}" "${stage}" "${arm}" "${candidate}" "${NCBR_DIAGNOSTICS_MODE}" "${code_sha}" \
    "${manifest_sha:-unavailable}" "${stage_exit_code}" "${stop_exit_code}" "${teardown_exit_code}" \
    >"${receipt_dir}/node_${node}_${task_key}_stage_receipt.json"
  sync
  trap - EXIT
  if (( stage_exit_code == 0 )); then
    stage_exit_code=1
  fi
  exit "${stage_exit_code}"
}
trap cleanup EXIT
"${repo_root}/tools/ncbr_profile/start_standalone_ray.sh" "${node}" "${object_store_bytes}" "${num_cpus}"
ray_started=1
ray status >"${receipt_dir}/node_${node}_ray_status_before.txt"
python "${repo_root}/tools/ncbr_profile/verify_local_ray.py" \
  --node "${node}" \
  --preflight "${preflight_json}" \
  --output "${receipt_dir}/node_${node}_ray_probe.json"
NCBR_PRINT_COMMAND_ONLY=1 \
  "${repo_root}/examples/natural_continuation_boundary_return/run_qwen3_1p7b_profile_fsdp.sh" \
  >"${receipt_dir}/node_${node}_${task_key}_resolved_command.sh"
{
  NCBR_PRINT_COMMAND_ONLY=1 \
    "${repo_root}/examples/natural_continuation_boundary_return/run_qwen3_1p7b_profile_fsdp.sh" --cfg job
} 2>"${receipt_dir}/node_${node}_${task_key}_resolved_config.stderr" \
  | bash >"${receipt_dir}/node_${node}_${task_key}_resolved_config.yaml"
"${repo_root}/examples/natural_continuation_boundary_return/run_qwen3_1p7b_profile_fsdp.sh" \
  2>&1 | tee "${receipt_dir}/node_${node}_${task_key}.log"
if [[ "${stage}" == acceptance ]]; then
  python "${repo_root}/tools/ncbr_profile/validate_acceptance_log.py" \
    --log "${receipt_dir}/node_${node}_${task_key}.log" \
    --output "${receipt_dir}/node_${node}_${task_key}_validation.json"
  python "${repo_root}/tools/ncbr_profile/validate_checkpoint.py" \
    --checkpoint "${NCBR_ARTIFACT_ROOT}/acceptance/${candidate}/${arm}_s5/checkpoints/global_step_5" \
    --step 5 \
    --world-size 8 \
    --output "${receipt_dir}/node_${node}_${task_key}_checkpoint.json"
fi
sync
ray stop --force 2>&1 | tee "${receipt_dir}/node_${node}_${task_key}_ray_teardown.log"
ray_started=0
trap - EXIT
python "${repo_root}/tools/ncbr_profile/verify_teardown.py" \
  --node "${node}" \
  --output "${receipt_dir}/node_${node}_${task_key}_teardown.json"
code_sha="$(git -C "${repo_root}" rev-parse HEAD)"
manifest_sha="$(sha256sum "${NCBR_STAGE_MANIFEST}" | cut -d ' ' -f 1)"
printf '{"schema_version":"ncbr-node-stage-receipt-v1","node":"%s","stage":"%s","arm":"%s","candidate":"%s","diagnostics":"%s","code_sha":"%s","manifest_sha256":"%s","status":"PASS"}\n' \
  "${node}" "${stage}" "${arm}" "${candidate}" "${NCBR_DIAGNOSTICS_MODE}" "${code_sha}" "${manifest_sha}" \
  >"${receipt_dir}/node_${node}_${task_key}_stage_receipt.json"
