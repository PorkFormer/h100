#!/usr/bin/env bash
set -euo pipefail

# Qwen3-1.7B Baseline/NCBR-v1 launcher.  GPU execution requires a separately
# generated stage manifest pinned to the final code SHA and model revision.
arm="${NCBR_ARM:?set NCBR_ARM to baseline or v1}"
candidate="${NCBR_PROFILE_CANDIDATE:?set NCBR_PROFILE_CANDIDATE to P0, P1, or P2}"
stage="${NCBR_STAGE:?set NCBR_STAGE to calibration, gate0, profile, mechanism_panel, fixed_replay, acceptance, or formal_s300}"
stage_manifest="${NCBR_STAGE_MANIFEST:?set NCBR_STAGE_MANIFEST to the approved manifest path}"
node="${NCBR_NODE:?set NCBR_NODE to A or B}"
diagnostics_mode="${NCBR_DIAGNOSTICS_MODE:?set NCBR_DIAGNOSTICS_MODE to on or off}"
ray_address="${RAY_ADDRESS:?set RAY_ADDRESS to the approved shared Ray GCS address}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${repo_root}"

case "${node}" in
  A) ray_node_resource=ncbr_node_A ;;
  B) ray_node_resource=ncbr_node_B ;;
  *)
    echo "unsupported NCBR_NODE=${node}" >&2
    exit 2
    ;;
esac

task_key="${stage}_${candidate}_${arm}_diag_${diagnostics_mode}"
runtime_root="${NCBR_RUNTIME_ROOT:-/tmp/q17ncbr/${node}/${task_key}}"
export XDG_CACHE_HOME="${runtime_root}/xdg-cache"
export XDG_CONFIG_HOME="${runtime_root}/xdg-config"
export FLASHINFER_WORKSPACE_BASE="${runtime_root}/flashinfer"
export PYTHONPYCACHEPREFIX="${runtime_root}/pycache"
export TORCH_EXTENSIONS_DIR="${runtime_root}/torch-extensions"
export PYTHONPATH="${repo_root}"
mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${PYTHONPYCACHEPREFIX}" \
  "${TORCH_EXTENSIONS_DIR}"

python "${repo_root}/tools/ncbr_profile/validate_stage_manifest.py" \
  --manifest "${stage_manifest}" \
  --repo "${repo_root}" \
  --arm "${arm}" \
  --candidate "${candidate}" \
  --stage "${stage}" \
  --node "${node}" \
  --diagnostics "${diagnostics_mode}" >&2

case "${diagnostics_mode}" in
  on) diagnostics_enable=true ;;
  off) diagnostics_enable=false ;;
  *)
    echo "unsupported NCBR_DIAGNOSTICS_MODE=${diagnostics_mode}" >&2
    exit 2
    ;;
esac

mapfile -t manifest_paths < <(
  python -c \
    'import json,sys; m=json.load(open(sys.argv[1], encoding="utf-8")); print(m["model"]["local_path"]); print(m["data"]["train"]["path"]); print(m["data"]["AIME2024"]["path"]); print(m["data"]["AIME2025"]["path"])' \
    "${stage_manifest}"
)
if [[ "${#manifest_paths[@]}" -ne 4 ]]; then
  echo "stage manifest did not resolve exactly four model/data paths" >&2
  exit 2
fi
model_path="${manifest_paths[0]}"
train_data="${manifest_paths[1]}"
aime2024_data="${manifest_paths[2]}"
aime2025_data="${manifest_paths[3]}"

case "${arm}" in
  baseline)
    boundary_mode=off
    experiment_stem=qwen3_1p7b_base_dapo_ctx9216_b256_g768_m16_n8_h2048
    ;;
  v1)
    boundary_mode=replace
    experiment_stem=qwen3_1p7b_base_ncbr_v1_b256_g768_m16_n8_h2048_l8192
    ;;
  *)
    echo "unsupported NCBR_ARM=${arm}" >&2
    exit 2
    ;;
esac

case "${candidate}" in
  P0)
    tp=1
    gpu_memory_utilization=0.45
    max_num_seqs=256
    max_num_batched_tokens=32768
    rollout_logprob_micro=2
    ref_logprob_micro=2
    optimizer_offload=true
    ref_param_offload=true
    ;;
  P1)
    tp=1
    gpu_memory_utilization=0.55
    max_num_seqs=384
    max_num_batched_tokens=49152
    rollout_logprob_micro=4
    ref_logprob_micro=4
    optimizer_offload=false
    ref_param_offload=false
    ;;
  P2)
    tp=2
    gpu_memory_utilization=0.50
    max_num_seqs=256
    max_num_batched_tokens=32768
    rollout_logprob_micro=2
    ref_logprob_micro=2
    optimizer_offload=true
    ref_param_offload=true
    ;;
  *)
    echo "unsupported NCBR_PROFILE_CANDIDATE=${candidate}" >&2
    exit 2
    ;;
esac

artifact_root="${NCBR_ARTIFACT_ROOT:-/workspace/rl/h100/outputs/qwen3_1p7b_ncbr_profile}"
case "${stage}" in
  calibration)
    if [[ "${arm}" != baseline || "${candidate}" != P0 ]]; then
      echo "node calibration is fixed to baseline/P0" >&2
      exit 2
    fi
    total_steps=1
    val_before_train=false
    test_freq=-1
    save_freq=-1
    gate_cycles=0
    logger='["console"]'
    run_dir="${artifact_root}/calibration/node_${node}/baseline_P0_diag_${diagnostics_mode}"
    ;;
  gate0)
    total_steps=1
    val_before_train=false
    test_freq=-1
    save_freq=-1
    gate_cycles=3
    logger='["console"]'
    run_dir="${artifact_root}/gate0/${arm}_${candidate}"
    ;;
  profile)
    total_steps=6
    profile_coordination_dir="${NCBR_PROFILE_COORDINATION_DIR:?set a fresh shared profile coordination directory}"
    val_before_train=false
    test_freq=-1
    save_freq=-1
    gate_cycles=0
    logger='["console","wandb"]'
    run_dir="${artifact_root}/profile/${candidate}/${arm}_s4_or_s6_diag_${diagnostics_mode}"
    ;;
  mechanism_panel)
    if [[ "${arm}" != v1 ]]; then
      echo "the mechanism panel is restricted to the V1 arm" >&2
      exit 2
    fi
    mechanism_panel_path="$(
      python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["frozen_artifacts"]["hard_prefix_panel"]["path"])' \
        "${stage_manifest}"
    )"
    total_steps=1
    val_before_train=false
    test_freq=-1
    save_freq=-1
    gate_cycles=0
    logger='["console"]'
    run_dir="${artifact_root}/mechanism_panel/${candidate}/node_${node}/v1"
    ;;
  fixed_replay)
    if [[ "${diagnostics_mode}" != on ]]; then
      echo "fixed replay internally alternates diagnostics off/on and requires NCBR_DIAGNOSTICS_MODE=on" >&2
      exit 2
    fi
    total_steps=1
    val_before_train=false
    test_freq=-1
    save_freq=-1
    gate_cycles=0
    logger='["console"]'
    run_dir="${artifact_root}/fixed_replay/${candidate}/${arm}"
    ;;
  acceptance)
    total_steps=5
    val_before_train=true
    test_freq=5
    save_freq=5
    gate_cycles=0
    logger='["console","wandb"]'
    run_dir="${artifact_root}/acceptance/${candidate}/${arm}_s5"
    ;;
  formal_s300)
    if [[ "${NCBR_AUTHORIZE_S300:-}" != AUTHORIZE_S300 ]]; then
      echo "formal S300 is not authorized; set the explicit approval token only after user approval" >&2
      exit 3
    fi
    total_steps=300
    val_before_train=true
    test_freq=10
    save_freq=50
    gate_cycles=0
    logger='["console","wandb"]'
    export WANDB_RUN_ID="${NCBR_WANDB_RUN_ID:?set an arm-specific W&B run ID for formal S300}"
    export WANDB_RESUME=never
    if [[ "${arm}" == baseline ]]; then
      run_dir=/workspace/rl/h100/outputs/qwen3_1p7b_base_dapo_ctx9216_b256_g768_m16_n8_h2048_s300_seed42_v1
    else
      run_dir=/workspace/rl/h100/outputs/qwen3_1p7b_base_ncbr_v1_b256_g768_m16_n8_h2048_l8192_s300_seed42_v1
    fi
    ;;
  *)
    echo "unsupported NCBR_STAGE=${stage}" >&2
    exit 2
    ;;
esac

mkdir -p "${run_dir}/profiling" "${run_dir}/receipts"
export NCBR_PROFILE_INTERVALS_PATH="${run_dir}/profiling/intervals.jsonl"
export WANDB_DIR="${run_dir}/wandb"
mkdir -p "${WANDB_DIR}"

command=(
  python -m verl.experimental.natural_continuation_boundary_return.main_dapo_boundary_return
  "data.train_files=${train_data}"
  "data.val_files=['${aime2024_data}','${aime2025_data}']"
  data.prompt_key=prompt
  data.max_prompt_length=1024
  data.max_response_length=2048
  data.filter_overlong_prompts=true
  data.truncation=error
  data.train_batch_size=256
  +data.gen_batch_size=768
  data.seed=42
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=true
  algorithm.use_kl_in_reward=false
  algorithm.kl_ctrl.kl_coef=0
  algorithm.filter_groups.enable=true
  algorithm.filter_groups.metric=acc
  algorithm.filter_groups.max_num_gen_batches=10
  algorithm.probe_credit.enable=false
  algorithm.censor_aware_advantage.enable=false
  algorithm.readiness_dominance.mode=off
  algorithm.success_support_floor.mode=off
  algorithm.on_policy_budgeted_capability_floor.mode=off
  algorithm.rollout_correction.rollout_is=null
  algorithm.rollout_correction.rollout_rs=null
  algorithm.rollout_correction.bypass_mode=false
  critic.enable=false
  "actor_rollout_ref.model.path=${model_path}"
  +actor_rollout_ref.model.override_config.model_type=qwen3
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.data_loader_seed=42
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.kl_loss_coef=0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.calculate_entropy=false
  "actor_rollout_ref.actor.diagnostics.enable=${diagnostics_enable}"
  actor_rollout_ref.actor.diagnostics.numerical_log_ratio_min=-20
  actor_rollout_ref.actor.diagnostics.numerical_log_ratio_max=20
  actor_rollout_ref.actor.diagnostics.entropy_bucket_size=256
  actor_rollout_ref.actor.clip_ratio_low=0.20
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.clip_ratio_c=3.0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload}"
  actor_rollout_ref.actor.fsdp_config.seed=42
  'actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]'
  'actor_rollout_ref.actor.checkpoint.load_contents=["model","optimizer","extra"]'
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${tp}"
  actor_rollout_ref.rollout.max_model_len=9216
  "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
  "actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}"
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.free_cache_engine=true
  actor_rollout_ref.rollout.enforce_eager=false
  actor_rollout_ref.rollout.disable_log_stats=false
  actor_rollout_ref.rollout.ignore_eos=false
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_logprob_micro}"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42
  actor_rollout_ref.rollout.forced_answer_probe.enable=false
  actor_rollout_ref.rollout.forced_answer_probe.training_credit.enable=false
  "actor_rollout_ref.rollout.boundary_return.mode=${boundary_mode}"
  actor_rollout_ref.rollout.boundary_return.long_response_length=8192
  actor_rollout_ref.rollout.boundary_return.correctness_key=acc
  actor_rollout_ref.rollout.boundary_return.correctness_threshold=0.5
  actor_rollout_ref.rollout.boundary_return.task_score_key=score
  actor_rollout_ref.rollout.boundary_return.max_concurrent_requests=128
  actor_rollout_ref.rollout.boundary_return.request_batch_size=512
  actor_rollout_ref.rollout.boundary_return.request_timeout_seconds=600
  actor_rollout_ref.rollout.boundary_return.long_reward_chunk_size=256
  actor_rollout_ref.rollout.boundary_return.seed=42
  actor_rollout_ref.rollout.boundary_return.strict=true
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_logprob_micro}"
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
  "actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload}"
  actor_rollout_ref.ref.fsdp_config.seed=42
  reward_model.reward_manager=dapo
  reward_model.enable=false
  reward_model.model.path=null
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=true
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=410
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=false
  +reward_model.reward_kwargs.max_resp_len=2048
  distillation.enabled=false
  "trainer.logger=${logger}"
  trainer.project_name=a100_verl
  "trainer.experiment_name=${experiment_stem}_${stage}_${candidate}_diag_${diagnostics_mode}_seed42_v1"
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  "+trainer.ray_node_resource=${ray_node_resource}"
  "trainer.val_before_train=${val_before_train}"
  "trainer.test_freq=${test_freq}"
  "trainer.save_freq=${save_freq}"
  trainer.total_epochs=1
  "trainer.total_training_steps=${total_steps}"
  "trainer.dynamic_sampling_gate_cycles=${gate_cycles}"
  "trainer.dynamic_sampling_gate_receipt_path=${run_dir}/receipts/dynamic_sampling_gate.jsonl"
  "trainer.profile_interval_path=${run_dir}/profiling/intervals.jsonl"
  "trainer.profile_coordination_dir=${profile_coordination_dir:-null}"
  "trainer.profile_arm=${arm}"
  "trainer.profile_candidate=${candidate}"
  "trainer.profile_diagnostics_mode=${diagnostics_mode}"
  trainer.profile_min_steps=4
  trainer.profile_max_steps=6
  trainer.profile_cv_threshold=0.10
  trainer.profile_coordination_timeout_seconds=900
  trainer.hard_prefix_source_path=null
  trainer.diagnostic_dump_dir=null
  trainer.nondeterminism_diagnostics.enabled=false
  trainer.nondeterminism_diagnostics.output_dir=null
  trainer.nondeterminism_diagnostics.run_id=null
  trainer.actor_fixed_replay.enabled=false
  trainer.actor_fixed_replay.repeats=2
  trainer.actor_fixed_replay.receipt_dir=null
  trainer.mechanism_panel_path=null
  trainer.mechanism_panel_receipt_path=null
  trainer.mechanism_rows_path=null
  "trainer.default_local_dir=${run_dir}/checkpoints"
  trainer.resume_mode=disable
  "+ray_kwargs.ray_init.address=${ray_address}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
  "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR=${WANDB_DIR}"
)

if [[ "${stage}" == calibration ]]; then
  command+=(
    "trainer.hard_prefix_source_path=${run_dir}/mechanism/h2048_cap_source.jsonl"
    "trainer.diagnostic_dump_dir=${run_dir}/actor_batch"
    trainer.nondeterminism_diagnostics.enabled=true
    "trainer.nondeterminism_diagnostics.output_dir=${run_dir}/workload"
    "trainer.nondeterminism_diagnostics.run_id=calibration_node_${node}"
  )
fi

if [[ "${stage}" == fixed_replay || "${stage}" == calibration ]]; then
  command+=(
    trainer.actor_fixed_replay.enabled=true
    trainer.actor_fixed_replay.repeats=2
    "trainer.actor_fixed_replay.receipt_dir=${run_dir}/receipts/fixed_actor_replay"
  )
fi

if [[ "${stage}" == mechanism_panel ]]; then
  command+=(
    "trainer.mechanism_panel_path=${mechanism_panel_path}"
    "trainer.mechanism_panel_receipt_path=${run_dir}/receipts/mechanism_panel.json"
  )
fi

if [[ "${arm}" == v1 && "${diagnostics_mode}" == on && "${stage}" =~ ^(gate0|profile)$ ]]; then
  command+=("trainer.mechanism_rows_path=${run_dir}/receipts/candidate_mechanism_rows.jsonl")
fi

if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  command+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RUN_ID=${WANDB_RUN_ID}")
fi
if [[ -n "${WANDB_RESUME:-}" ]]; then
  command+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RESUME=${WANDB_RESUME}")
fi

if [[ "${NCBR_PRINT_COMMAND_ONLY:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}" "$@"
  printf '\n'
  exit 0
fi

exec "${command[@]}" "$@"
