#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
worktree_root=$(cd -- "${script_dir}/../../.." && pwd)
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_name=${FA_RAR_RUN_NAME:-qwen3_8b_dapo_r2048_fa_rar_smoke_s4_${run_stamp}}
run_root=${FA_RAR_RUN_ROOT:-${worktree_root}/analysis/fa_rar_smoke/runs/${run_name}}
gcs_port=${FA_RAR_GCS_PORT:-16385}
ray_namespace=${FA_RAR_RAY_NAMESPACE:-fa_rar_smoke_${run_stamp}}
model_path=${FA_RAR_MODEL_PATH:-/workspace/models/Qwen3-8B-Base}

mkdir -p "${run_root}/logs" "${run_root}/provenance" "${run_root}/checkpoints" "${run_root}/ray"
export PYTHONPATH="${worktree_root}:${worktree_root}/verl:/workspace/rl/verl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX="${run_root}/pycache"
export FLASHINFER_WORKSPACE_BASE="${run_root}/flashinfer"
export WANDB_DIR="${run_root}/wandb"
export FA_CAC_TARGET_NODE_IP=10.8.191.129
export FA_RAR_PRIVATE_RAY_TEMP_ROOT="${run_root}/ray"
export FA_RAR_RAY_NAMESPACE="${ray_namespace}"
export RAY_GCS_SERVER_PORT="${gcs_port}"
unset RAY_ADDRESS RAY_REDIS_ADDRESS

launcher="${worktree_root}/analysis/fa_cac_v2/tools/matched_dapo_main.py"
resolved_config="${run_root}/provenance/resolved_config.yaml"
config_gate_log="${run_root}/provenance/resolved_config_gate.txt"

overrides=(
  data.train_files=/workspace/rl/data/dapo_math_17k_train.parquet
  'data.val_files=[/workspace/rl/data/aime-2024-verl.parquet]'
  data.prompt_key=prompt
  data.max_prompt_length=1024
  data.max_response_length=2048
  data.filter_overlong_prompts=true
  data.truncation=error
  data.shuffle=true
  data.seed=42
  data.train_batch_size=2
  data.gen_batch_size=2
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=true
  algorithm.use_kl_in_reward=false
  algorithm.kl_ctrl.kl_coef=0.0
  algorithm.filter_groups.enable=true
  algorithm.filter_groups.metric=acc
  algorithm.filter_groups.max_num_gen_batches=10
  algorithm.probe_credit.enable=false
  critic.enable=false
  actor_rollout_ref.model.path="${model_path}"
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.ppo_mini_batch_size=8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.data_loader_seed=42
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio_low=0.20
  actor_rollout_ref.actor.clip_ratio_high=0.28
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.actor.fsdp_config.seed=42
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=4
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.response_length=2048
  actor_rollout_ref.rollout.max_model_len=4096
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.free_cache_engine=true
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.forced_answer_probe.enable=true
  actor_rollout_ref.rollout.forced_answer_probe.num_samples=2
  actor_rollout_ref.rollout.forced_answer_probe.max_new_tokens=64
  actor_rollout_ref.rollout.forced_answer_probe.temperature=1.0
  actor_rollout_ref.rollout.forced_answer_probe.top_p=1.0
  actor_rollout_ref.rollout.forced_answer_probe.correctness_key=acc
  actor_rollout_ref.rollout.forced_answer_probe.correctness_threshold=0.5
  actor_rollout_ref.rollout.forced_answer_probe.training_credit.enable=false
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  actor_rollout_ref.ref.fsdp_config.seed=42
  reward.reward_manager.name=dapo
  reward.reward_kwargs.overlong_buffer_cfg.enable=true
  reward.reward_kwargs.overlong_buffer_cfg.len=410
  reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
  reward.reward_kwargs.overlong_buffer_cfg.log=false
  reward.reward_kwargs.max_resp_len=2048
  'trainer.logger=[console,wandb]'
  trainer.project_name=h100_verl
  trainer.experiment_name="${run_name}"
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  trainer.val_before_train=false
  trainer.test_freq=-1
  trainer.save_freq=-1
  trainer.total_epochs=1
  trainer.total_training_steps=4
  trainer.default_local_dir="${run_root}/checkpoints"
  trainer.resume_mode=disable
  algorithm.censor_aware_advantage._target_=verl.trainer.config.CensorAwareAdvantageConfig
  algorithm.censor_aware_advantage.enable=true
  algorithm.censor_aware_advantage.apply=true
  algorithm.censor_aware_advantage.mode=reliability_redistribution
)

printf '%q ' /usr/bin/python3 "${launcher}" "${overrides[@]}" >"${run_root}/provenance/command.txt"
printf '\n' >>"${run_root}/provenance/command.txt"

CUDA_VISIBLE_DEVICES='' /usr/bin/python3 "${launcher}" "${overrides[@]}" --cfg job --resolve \
  | tee "${resolved_config}"
/usr/bin/python3 "${script_dir}/assert_resolved_config.py" "${resolved_config}" \
  --run-name "${run_name}" --output-dir "${run_root}/checkpoints" \
  | tee "${config_gate_log}"

if [[ ${FA_RAR_CONFIG_ONLY:-0} == 1 ]]; then
  echo "FA-RAR config-only gate complete: ${run_root}"
  exit 0
fi

/usr/bin/python3 "${script_dir}/private_gpu_gate.py" \
  --port "${gcs_port}" --output "${run_root}/provenance/private_gpu_gate.json"

cd "${worktree_root}"
/usr/bin/python3 "${launcher}" "${overrides[@]}" 2>&1 | tee "${run_root}/logs/train.log"
