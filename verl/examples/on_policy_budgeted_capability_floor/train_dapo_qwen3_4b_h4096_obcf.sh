#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-4B checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the DAPO math training parquet}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet}"
: "${OBCF_CACHE_PATH:?Set OBCF_CACHE_PATH to a validated capability floor cache}"

python -m verl.experimental.on_policy_budgeted_capability_floor.main_dapo_obcf \
  --config-name=on_policy_budgeted_capability_floor_dapo_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.load_format=auto \
  algorithm.use_kl_in_reward=false \
  algorithm.filter_groups.enable=true \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=0 \
  algorithm.on_policy_budgeted_capability_floor.mode=dual \
  algorithm.on_policy_budgeted_capability_floor.cache_path="${OBCF_CACHE_PATH}" \
  algorithm.on_policy_budgeted_capability_floor.reference_budget=2048 \
  algorithm.on_policy_budgeted_capability_floor.base_rollouts_per_prompt=8 \
  algorithm.on_policy_budgeted_capability_floor.support_threshold=2 \
  algorithm.on_policy_budgeted_capability_floor.reference_tolerance_count=1 \
  algorithm.on_policy_budgeted_capability_floor.delta=0.05 \
  algorithm.on_policy_budgeted_capability_floor.update_interval=1 \
  algorithm.on_policy_budgeted_capability_floor.lambda_init=0.0 \
  algorithm.on_policy_budgeted_capability_floor.lambda_max=10.0 \
  algorithm.on_policy_budgeted_capability_floor.dual_lr=0.01 \
  algorithm.on_policy_budgeted_capability_floor.dual_ema_beta=0.9 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps=200 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.experiment_name=obcf-dual-h4096 \
  trainer.logger='["console","wandb"]'
