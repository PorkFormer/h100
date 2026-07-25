#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-8B checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the DAPO math training parquet}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet}"

python -m verl.experimental.probe_credit.main_dapo_readiness_dominance \
  --config-name=readiness_dominance_dapo_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=2 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.load_format=auto \
  algorithm.filter_groups.enable=true \
  algorithm.filter_groups.metric=acc \
  algorithm.filter_groups.max_num_gen_batches=4 \
  algorithm.readiness_dominance.mode=shadow \
  algorithm.readiness_dominance.n=2 \
  algorithm.readiness_dominance.absolute_horizons='[512,1024,2048]' \
  algorithm.readiness_dominance.max_tokens=32 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.experiment_name=readiness-dominance-shadow-smoke \
  trainer.logger='["console"]'
