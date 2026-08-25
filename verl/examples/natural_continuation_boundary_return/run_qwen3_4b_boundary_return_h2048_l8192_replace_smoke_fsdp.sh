#!/usr/bin/env bash
set -euo pipefail

# One-step protocol smoke only. This command is documentation; run it only after
# the normal GPU/Ray/W&B authorization and capacity gates have passed.
exec "$(dirname "$0")/run_qwen3_4b_boundary_return_h2048_l8192_replace_fsdp.sh" \
  data.train_batch_size=2 \
  +data.gen_batch_size=4 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  'trainer.logger=["console"]' \
  trainer.experiment_name=qwen3_4b_boundary_return_h2048_l8192_replace_smoke
