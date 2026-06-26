# Offline Long-Rollout Reward Probe

This is an offline counterfactual long-rollout reward probe for saved VERL checkpoints. It samples long responses from a fixed checkpoint and compares reward behavior across token prefixes, especially 2048-token prefixes versus full 10240-token responses.

It does not train a model, update parameters, restore optimizer state, or modify the PPO trainer. It is a fixed-checkpoint counterfactual probe, not exact replay of training rollouts.

The standard runtime is the `verlai/verl:vllm018.dev1` container. Enter the container yourself, change to the VERL repository root, and run the tool from there.

## Minimal Test

```bash
cd /workspace/verl

python tools/offline_long_reward_probe/offline_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 \
  --mode all \
  --num-prompts 8
```

## Pilot Run

```bash
python tools/offline_long_reward_probe/offline_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 \
  --mode all \
  --num-prompts 512
```

## Multiple Checkpoints

Run checkpoints manually, for example:

```bash
for s in 0 100 200 300 400 500; do
  python tools/offline_long_reward_probe/offline_probe.py \
    --config tools/offline_long_reward_probe/probe_config.yaml \
    --step "$s" \
    --mode all \
    --num-prompts 512
done
```

Useful overrides:

```bash
python tools/offline_long_reward_probe/offline_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 \
  --mode rollout \
  --num-prompts 8 \
  --output-dir /path/to/out \
  --checkpoint-path /path/to/actor_hf
```

Outputs are written under `output_dir`:

- `rollouts/step_0400/rollouts.parquet`
- `scored/step_0400/scored.parquet`
- `analysis/step_0400/*.csv`
- `examples/step_0400/*.jsonl`
- per-stage `metadata.json`
