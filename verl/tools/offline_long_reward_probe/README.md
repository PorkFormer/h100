# Offline Long-Rollout Reward Probe

This is an offline counterfactual long-rollout reward probe for saved VERL checkpoints. It samples long responses from a fixed checkpoint and compares reward behavior across token prefixes, especially 2048-token prefixes versus full 10240-token responses.

It does not train a model, update parameters, restore optimizer state, or modify the PPO trainer. It is a fixed-checkpoint counterfactual probe, not exact replay of training rollouts.

The standard runtime is the `verlai/verl:vllm018.dev1` container. Enter the container yourself, change to the VERL repository root, and run the tool from there.

Prompt selection defaults to filtering prompts longer than `data.max_prompt_length` before applying `prompt_start` and `num_prompts`. Set `data.filter_overlong_before_slice: false` only if you want the older behavior of slicing first and then dropping overlong prompts.

## Sanity Checks

Before launching long rollouts, you can check Python syntax and YAML parsing:

```bash
python -m py_compile tools/offline_long_reward_probe/offline_probe.py

python - <<'PY'
import yaml
p = "tools/offline_long_reward_probe/probe_config.yaml"
with open(p) as f:
    cfg = yaml.safe_load(f)
print(cfg.keys())
print(cfg["rollout"])
PY
```

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

For 10240-token rollouts, start conservatively if you see OOMs. In `probe_config.yaml`, lower `rollout.batch_size_prompts` to `8` or `16` and set `rollout.max_num_seqs` around `64` before increasing throughput.

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

`analysis/step_0400/stop_reason_summary.csv` groups trajectories by raw `finish_reason`, raw `stop_reason`, and a coarse stop category: before-2048 stop, stop after 2048 before the 10240 cap, cap hit, or other stop reason.
