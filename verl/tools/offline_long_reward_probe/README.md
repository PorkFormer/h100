# Offline Long-Rollout Reward Probe

This is an offline counterfactual long-rollout reward probe for saved VERL checkpoints. It samples long responses from a fixed checkpoint and compares reward behavior across token prefixes, especially 2048-token prefixes versus full 10240-token responses.

It does not train a model, update parameters, restore optimizer state, or modify the PPO trainer. It is a fixed-checkpoint counterfactual probe, not exact replay of training rollouts.

The standard runtime is the `verlai/verl:vllm018.dev1` container. Enter the container yourself, change to the VERL repository root, and run the tool from there.

Prompt selection defaults to filtering prompts longer than `data.max_prompt_length` before applying `prompt_start` and `num_prompts`. Set `data.filter_overlong_before_slice: false` only if you want the older behavior of slicing first and then dropping overlong prompts.

For small pilot runs, `data.shuffle: true` samples from a deterministic seeded shuffle before filtering and slicing. `original_index` still records the source parquet row index for traceability.

## Sanity Checks

Before launching long rollouts, you can check Python syntax and YAML parsing:

```bash
python -m py_compile \
  tools/offline_long_reward_probe/offline_probe.py \
  tools/offline_long_reward_probe/forced_answer_probe.py

python - <<'PY'
import yaml
p = "tools/offline_long_reward_probe/probe_config.yaml"
with open(p) as f:
    cfg = yaml.safe_load(f)
print(cfg.keys())
print(cfg["rollout"])
print(cfg["forced_answer"])
PY

pytest -q \
  tests/tools/test_offline_long_reward_probe.py \
  tests/tools/test_forced_answer_probe.py
```

## Minimal Test

```bash
cd /workspace/verl

python tools/offline_long_reward_probe/offline_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 \
  --mode all \
  --num-prompts 8 \
  --force
```

## Pilot Run

```bash
python tools/offline_long_reward_probe/offline_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 \
  --mode all \
  --num-prompts 512 \
  --force
```

`storage.skip_existing: true` is useful for resuming, but it can reuse stale parquet files after you change scoring, reward functions, prefix lengths, prompt filtering, or shuffle settings. Add `--force` when rerunning a stage after code or config changes.

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
  --checkpoint-path /path/to/actor_hf \
  --force
```

Outputs are written under `output_dir`:

- `rollouts/step_0400/rollouts.parquet`
- `scored/step_0400/scored.parquet`
- `analysis/step_0400/*.csv`
- `examples/step_0400/*.jsonl`
- per-stage `metadata.json`

`analysis/step_0400/stop_reason_summary.csv` groups trajectories by raw `finish_reason`, raw `stop_reason`, and a coarse stop category: before-2048 stop, stop after 2048 before the 10240 cap, cap hit, or other stop reason.

## Forced-answer prefix probe

`forced_answer_probe.py` is a second, independent offline stage. The original
prefix scan decodes a response prefix and asks the reward function whether that
prefix is already correct. The forced-answer stage instead takes the exact saved
token sequence

```text
prompt_token_ids + response_token_ids[:h] + cue_token_ids
```

and samples a new short completion. Only that short completion is sent to the
original reward function. The prompt, reasoning prefix, and cue are never sent
to the verifier. The tokenized input is passed directly to vLLM; it is not
decoded, re-tokenized, wrapped in a chat template, or given extra roles.

This distinction matters: `V(h)` estimates whether a fixed checkpoint can
produce a verifiable answer after seeing its own first `h` reasoning tokens. It
does not say that the truncated reasoning text itself is a valid answer.

The checkpoint is fixed for the whole generate stage. This tool does not resume
training state, update weights, modify PPO/GRPO, restore an optimizer, or change
the existing prefix-scan outputs. It reuses an existing
`scored/step_XXXX/scored.parquet`, including its trajectory token IDs, terminal
reward, ground truth, extra info, and reward-function configuration.

### Configuration and cost

The `forced_answer` block in `probe_config.yaml` defines fixed horizons,
preterminal tail offsets, ordered cues, sampling parameters, and the stability
threshold used by taxonomy. The first cue is the primary cue for taxonomy and
counterfactual-credit (HC) analysis. Cue names and texts must be unique.

For a response shorter than a fixed horizon, the terminal reward is carried
forward without a model call. Preterminal points are always strictly before the
response endpoint. A request that would exceed `rollout.max_model_len` is
recorded as `context_overflow` and is excluded from the answer and stage-error
denominators.

The approximate number of short completions is:

```text
selected trajectories × non-terminal probe points × cues × forced_answer.n
```

This can still be expensive for 512 trajectories. Start with the smoke command,
then increase `--max-trajectories` and `--probe-n`. `rollout` capacity settings,
including tensor parallelism, memory utilization, maximum batched tokens, and
prefix caching, are reused. Generation requests are additionally limited by
`forced_answer.batch_size_requests`.

### VIE smoke and 512-trajectory runs

Run these inside the same `verlai/verl:vllm018.dev1` VIE environment used by the
base probe, from the VERL repository root. Supply the fixed checkpoint and the
already-produced scored parquet explicitly when they are not at the config
defaults:

```bash
python tools/offline_long_reward_probe/forced_answer_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 \
  --mode all \
  --checkpoint-path /path/to/global_step_400/actor_hf \
  --source-scored /path/to/scored/step_0400/scored.parquet \
  --max-trajectories 8 \
  --probe-n 1 \
  --force
```

After that succeeds, run the 512-trajectory probe:

```bash
python tools/offline_long_reward_probe/forced_answer_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 \
  --mode all \
  --checkpoint-path /path/to/global_step_400/actor_hf \
  --source-scored /path/to/scored/step_0400/scored.parquet \
  --max-trajectories 512 \
  --force
```

There is intentionally no `probe_config_vie.yaml` in this tree. Use the real
paths for the VIE host through the CLI instead of copying an assumed machine
configuration into the repository.

### Stages, resume, and outputs

The stages can be run separately:

```bash
python tools/offline_long_reward_probe/forced_answer_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 --mode generate

python tools/offline_long_reward_probe/forced_answer_probe.py \
  --config tools/offline_long_reward_probe/probe_config.yaml \
  --step 400 --mode analyze
```

`storage.skip_existing: true` skips an existing stage before loading the
tokenizer or model. Use `--force` after changing cues, horizons, sampling,
scoring, source data, or code. Generate writes raw parquet and metadata before
raising if its equivalent-branch error rate exceeds the fixed 5% threshold, so
failures remain inspectable. Analyze is CPU-only and does not import vLLM or
Transformers or load a checkpoint.

Forced-answer outputs are written under `output_dir/forced_answer`:

- `raw/step_0400/raw.parquet` and `metadata.json`
- `analysis/step_0400/probe_curve.csv`
- `analysis/step_0400/trajectory_probe_values.parquet`
- `analysis/step_0400/taxonomy.parquet` and `taxonomy_summary.csv`
- `analysis/step_0400/counterfactual_advantage.parquet`
- `analysis/step_0400/advantage_summary.csv` and `metadata.json`
- `examples/step_0400/<taxonomy>.jsonl`

Text and token-ID columns remain present but contain null when their save switch
is disabled. Manual-review JSONL uses `response_text`, then `response_full`; if
neither exists it retains response token IDs and records a null response text.
Analysis never loads a tokenizer just to reconstruct missing text.

### Reading the analysis

`probe_curve.csv` reports binary `V(h)`, reward mean/std, valid denominators,
prompt-cluster standard error, and trajectory-level overflow/error rates for
each horizon, point kind, and cue. For an apples-to-apples r2048/r10240
comparison, keep the same checkpoint, primary cue, `forced_answer.seed`,
sampling parameters, source parquet, and selected row order, then compare the
`fixed` rows where `horizon` is 2048 and 10240. Do not compare runs with changed
cues or seeds as if the difference came only from horizon.

Taxonomy labels are nullable diagnostics, not ground-truth causal labels.
Missing points, overflow, or generation/scoring errors make labels unknown when
the required evidence is unavailable. In particular,
`terminal_only_candidate` is a candidate and `unstable_diagnostic` is a curve
diagnostic. `taxonomy_summary.csv` therefore reports true, false, and unknown
counts as well as the rate among known cases.

HC credit uses only the primary cue and complete adjacent fixed-horizon
intervals. It never bridges a missing point. Both interval HC values and
terminal rewards are standardized with population standard deviation in their
respective prompt groups; degenerate groups get zero advantage and are marked
as such. Sign disagreement includes zero-versus-nonzero cases. Summary rows are
provided overall and for `<=2048`, `2049-4096`, `4097-8192`, `8193-10240`, and
`>10240` response-length buckets, with prompt-cluster standard errors for rates.
