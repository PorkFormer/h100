# BSSF experiment protocol

Implementation base: `origin/main` at `798c84a1a995ee8bb2b7322bf1a79396de4f9667`. Development branch: `feat/budgeted-success-support-floor`. BSSF is independent of `fix/vllm-grouped-probe-branches`, `generate_grouped`, and `vllm_async_server.py`.

## Required gates

1. Build the Base cache and pass cached-logprob validation.
2. Evaluate Base, H2048 step 200, and H4096 step 200 with `evaluate_checkpoint_support.py`.
3. Join the output to the registered retained/delayed/lost and Base-support annotations. Report prompt-paired bootstrap intervals, retained-vs-delayed/lost AUROC, and Spearman association with observed q2048 shift. Base must be near zero shortfall, and H4096 delayed support must be directionally below retained on `S_base_2` before active training.
4. Run the complete CPU suite and config/launcher checks.
5. Run 1-step off, 2-step shadow, 5-step dual, then a 50-step H4096 smoke. Do not begin the 200-step run before these gates pass.

Offline support evaluation:

The optional annotation parquet is keyed by `prompt_key` or `prompt_id` and supplies `transition_group` (`retained`, `delayed`, or `lost`) plus `q2048_shift` (or `observed_q2048_shift`). The cache itself supplies `S_base_1/2/4` membership.

```bash
python tools/success_support_floor/evaluate_checkpoint_support.py \
  --cache-path /path/bssf-cache \
  --checkpoint Base=/path/base \
  --checkpoint H2048=/path/h2048-step200 \
  --checkpoint H4096=/path/h4096-step200 \
  --prompt-annotations /path/transition-annotations.parquet \
  --alpha 0.5 \
  --output /path/support-report.json
```

Shadow and active launchers:

```bash
MODEL_PATH=/path/qwen3-4b \
TRAIN_FILE=/path/train.parquet \
VAL_FILE=/path/val.parquet \
BSSF_CACHE_PATH=/path/bssf-cache \
bash examples/success_support_floor/train_dapo_qwen3_4b_h4096_bssf_smoke.sh

MODEL_PATH=/path/qwen3-4b \
TRAIN_FILE=/path/train.parquet \
VAL_FILE=/path/val.parquet \
BSSF_CACHE_PATH=/path/bssf-cache \
bash examples/success_support_floor/train_dapo_qwen3_4b_h4096_bssf.sh
```

Record the repository commit, cache fingerprint, model/checkpoint hashes, exact commands, environment, GPU-hours, peak allocated/reserved memory, step and actor wall time, added support tokens/ratio, and throughput. Acceptance requires no non-finite values, resumable dual state, no extra rollout/verifier/Base worker, measured smoke overhead at most 4%, stable reward/entropy/grad norm, improved Base-supported retention direction, and no obvious suppression of Base-unsolved full-budget gains.

If \(\lambda\) saturates, inspect cache/logprob convention first, then alpha, then dual learning rate. Do not add length rewards, prefix rewards, rescue replay, global KL, forced continuations, or new algorithm components to repair a failed surrogate gate.
