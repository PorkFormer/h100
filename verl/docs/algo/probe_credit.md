# On-Policy Probe Credit Redistribution

This experimental trainer retains official DAPO Dynamic Sampling and changes only the temporal distribution of the standard GRPO advantage. It supports synchronous single-turn GRPO with vLLM, no critic, and a synchronous verifier. The first version rejects KL-in-reward, active rollout correction, distillation/teacher policy, and configured profiling steps. The vLLM async engine remains supported because optimizer updates themselves are synchronous. The shared `RayPPOTrainer.fit()` is unchanged.

## Training order

At each pre-update policy version, the trainer generates and scores candidate groups, applies DAPO's positive-standard-deviation filter, accumulates complete groups, and selects exactly `data.train_batch_size` prompts with every `rollout.n` trajectory. Only that final retained batch is probed. Probe generation finishes before replicas sleep, old/reference log probabilities are computed, and the actor updates.

The Dynamic Sampling block comes from `verl-project/verl-recipe:dapo/dapo_ray_trainer.py@e477deff97b15d067f8b4e71f75b80ae58ad64c4`. Compatibility edits use the current rollout manager, `checkpoint_manager.update_weights(global_step)`, reward extraction, logging, and current actor APIs. Its metric, `np.std(metric_values) > 0` selection, singleton compatibility, accumulation, complete-group truncation, and `max_num_gen_batches` behavior are preserved.

## Immediate Answer Prefix Probe

For raw prompt IDs `p`, response IDs `y`, and `h=floor(qL)`, the input is exactly:

```text
p + y[:h] + encode("\n\nAnswer:", add_special_tokens=false)
```

No decoding/re-encoding, chat template, new turn, suffix, or manual EOS is used. The candidate is the first non-empty generated line and the configured verifier receives `Answer: {candidate}`. One grouped vLLM request uses one stable seed and returns `n` reproducible samples from that request RNG stream; these are not described as independent branch seeds. Context overflow, missing branches, and scoring errors fail closed when `strict=true`.

The first implementation accepts only `relative_positions=[0.0,0.25,0.5,0.75,0.9]`, `probe_zero_position=true`, and `strict=true`. Unsupported protocol variants fail during trainer startup, before any Probe request is generated.

The driver submits Probe requests in chunks of `request_batch_size` (default 512) and permits at most `max_concurrent_requests` (default 128) active grouped RPCs. This bounds driver tasks, Ray RPC pressure, and vLLM queue bursts without changing request seeds or sampling parameters.

## Credit equations

For `q=[0,.25,.50,.75,.90]`, local progress and backward returns are

```text
rP[k] = V[k+1] - V[k]
GP[k] = sum(rho**(j-k) * rP[j] for j=k..3)
```

Each segment is centered and optionally sample-standard-deviation normalized within its prompt group using `verl.utils.group_mean_std`. Singleton or near-zero-std group-segments are exactly zero.

The four segment advantages map to `[0,.25L)`, `[.25L,.50L)`, `[.50L,.75L)`, and `[.75L,.90L)`. A length-weighted per-trajectory mean is subtracted. The 90–100% tail and padding remain exactly zero, while the masked raw correction sum is approximately zero. This is raw advantage-mass conservation only; it does not claim invariance after PPO ratios, clipping, minibatching, or loss aggregation.

The trainer saves standard GRPO output as `terminal_advantages`, then sets both `advantages` and `returns` to `terminal_advantages + coef * probe_correction`. Terminal verifier output, `token_level_scores`, and `token_level_rewards` are unchanged. With `enable=false`, no Probe requests or seeds are consumed. With `coef=0`, final advantages equal the terminal baseline.

## Metrics and smoke use

Metrics include configured request concurrency/chunk size, grouped request/branch counts, Probe input/output token totals and output-length mean/max, position-wise `V(q)`, pseudo rewards, temporal returns, degenerate rate, correction magnitude, zero-mass residual, and terminal/correction/final advantage statistics. Dynamic Sampling generation counts, accepted/filtered prompt groups, trajectories, response tokens, and timing are accumulated over every generation batch in one optimizer step. Normal generation, Probe generation/scoring, advantage work, actor update, weight publication, and total step timers remain separate.

The launcher under `examples/probe_credit/` requires an explicit positive `PROBE_CREDIT_COEF`. It invokes Python directly and is intended for a one-step engineering smoke only after CPU tests pass. It does not allocate resources or submit a training job by itself.
