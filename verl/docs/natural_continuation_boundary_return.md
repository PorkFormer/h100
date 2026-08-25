# Natural-continuation boundary return

Last updated: 08/25/2026

This experimental synchronous DAPO path asks what the same sampled trajectory
would score if a response that reached the short budget `H` were allowed to
continue naturally to total response budget `L`. It is a boundary-return
replacement, not a bonus. It does not establish efficacy, a causal mechanism,
or academic novelty.

## Semantics

The feature is disabled by default under
`actor_rollout_ref.rollout.boundary_return.mode=off`. The supported modes are:

- `off`: return before cap detection and add no fields or metrics.
- `shadow`: generate and score continuations, but keep selection, the actor
  batch, every training tensor, row order, and RNG state equal to the baseline.
- `replace`: replace the task-return boundary for cap-hit rows before Dynamic Sampling.
  The resolved Hydra filter metric remains the original correctness
  key; only a local effective metric named `boundary_acc` is used.

For every real cap hit, the continuation request contains exactly the valid
original prompt tokens followed by the valid original response prefix. It adds
no instruction, answer prefix, gold answer, or extra stop. Sampling is `K=1`
with the normal rollout temperature, top-p, top-k, and repetition penalty, and
`max_tokens=L-prefix_length`. Immediate EOS and a zero-token tail are valid.

The verifier scores an independent full-response reward-only batch:

```text
prompt   = original prompt
response = original H-prefix + natural tail
```

All row-aligned verifier input metadata and non-reward meta-info are copied.
Short reward outputs and boundary-internal fields are excluded. Both short and
long rewards must expose explicit, aligned, finite scalar fields for raw
correctness and task score. The default keys are `acc` and `score`. A scalar
reward-manager result with no explicit task-score field fails closed. The
correctness threshold is only for correct/wrong classification and diagnostics;
it never derives task score.

## Replacement formula

Let `original_prefix_shaped` be the sum of the original H-token shaped reward,
and let `short_task` and `long_task` be the explicit task-score fields. For a
cap-hit trajectory:

```text
corrected = original_prefix_shaped + long_task - short_task
```

The exact delta `long_task-short_task` is placed at the last valid original
response token. `token_level_scores` and `token_level_rewards` are then
explicitly synchronized. Long shaped reward is telemetry only and is never read
for training. This preserves the original prefix shaping residual, enforced by
`boundary_return/prefix_penalty_drift_max == 0`.

Original `acc` and `score` fields remain unchanged. In `replace` mode only, the
candidate receives:

- `boundary_acc`: short correctness for non-cap rows, long correctness for cap
  rows;
- `boundary_task_score`: short task score for non-cap rows, long task score for
  cap rows.

In `shadow`, the corresponding arrays live only in the isolated diagnostic
result and never enter the candidate, filter path, retained batch, or actor
batch.

## Ordering and actor boundary

The synchronous order is:

```text
publish(v)
-> normal rollout(v)
-> short reward / acc / task score
-> validate actual policy version
-> natural continuation(v)
-> long reward-only score
-> shadow or replacement handling
-> local effective-metric filter
-> retained accumulation
-> sleep rollout replicas
-> old/ref log-prob
-> canonical GRPO
-> actor update
-> publish(v+1)
```

If the first candidate batch retains too few prompt groups, the next candidate
batch uses the same published policy version. Rollout replicas stay awake; no
weights are updated between batches. Every normal and continuation output is
validated again. Continuation or long-reward failure cancels sibling requests,
sleeps replicas exactly once, and is re-raised before filtering or actor update.

Continuation tails never enter the actor batch. Responses, input IDs, attention
and position IDs, response masks, old/ref log-probabilities, advantages,
returns, loss masks, and the actor token denominator retain H-token semantics.
This is a prefix-only actor even though the verifier sees the full L-budget
response.

## Step-global metrics

Metrics pool every candidate row from every generation batch in the optimizer
step, including rows later filtered out. Counts are summed, means use sample
weights, and p50/p90 are recomputed from all tail lengths rather than averaging
batch quantiles.

- `extra_generated_token_ratio` is all continuation tail tokens divided by all
  valid normal candidate response tokens in the same optimizer step.
- `long_success_rate_given_cap` divides long-correct cap hits by cap hits with a
  valid long score.
- `recovered_rate_given_cap_failure` divides recovered trajectories by cap-hit
  short-wrong trajectories with a valid long score.
- `regressed_rate_given_cap_success` divides regressed trajectories by cap-hit
  short-correct trajectories with a valid long score. H-correct to L-wrong must
  retain its negative task-score delta.
- `unlocked_group_rate` divides short-budget all-wrong UID groups that become
  positive-variance under `boundary_acc` by all short-budget all-wrong UID
  groups. A zero denominator reports zero.

The four transition counts and rates are logged separately:

- H-wrong to L-wrong;
- H-wrong to L-correct;
- H-correct to L-correct;
- H-correct to L-wrong.

Additional counts include candidates, cap hits, valid long scores, recovered
and regressed trajectories, total tail tokens, tail mean/p50/p90, task-score
delta mean/min/max, short-all-wrong groups, and unlocked groups.

## Strict scope and limitations

The active feature requires synchronous GRPO, no critic, vLLM, single-turn
rollout, `ignore_eos=false`, no reward-side KL, `L > H`, and
`max_model_len >= prompt_length + L`. `replace` additionally requires
`filter_groups.enable=true` and the Hydra filter metric to equal the configured
correctness key.

It rejects concurrent forced-answer or FA-TR, FA-CAC/FA-RAR, Probe Credit,
Readiness Dominance, BSSF, OBCF, active rollout correction, distillation,
teacher policy, and configured profiling steps. The first implementation is
strict-only and targets text-only math-verifier workloads; multi-turn/tool and
multimodal reward reconstruction are outside its validated scope.

The formal H2048/L8192 example preserves the verified Qwen3-4B H2048 DAPO
actor recipe and changes only the dedicated entry point, boundary configuration,
experiment name, and `max_model_len` from `1024+2048` to `1024+8192=9216` so the
auxiliary continuation has context. The smoke script is a one-step command
description, not authorization to launch training.
