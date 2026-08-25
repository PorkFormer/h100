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
- `shadow`: generate and score continuations. Given the same normal candidate
  batches, its candidate `DataProto`, filter inputs and selection, actor batch,
  training tensors, row order, and driver RNG state are no-ops relative to
  `off`. This conditional gate does not claim bitwise equality of subsequent
  real-vLLM rollouts across separate launches, because auxiliary requests can
  perturb backend scheduling.
- `replace`: replace the task-return boundary for cap-hit rows before Dynamic Sampling.
  The resolved Hydra filter metric remains the original correctness
  key; only a local effective metric named `boundary_acc` is used.

For every real cap hit, the continuation request contains exactly the valid
original prompt tokens followed by the valid original response prefix. It adds
no instruction, answer prefix, gold answer, or extra stop. Sampling is `K=1`
and uses the same sampling-parameter builder as normal rollout, with
`max_tokens=L-prefix_length`. Immediate EOS and a zero-token tail are valid only
when the backend reports a legal EOS/stop/completed terminal state. Abort,
error, cancelled, timeout, or missing terminal states fail closed, as does a
tail longer than its request budget.

The verifier scores an independent full-response reward-only batch:

```text
prompt   = original prompt
response = original H-prefix + natural tail
```

All row-aligned verifier input metadata and non-reward meta-info are copied.
Short reward outputs and boundary-internal fields are excluded. Long scoring is
chunked (256 rows by default), and only the explicit scalar fields survive each
chunk; long shaped reward is discarded. Both short and
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
validated again. Continuation or long-reward failure remotely aborts active
sibling vLLM requests, waits for abort acknowledgements and server drain, then
awaits load-balancer release before settling local tasks and sleeping replicas.
Cleanup errors are attached to, and never replace, the primary exception. If
remote drain/release cannot be attested, replicas are not slept.

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
- `unlocked_group_rate` uses actual per-UID standard deviation: it divides UID
  groups with short-return std at or below the shared numeric tolerance that
  become positive-std under `boundary_acc` by all short-locked groups.
  `newly_locked_group_rate` records the reverse transition. Separate
  all-wrong-unlocked counts/rates retain the original all-wrong diagnostic.

The four transition counts and rates are logged separately:

- H-wrong to L-wrong;
- H-wrong to L-correct;
- H-correct to L-correct;
- H-correct to L-wrong.

Additional counts include candidates, cap hits, valid long scores, recovered
and regressed trajectories, total tail tokens, tail mean/p50/p90, task-score
delta mean/min/max, short-all-wrong groups, unlocked/newly-locked groups,
continuation request timeouts, input-token totals/mean/max, and long-budget cap
hits. `request_timeout_seconds` defaults to 600.

For a same-batch real-vLLM shadow gate, set
`boundary_return.verify_shadow_candidate_noop=true` and require
`boundary_return/shadow_candidate_noop_gate_pass_count` to equal the number of
normal candidate batches (at least two for the smoke gate). This compares each
real normal candidate batch to the shadow-processed `DataProto` in the same
run; it is not a cross-launch token identity test.

## Strict scope and limitations

The active feature requires synchronous GRPO training with `rollout.mode=async`,
no critic, vLLM, `agent.default_agent_loop=single_turn_agent`, text-only input,
`ignore_eos=false`, no reward-side KL, `L > H`, and
`max_model_len >= prompt_length + L`. `replace` additionally requires
`filter_groups.enable=true` and the Hydra filter metric to equal the configured
correctness key.

The v1 reward contract is the registered DAPO manager with exact keys
`correctness_key=acc` and `task_score_key=score`; reward-model, importlib/custom,
and sandbox reward paths fail preflight. It rejects concurrent forced-answer or
FA-TR, FA-CAC/FA-RAR, Probe Credit,
Readiness Dominance, BSSF, OBCF, active rollout correction, distillation,
teacher policy, and configured profiling steps. The first implementation is
strict-only and targets text-only math-verifier workloads; multi-turn/tool and
multimodal reward reconstruction are outside its validated scope.

The formal H2048/L8192 example preserves the verified Qwen3-4B H2048 DAPO
actor recipe and changes only the dedicated entry point, boundary configuration,
experiment name, and `max_model_len` from `1024+2048` to `1024+8192=9216` so the
auxiliary continuation has context. The smoke script is a one-step command
description, not authorization to launch training.
