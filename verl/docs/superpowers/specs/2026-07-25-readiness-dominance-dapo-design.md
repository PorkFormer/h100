# Readiness Dominance DAPO Design

## Scope

This document specifies an experimental Readiness Dominance algorithm beside the existing On-Policy Probe Credit Redistribution implementation. It preserves the existing DAPO Dynamic Sampling, retained complete prompt-group selection, rollout policy-version checks, Immediate Answer Prefix Probe protocol, grouped vLLM generation, verifier scoring, checkpointing, validation, logging, and actor update flow.

The feature compares terminal-equivalent successful trajectories at fixed absolute token horizons and moves positive standard-GRPO advantage mass from directly dominated trajectories to the directly nondominated set. It does not implement AUC rewards, random interruption budgets, token-local potentials, temporal returns, GAE-like traces, HMMs, online dynamics estimation, difficulty or length models, dual variables, scale clipping, or a new verifier reward. It does not change the existing relative Probe API, ProbeCredit defaults, five-position protocol, Dynamic Sampling semantics, entry point, tests, or mathematical behavior.

## Architecture

The implementation uses the existing experimental trainer without extracting another common base and without copying its complete `fit()`:

- `readiness_dominance.py` contains pure PyTorch direct readiness dominance, directly nondominated-set classification, and token-mean positive-mass redistribution.
- `probe_runtime.py` gains a separate absolute-horizon request planner. Existing `relative_horizons()` and `build_probe_requests()` remain unchanged.
- `dapo_dominance_trainer.py` defines `RayDAPOReadinessDominanceTrainer(RayDAPOProbeCreditTrainer)` and overrides configuration validation, `_prepare_final_retained_batch`, final-retained probing, and post-GRPO advantage processing.
- `main_dapo_readiness_dominance.py`, `readiness_dominance_dapo_trainer.yaml`, and an independent smoke launcher select the new trainer without changing ProbeCredit entry points.

The inherited `fit()` remains the single training-loop implementation. Dynamic dispatch invokes the dominance-specific hooks while all normal rollout, reward extraction, filtering, complete-group accumulation, old/reference log-probability, actor update, weight publication, checkpoint, validation, and logging behavior stays in the verified parent implementation.

The subclass must override `_prepare_final_retained_batch`. The parent implementation reads `ProbeCreditConfig.enable`, which is not the Readiness Dominance mode switch. The override validates the retained policy version, runs no Probe in `off`, runs the absolute Probe in `shadow` or `reweight`, and then sleeps replicas exactly once.

## Training Flow

The exact order is:

```text
normal rollout
-> terminal verifier reward and reward extra info
-> DAPO Dynamic Sampling filter
-> final retained complete prompt groups
-> validate retained rollout policy version
-> derive terminal-success mask from retained reward extra info
-> absolute-horizon probes for retained terminal-success trajectories only
-> sleep rollout replicas
-> old/reference log probability
-> standard GRPO advantage
-> derive positive-trajectory mask
-> direct readiness-dominance analysis
-> optional frontier reweighting
-> actor update
-> publish new rollout weights
```

Probe request planning never reads `advantages`, because standard GRPO has not run yet. It uses only final-retained identities, raw tokens, policy version, response lengths, and the terminal-success mask. The positive-trajectory mask is constructed only after standard GRPO has produced `advantages`.

Mode behavior is:

- `off`: no absolute Probe requests, no dominance request seeds, no dominance computation, and no advantage or return changes.
- `shadow`: generate Probes and compute dominance diagnostics after standard GRPO, then assert that `advantages` and `returns` remain bitwise equal to their standard-GRPO values.
- `reweight`: save standard GRPO `advantages` as `terminal_advantages`, apply frontier reweighting, and set both `advantages` and `returns` to the reweighted tensor.

All modes leave `token_level_scores`, `token_level_rewards`, `uid`, `trajectory_id`, retained ordering, and Dynamic Sampling decisions unchanged.

## Terminal-Success Source

Terminal success is not inferred from advantages. The authoritative first-version source is the retained reward extra-info field `acc`, which is also the configured DAPO correctness/filter metric in the dedicated trainer configuration.

The trainer computes the diagnostic score-sum candidate:

```text
score_sum_success = token_level_scores.sum(-1) > 0
```

and reports the fraction that disagrees with `acc`. Score-sum is not used as an automatic fallback because a general reward may include format, overlong, shaping, or other non-correctness terms. The first version fails closed when `acc` is absent, malformed, non-finite, or non-binary. Disagreement with score-sum is diagnostic only and does not stop training or change the authoritative `acc` mask. A future explicitly configured pure-correctness reward integration may permit score-sum fallback, but this implementation does not guess that property.

Probe candidate scoring continues to use the existing verifier path and does not create or modify terminal rewards.

## Absolute-Horizon Probe Runtime

Configured horizons are positive, strictly increasing absolute token counts, defaulting to:

```text
[256, 512, 1024, 2048]
```

For trajectory `i` with active response length `L_i`, horizon `h_k` is valid exactly when:

```text
h_k < L_i
```

Only valid terminal-success trajectory/horizon cells produce requests. Invalid cells do not produce requests and remain invalid during aggregation. EOS padding is never filled with zero, one, or the trajectory's terminal outcome.

Configuration validation is split into two levels. `ReadinessDominanceConfig.validate()` checks the horizon list in isolation. Trainer startup additionally requires every configured horizon to be strictly less than `actor_rollout_ref.rollout.response_length`; a horizon greater than or equal to the maximum response length can never satisfy `h < L` and is rejected before generation.

Every valid request input remains:

```text
prompt_token_ids
+ response_token_ids[:h_k]
+ encode(answer_prefix, add_special_tokens=false)
```

`ProbeRequest` represents its position explicitly with `position_kind`, which is `"relative"` for the existing API and `"absolute"` for Readiness Dominance. Existing relative requests retain their numeric `relative_position`. Absolute requests set `relative_position=None` and carry the horizon only in `absolute_horizon`. They use a new absolute request-ID namespace and `derive_absolute_grouped_request_seed(policy_version, uid, trajectory_id, absolute_horizon, ordered_branch_ids)`. No absolute horizon is cast into `relative_position`, and the existing relative seed payload and results remain unchanged.

Absolute requests reuse prompt-group routing, grouped branch generation, strict context-overflow checks, explicit branch aggregation, and actual policy-version verification. The absolute planner returns requests plus the full `[B,K]` active validity plan and configured horizon vector. Aggregation must reproduce the planned validity mask exactly in strict mode.

## Eligibility and Active Common Support

The post-GRPO eligible trajectory mask is:

```text
terminal_success AND positive_trajectory_mask
```

For the supported token-mean GRPO path, a trajectory is positive when its response-masked standard GRPO advantage has positive total mass. A pair is comparable only when:

1. both trajectories have the same `uid`;
2. both are eligible;
3. at least `min_common_positions` horizons are valid for both.

The default is `min_common_positions=2`. Pairwise comparisons use only the pair's shared valid horizons. Different pairs may therefore use different supports.

The algorithm computes direct pairwise dominance only. It must not calculate a transitive closure, because `i` versus `j` and `j` versus `k` may have different active common supports.

## Direct Readiness Dominance and Directly Nondominated Set

Let `S_ij` be the configured absolute horizons valid for both trajectories `i` and `j`. With branch count `n` and integer `strict_branch_margin`, trajectory `i` directly dominates `j` exactly when:

```text
same uid
AND both terminal-success
AND both have positive standard-GRPO trajectory advantage
AND |S_ij| >= min_common_positions
AND V_i(h) >= V_j(h) for every h in S_ij
AND V_i(h) - V_j(h) >= strict_branch_margin / n for at least one h in S_ij
```

Self-dominance is always false. Crossing profiles do not dominate each other. Identical profiles do not dominate each other. Terminal failures, nonpositive trajectories, different prompts, and pairs with insufficient common support are not compared.

Within each prompt group:

- `dominated` contains an eligible trajectory with at least one incoming direct dominance edge;
- the `directly nondominated set` (reported in tensors as `frontier_mask` for interface brevity) contains every eligible trajectory with no incoming direct dominance edge;
- `group_has_dominance` is true only when the group contains at least one direct edge.

Groups without a direct dominance edge are left bitwise unchanged.

Because each pair can have a different active common support, direct readiness dominance is not assumed to be transitive. The directly nondominated set is computed from incoming edges in the direct matrix only, never from a transitive closure.

## Token-Mean Advantage Redistribution

The current formal DAPO recipe and the repository's DAPO configuration use `actor_rollout_ref.actor.loss_agg_mode=token-mean`. The first version supports only this mode and rejects every other aggregation mode during trainer validation.

Before reweighting, the implementation verifies the expected standard-GRPO structure: all response-token advantages within each trajectory are exactly constant, and all padding advantages are exactly zero. This is the structure produced by the supported GRPO outcome estimators in the current repository. Unexpected token-varying advantages fail closed rather than being assigned an ill-defined trajectory weight.

For each group with direct dominance, define masked positive advantage mass:

```text
M_before =
  sum(response_mask * clamp(standard_advantages, min=0))
  over eligible trajectories

M_frontier =
  sum(response_mask * clamp(standard_advantages, min=0))
  over directly nondominated trajectories
```

The group scale is:

```text
scale = M_before / M_frontier
```

Reweighting is implemented as one trajectory-level weight:

```text
dominated eligible trajectory: weight = 0
directly nondominated eligible trajectory in a dominance group: weight = scale
all other trajectories: weight = 1
```

The weight multiplies the full standard-GRPO response-token advantage row. Because the supported GRPO row is constant on active response tokens and eligible rows are positive, this is equivalent to moving its masked positive mass. Padding remains zero. Terminal-negative trajectories, terminal-success but noneligible trajectories, and every trajectory in a group without dominance retain weight one.

This preserves the positive-advantage numerator relevant to `token-mean` aggregation while response masks and the denominator remain fixed. It makes no claim for sequence-level aggregation modes, importance-ratio clipping, or post-minibatch gradient equality.

No scale clipping is introduced. `M_before` and `M_frontier` must be finite, `M_frontier` must be strictly positive, `scale` must be finite, and the post-reweight mass residual must be finite and within numerical tolerance. Any NaN, infinity, or zero frontier mass fails closed; it is never converted to a silent skip. Scale diagnostics include mean, maximum, and p50/p90/p99 quantiles.

`reweight` remains an experimental engineering mode. Formal reweight training must wait for offline split-half agreement, repeated-Probe stability, and branch-count sensitivity checks. The shipped smoke launcher is fixed to `shadow` and provides no default path that starts reweight training.

## Configuration

`ReadinessDominanceConfig` is independent of `ProbeCreditConfig`:

```yaml
mode: off
absolute_horizons: [256, 512, 1024, 2048]
n: 4
temperature: 0.7
top_p: 0.95
top_k: -1
max_tokens: 32
stop: ["\n"]
answer_prefix: "\n\nAnswer:"
strict: true
strict_branch_margin: 1
min_common_positions: 2
max_concurrent_requests: 128
request_batch_size: 512
```

Only `off`, `shadow`, and `reweight` modes are valid. Validation requires nonempty positive strictly increasing horizons, positive `n`, positive `max_tokens`, positive `strict_branch_margin <= n`, positive `min_common_positions`, valid sampling parameters, strict operation, positive request limits, and `request_batch_size >= max_concurrent_requests`.

The dedicated DAPO trainer additionally requires synchronous GRPO, vLLM rollout, no critic, no KL-in-reward, no active rollout correction, no distillation or teacher policy, no configured profiling steps, `filter_groups.enable=true`, `filter_groups.metric=acc`, and `loss_agg_mode=token-mean`.

## Metrics

Required dominance metrics are:

```text
dominance/eligible_group_rate
dominance/group_with_dominance_rate
dominance/eligible_positive_rate
dominance/dominated_positive_rate
dominance/frontier_fraction
dominance/frontier_size_mean
dominance/profile_cross_rate
dominance/positive_mass_before
dominance/positive_mass_after
dominance/mass_residual_max
dominance/frontier_scale_mean
dominance/frontier_scale_max
dominance/frontier_scale_p50
dominance/frontier_scale_p90
dominance/frontier_scale_p99
dominance/skipped_invalid_mass_groups
dominance/terminal_success_score_disagreement_rate
dominance/probe_valid_cell_rate
dominance/same_group_eligible_pair_count
dominance/comparable_pair_count
dominance/pair_coverage_rate
dominance/common_positions_mean
dominance/common_positions_min
dominance/common_positions_max
```

Strict invalid mass is fail-closed, so `skipped_invalid_mass_groups` remains zero on successful first-version steps and records no silently modified group.

Probe runtime metrics retain separate request count, branch count, input/output token counts, concurrency, batch size, and generation/scoring timing under the `dominance/` namespace.

Every rate has a fixed denominator:

| Metric | Numerator | Denominator |
| --- | --- | --- |
| `eligible_group_rate` | prompt groups containing at least one comparable pair | all retained prompt groups |
| `group_with_dominance_rate` | prompt groups containing at least one direct edge | all retained prompt groups |
| `eligible_positive_rate` | trajectories satisfying authoritative terminal success and positive standard GRPO advantage | all retained trajectories |
| `dominated_positive_rate` | eligible trajectories with an incoming direct edge | all eligible trajectories |
| `frontier_fraction` | eligible trajectories with no incoming direct edge | all eligible trajectories |
| `profile_cross_rate` | comparable unordered pairs whose profiles cross | all comparable unordered pairs |
| `terminal_success_score_disagreement_rate` | trajectories where authoritative `acc` differs from score-sum diagnostic | all retained trajectories |
| `probe_valid_cell_rate` | valid terminal-success trajectory/horizon cells | all terminal-success trajectory/horizon cells |
| `pair_coverage_rate` | same-group eligible unordered pairs meeting `min_common_positions` | all same-group eligible unordered pairs |

If a rate denominator is zero, the reported rate is zero. `common_positions_mean/min/max` use the common-valid-position counts of comparable unordered pairs; all three are zero when no comparable pair exists. Pair counts are reported as raw float-valued metrics.

## Tensors and Invariants

In `reweight` mode the batch saves:

```text
terminal_advantages
dominance_probe_values
dominance_probe_valid_mask
dominance_absolute_horizons
dominance_terminal_success
dominance_frontier_mask
dominance_dominated_mask
dominance_weights
```

Shadow mode may attach diagnostics but must not replace `advantages` or `returns`. Off mode attaches no dominance Probe or analysis tensors.

The trainer asserts before and after dominance processing that terminal scores, terminal rewards, identities, and ordering are unchanged. Strict aggregation requires every planned valid Probe cell to have all branches and every planned invalid cell to remain invalid.

## Testing and Delivery

Development follows red-green-refactor TDD. Off and shadow behavior is completed and tested before reweight is implemented.

Pure CPU tests cover direct dominance, crossing, identical profiles, UID isolation, terminal mismatch, nonpositive trajectories, insufficient common support, multi-trajectory directly nondominated sets, shape/dtype/range/finite validation, determinism, lack of transitive closure, mass conservation, bitwise no-op groups, unchanged negatives and padding, independent groups, singleton positives, scale diagnostics, and invalid mass failures.

Absolute runtime tests cover equal absolute truncation across different lengths, inactive horizons, no invalid requests, stable IDs and seeds, branch aggregation, prompt routing, mixed policy versions, strict missing branches, context overflow, and unchanged relative-Probe regression tests.

Trainer CPU/mock tests verify the full event order, off no-op behavior, shadow bitwise parity, reweight behavior, immutable reward/score tensors, stable retained IDs and ordering, final-retained-only probing, no filtered-group probes, incomplete-group failures, pre-Probe policy mismatch, strict branch failure, no-dominance parity, authoritative `acc`, missing/invalid correctness failures, nonfatal disagreement metrics, and token-mean-only validation.

Each complete testable task is committed directly on local `main`. Before the final push, the implementation fetches `origin`, rebases local `main` on `origin/main` if required, reruns all tests after any rebase, and pushes only `main` without force.
