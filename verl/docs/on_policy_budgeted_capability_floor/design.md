# On-Policy Budgeted Capability Floor

OBCF uses the Base model only to define an offline verifier-certified capability floor. The online constraint gradient is estimated entirely from current-policy rollouts and acts only on tokens within the protected budget.

## Scientific contract

For prompt `x`, the offline audit records the number `k_x` of Base rollouts whose existing `prefix_reward_<B>` is true. A prompt is protected when `k_x >= support_threshold`. With `n_0=base_rollouts_per_prompt` and tolerance count `c`, its floor is

```text
f_0(x) = max(k_x - c, 0) / n_0.
```

The event is exactly the reward pipeline's operational prefix `acc` at `B`. It does not require natural completion, EOS before `B`, a full-reward success, or an answer suffix. Prefix batches contain the current rollout's exact token IDs through position `B-1`; they are never decoded and re-encoded or continued.

Offline artifacts may define `prefix_reward_<B>` differently even when they name the same verifier. Before cache construction, the event-equivalence validator therefore rebuilds a `DataProto` from frozen prompt and response token IDs, truncates those IDs at `B`, and calls the same `RewardLoopWorker.compute_score_batch` pipeline used online. It requires exact row-wise agreement and emits an attested `prefix_protocol_fingerprint`. Verifier provenance by itself is not event equivalence.

For each protected retained current-policy group, OBCF computes

```text
q(x) = mean_i r^B_{x,i}
d(x) = max(f_0(x) - q(x), 0)
A_cap[x,i,t] = 1[d(x) > 0] * (r^B_{x,i} - q(x)) * 1[t < B].
```

The centering is a group mean only. OBCF does not divide by group standard deviation. Because a sample is included in its own group mean, the same-sample centered score-function signal has the finite-group factor `(n-1)/n`. The actor receives `A_terminal + lambda * A_cap` in the existing `advantages` tensor, and the existing actor loss performs one normal optimizer update. Base-unsolved, below-threshold, unprotected, and inactive groups have exactly zero capability advantage. An active all-zero group has a measured deficit but zero centered signal; this is a preservation limitation, not a recovery branch, and metrics report it without witness rescue or another rollout.

With `n` current rollouts, the smallest positive empirical rate is `1/n`. Under the strict gate `q < f`, a floor can act on a mixed group only when `f > 1/n`. Consequently, Base `2/8` with tolerance 1 produces `f=1/8` and is structurally inert when current `rollout.n=8`: at `0/8` it can report a deficit but has no centered gradient, while at `1/8` the strict gate is inactive. The first active configuration uses tolerance 0, giving `f=2/8`. Tolerance 1 remains a shadow/simulator diagnostic and may enter dual mode only when every protected floor is still actionable.

After a protected observation, the controller initializes its EMA to the first observed batch deficit. Later observations apply the configured EMA. Lambda ascent uses the configured update interval and projects onto `[0, lambda_max]`; actor composition always uses the pre-update lambda. `update_interval` controls lambda ascent only. Prefix scoring and capability-advantage computation occur on every training step in shadow and dual modes. A batch with no protected prompt does not change lambda, EMA, observation count, or last-observation step.

`mode=off` delegates to inherited DAPO without cache or prefix scoring. `mode=shadow` performs prefix scoring and metrics without changing actor inputs or dual state. `mode=dual` composes capability advantages before the sole actor update and updates the controller afterward.

## Scope and limitations

The first implementation is synchronous, single-turn, text-only GRPO/DAPO with vLLM rollout, no critic, no online reference worker, no global KL, no distillation, no rollout correction, and no ProbeCredit, ReadinessDominance, or witness-BSSF composition.

Scientific interpretation is limited by:

- the finite-sample Base floor, including audit sampling error;
- bias from using the same current group for the deficit gate and centered signal;
- a verifier specification gap between operational `acc` and broader task capability;
- PPO clipping, non-convex optimization, and finite optimization accuracy;
- active all-zero groups, which expose deficit but provide no centered gradient;
- Base-only coverage: prompts outside Base-supported protection receive no OBCF term;
- seed and dataset dependence, so one seed cannot support a universal causal claim.

OBCF results must not be described as short-answer optimization, capability recovery, proof of internal forgetting, unbiased exact constrained optimization, or trajectory preservation. The method constrains a verifier-certified operational event under the stated sampling and optimization protocol.
