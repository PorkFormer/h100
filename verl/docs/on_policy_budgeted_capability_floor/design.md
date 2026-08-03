# On-Policy Budgeted Capability Floor

OBCF uses the Base model only to define an offline verifier-certified capability floor. The online constraint gradient is estimated entirely from current-policy rollouts and acts only on tokens within the protected budget.

## Scientific contract

For prompt `x`, the offline audit records the number `k_x` of Base rollouts whose existing `prefix_reward_<B>` is true. A prompt is protected when `k_x >= support_threshold`. With `n_0=base_rollouts_per_prompt` and tolerance count `c`, its floor is

```text
f_0(x) = max(k_x - c, 0) / n_0.
```

The event is exactly the reward pipeline's operational prefix `acc` at `B`. It does not require natural completion, EOS before `B`, a full-reward success, or an answer suffix. Prefix batches contain the current rollout's exact token IDs through position `B-1`; they are never decoded and re-encoded or continued.

For each protected retained current-policy group, OBCF computes

```text
q(x) = mean_i r^B_{x,i}
d(x) = max(f_0(x) - q(x), 0)
A_cap[x,i,t] = 1[d(x) > 0] * (r^B_{x,i} - q(x)) * 1[t < B].
```

The centering is a group mean only. OBCF does not divide by group standard deviation. The actor receives `A_terminal + lambda * A_cap` in the existing `advantages` tensor, and the existing actor loss performs one normal optimizer update. Base-unsolved, below-threshold, unprotected, and inactive groups have exactly zero capability advantage. An active all-zero group has a measured deficit but zero centered signal; metrics report that absence rather than substituting a recovery mechanism.

After a protected observation, the controller initializes its EMA to the first observed batch deficit. Later observations apply the configured EMA. Lambda ascent uses the configured update interval and projects onto `[0, lambda_max]`; actor composition always uses the pre-update lambda. A batch with no protected prompt does not change lambda, EMA, observation count, or last-observation step.

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
