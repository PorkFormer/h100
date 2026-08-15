# FA-CAC v2 formal canonical DAPO protocol

## Immutable provenance gates

- h100 base and Step200 provenance commit: `3f5e56873c29a72857ac84a3e9f7efd9b3ce33b1`
- detached canonical DAPO commit: `7aed6b230776f963fa09509c10d9c3a767d1102c`
- canonical pre-patch `RayDAPOTrainer.fit` SHA-256: `e75af411cabf44003d4e7f2f7aed16549ac97b023480afa54840ae64517bf63c`
- historical v1 adapter SHA-256: `c71b5480c568d4655d057c0106079a486d9dbebb7bfbfb536f9868d031c45120`
- executed historical v1 fit SHA-256: `39b3693224e4fb37e63b478bfbbda23d801de1f5561b7c974aab87db539e0946`

The v2 adapter verifies all of these runtime source assertions and refuses a
dirty or mismatched canonical repository. It patches the canonical fit method
only in memory. It does not edit or vendor any canonical DAPO file, and the
historical v1 adapter remains byte-for-byte unchanged.

## Reward and GRPO semantics

FA-CAC consumes only the exact `reward_extra_info["score"]` produced by the
reward manager. It fails closed when that vector is absent, non-finite, or not
aligned with the retained PPO batch; it never reconstructs task score from
`acc`. For the formal `math_dapo` run, task score is `+1` for correct and `-1`
otherwise. With response cap 2048, overlong buffer 410, and factor 1.0, the
formal residual is:

```text
s_reg = min(-(response_length - 1638) / 410, 0)
```

The generic implementation defines `s_reg = token_level_rewards.sum(-1) -
s_task`, so in-reward KL or other future shaping remains in the residual.

The extracted GRPO helper preserves historical UID grouping, response masks,
`torch.std` default correction, epsilon `1e-6`, singleton `mean=0/std=1`, and
Dr.GRPO behavior when standard-deviation normalization is disabled. Task and
residual numerators use their own group means and the total-reward standard
deviation. Runtime reconstruction requires:

```text
max |A_vanilla - A_task - A_reg| <= 1e-6
```

The mechanism first defines censor candidates independently of GRPO signs:

```text
censor_candidate = hit_response_cap
                   and probe_attempted
                   and not context_overflow
                   and raw_correctness < correctness_threshold
```

Candidate pFA evidence must be present, finite, and in `[0, 1]`. Candidates
are assigned to exactly one outcome in this stable first-failure order:

```text
pFA <= 0                 -> excluded_pfa_zero
A_vanilla >= 0           -> excluded_nonnegative_vanilla_adv
A_task >= 0              -> excluded_nonnegative_task_adv
otherwise                -> CAC eligible
```

Thus CAC eligibility is the conjunction of the four candidate conditions,
`pFA > 0`, `A_vanilla < 0`, and `A_task < 0`. Telemetry must satisfy:

```text
candidate_count = eligible_count
                  + excluded_pfa_zero_count
                  + excluded_nonnegative_vanilla_adv_count
                  + excluded_nonnegative_task_adv_count
```

For CAC-eligible rows:

```text
A_pre = A_reg + (1 - pFA) * A_task
A_cac = min(0, A_pre)
```

`A_pre` retains the complete residual component. The final minimum is a
conservative sign projection; when it activates, exact residual preservation
is not claimed. Non-target rows and padding remain bitwise unchanged. Applied
GRPO assigns actor-visible `advantages` and `returns` consistently and never
changes reward tensors. Shadow mode (`enable=true, apply=false`) computes the
same projection and telemetry while leaving both actor tensors Vanilla.
Every retained PPO row must have at least one valid response token when CAC
is enabled. Reward and score tensors are snapshotted and checked bitwise after
the hook; reported drift is computed from those snapshots rather than assumed.

Batch-wide `before` diagnostics describe Vanilla scalar advantages, while
batch-wide `after` diagnostics always describe the counterfactual CAC
projection. Their mean, absolute mean, RMS, and token-weighted sum are
therefore identical between shadow and apply runs on the same input. Only
actor-visible telemetry differs. The safety counts
`raw_correct_changed_count` and `incorrect_became_positive_count` must be zero.

## Four-mode truth table

| FA-TR v1 | FA-CAC v2 | Behavior |
| --- | --- | --- |
| off | off | canonical Vanilla rewards and GRPO |
| on | off | historical v1 terminal reward replacement before GRPO |
| off | on | original rewards, Vanilla GRPO, then shared FA-CAC hook |
| on | on | fail fast before training |

The shared `apply_fa_cac_post_advantage_hook` is called in both
`RayPPOTrainer.fit` and the in-memory canonical DAPO adapter immediately after
Vanilla `compute_advantage` and before actor update.

## Launcher and import chain

The formal launcher is:

```text
/workspace/rl/h100-fa-cac-v2/analysis/fa_cac_v2/tools/matched_dapo_main.py
  -> analysis.fa_cac_v2.tools.dapo_adapter
  -> MatchedFACACDAPOTaskRunner
  -> guarded in-memory recipe.dapo.dapo_ray_trainer.RayDAPOTrainer.fit
  -> verl.trainer.ppo.censor_aware_advantage.apply_fa_cac_post_advantage_hook
```

Because the detached canonical DAPO config does not define the CAC subtree,
the launcher removes `algorithm.censor_aware_advantage.*` CLI assignments
before canonical Hydra composition. It then installs the complete subtree
(`_target_`, `enable=false`, `apply=true`, and the supported mode) and replays
the saved CAC overrides in their original order before config printing or
training initialization. This permits ordinary overrides without Hydra `+`
syntax and preserves last-override-wins behavior.

## Verification boundary and experimental gate

The canonical CPU harness invokes the guarded, source-patched real fit method
with mocked rollout, reward-probe, actor, and lifecycle infrastructure. It does
not validate real Ray/vLLM replica sleep/wake behavior, grouped-output
transport, distributed balancing, network transport, or lifecycle timing.

No GPU work is part of implementation verification. The next experimental
gate is a short canonical DAPO shadow-mode GPU smoke with
`censor_aware_advantage.enable=true` and `apply=false`. An `apply=true`
experiment must not be considered until that gate passes.
