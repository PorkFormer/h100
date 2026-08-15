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

For eligible capped, attempted, non-overflowed raw-verifier failures:

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

## Verification boundary and experimental gate

The canonical CPU harness invokes the guarded, source-patched real fit method
with mocked rollout, reward-probe, actor, and lifecycle infrastructure. It does
not validate real Ray/vLLM replica sleep/wake behavior, grouped-output
transport, distributed balancing, network transport, or lifecycle timing.

No GPU work is part of implementation verification. The next experimental
gate is a short canonical DAPO shadow-mode GPU smoke with
`censor_aware_advantage.enable=true` and `apply=false`. An `apply=true`
experiment must not be considered until that gate passes.
