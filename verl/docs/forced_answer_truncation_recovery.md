# Forced-Answer Truncation Recovery v1

## Definition

Forced-Answer Truncation Recovery (FA-TR) modifies training reward only for
high-confidence forced-answer-recoverable truncated failures. It changes
credit, not trajectory tokens.

For each response-cap trajectory, the existing probe generates `K` independent
short branches from the same current-policy prefix. Branch success is determined
only from raw verifier correctness (`reward_extra_info["acc"]` by default):

```text
pFA = correct forced-answer branches / K
```

DAPO shaped reward is retained for telemetry but is not used to decide forced-
answer success.

## Configuration and default

Training credit is disabled by default. Probe inference and reward correction
have separate switches:

```yaml
forced_answer_probe:
  enable: true
  num_samples: 4
  max_new_tokens: 64
  temperature: 1.0
  top_p: 1.0
  correctness_key: acc
  correctness_threshold: 0.5
  training_credit:
    enable: false
    activation_threshold: 0.75
    reward_mode: centered_pfa
```

Enabling training credit while probe inference is disabled fails fast. FA-TR v1
accepts only `reward_mode: centered_pfa` and requires an activation threshold in
`[0, 1]`.

## v1 policy

A trajectory is eligible only when it hit the response cap, its original raw
correctness is below `correctness_threshold`, and its probe was successfully
attempted without context overflow. An eligible trajectory activates when
`pFA >= 0.75`.

With `K=4`, zero, one, or two correct branches preserve the vanilla reward;
three correct branches select `+0.5`, and four select `+1.0`. The target is:

```text
r_target = 2 * pFA - 1
```

This target replaces the original terminal shaped scalar; an overlong penalty
is not added back. The original token reward tensor is cloned, its scalar sum is
compared with the target, and the exact delta is added only to the last valid
response token. Padding, response length, and every other token reward remain
unchanged. `batch.batch["rm_scores"]` remains the original reward-manager output.

The trainer dataflow is:

```text
original reward
  -> forced-answer scoring
  -> effective reward
  -> token_level_scores
  -> token_level_rewards
  -> advantage
  -> actor update
```

## Gradient and isolation

Forced-answer tokens receive no policy gradient because they never enter the
actor-training batch. They exist only in an independent inference and reward
batch. Actor `responses`, `input_ids`, `attention_mask`, `position_ids`,
`response_mask`, old/reference log probabilities, and the PPO objective are
unchanged. Advantage estimators and dynamic sampling are also unchanged.

## Identity and scope

Probe capture assigns a pre-balance parent index to each trajectory. That index
is stored on the PPO row before `balance_batch`; normal `DataProto.reorder`
permutes it with the row, providing an explicit current-row-to-parent join when
reward correction is applied. Unknown or duplicate parent identities fail
closed rather than applying credit to a different trajectory.

The main `RayPPOTrainer` flow used here does not filter generated groups between
probe capture and PPO reward correction. FA-TR v1 only corrects trajectories
that survive into the PPO batch. It does not recover groups discarded before
PPO and does not modify dynamic sampling.

Forced-answer answerability remains a sparse and noisy proxy. FA-TR v1 has not
yet demonstrated an accuracy improvement.

## Metrics

The existing `probe/*` metrics are unchanged. FA-TR additionally reports:

- `fa_tr/num_eligible_truncated_failures`
- `fa_tr/pfa_mean`
- `fa_tr/pfa_eq_0_rate`
- `fa_tr/pfa_ge_025_rate`
- `fa_tr/pfa_ge_050_rate`
- `fa_tr/pfa_ge_075_rate`
- `fa_tr/pfa_eq_1_rate`
- `fa_tr/num_reward_corrected`
- `fa_tr/reward_correction_rate`
- `fa_tr/reward_correction_rate_given_eligible`
- `fa_tr/original_reward_mean_corrected_subset`
- `fa_tr/effective_reward_mean_corrected_subset`
- `fa_tr/reward_delta_mean`
- `fa_tr/reward_delta_max`
- `fa_tr/num_groups`
- `fa_tr/num_groups_with_correction`
- `fa_tr/group_correction_rate`

Distribution metrics use the eligible subset. Reward-effect metrics use the
corrected subset. Empty denominators and empty corrected subsets report `0.0`,
never NaN.
