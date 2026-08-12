# Truncation-Aware Reward Recovery: Forced-Answer Probe

## Motivation

Short rollout horizons can censor reward: a trajectory may reach the response
cap before the policy naturally emits its final answer, and the verifier then
treats the incomplete response as unsuccessful. We call this experimental
failure mode **truncation-induced reward censoring**.

The probe asks a diagnostic question: among trajectories that hit the response
cap, how often can the same current policy produce a correct answer when
explicitly required to answer immediately? FA-TR v1 can optionally use that
signal for training credit; see
[Forced-Answer Truncation Recovery](forced_answer_truncation_recovery.md).

## Workflow

The rollout tokens and actor inputs are unchanged. Unless optional FA-TR
training credit is enabled, reward, advantage, and actor updates are unchanged
as well. For each single-turn trajectory, the rollout backend's finish reason
is used to distinguish natural EOS from maximum-length termination. Sequence
length and the EOS token provide a fallback when backend metadata is
unavailable.

When the feature is enabled, only response-cap trajectories receive an
independent grouped vLLM request containing:

```text
original prompt tokens
+ original capped response tokens
+ configured forced-answer instruction tokens
```

The same current-policy rollout engine generates `num_samples` short answers.
An independent reward-only `DataProto` sends those answers through the same
configured reward manager/verifier as the original rollout. Probe tokens and
rewards are never joined to the actor-training batch.

## Configuration

The feature is disabled by default under
`actor_rollout_ref.rollout.forced_answer_probe`:

```yaml
forced_answer_probe:
  enable: false
  num_samples: 4
  max_new_tokens: 64
  temperature: 1.0
  top_p: 1.0
  instruction: "\n\nProvide only the final answer in this exact format: Answer: <final answer>"
  correctness_key: acc
  correctness_threshold: 0.5
  training_credit:
    enable: false
    activation_threshold: 0.75
    reward_mode: centered_pfa
  success_threshold: 0.0
  high_confidence_threshold: 1.0
  save_examples: false
  max_examples_per_step: 8
  examples_dir: null
```

Recoverability metrics use raw verifier correctness (`acc` by default), not
DAPO shaped reward. Values at or above `correctness_threshold` are correct.
`success_threshold` is retained only for shaped-reward telemetry and backward
configuration compatibility; it does not classify answer correctness.
`high_confidence_threshold` is applied to each attempted trajectory's mean
binary raw-correctness result. It controls diagnostic telemetry independently
from FA-TR's `training_credit.activation_threshold`.

If qualitative examples are enabled, at most `max_examples_per_step` records
are written; each retains only the original response tail rather than its full
prefix.

The first online implementation supports single-turn vLLM rollouts. Before a
request is sent, it checks whether the model context can accommodate the
original prompt, capped response, forced instruction, and `max_new_tokens`.
Overflow probes are skipped without truncating the prompt or response prefix,
and are excluded from attempted-probe denominators. They are unobserved, not
incorrect.

## Correctness and DAPO shaping

DAPO may add an overlong penalty to the raw verifier score. Consequently, a
correct capped trajectory can have raw `acc = 1` but a shaped reward of zero or
less. Both original and forced-answer correctness are read from the configured
reward extra-info field after the normal `extract_reward()` path. Missing raw
correctness fails closed; the probe never infers correctness from `rm_scores`.

Shaped rewards remain visible as telemetry so a run can directly compare raw
answer correctness with the DAPO training reward. Probe-only operation leaves
training credit unchanged. When explicitly enabled, FA-TR uses only raw
correctness to decide whether to replace a qualifying terminal training score.

## Metrics

- `probe/hit_cap_rate`
- `probe/num_truncated_trajectories`
- `probe/num_probe_generations`
- `probe/context_overflow_count`
- `probe/context_overflow_rate`
- `probe/probe_attempted_count`
- `probe/probe_coverage_rate`
- `probe/success_rate_mean`
- `probe/p_any_success`
- `probe/p_all_success`
- `probe/extra_input_tokens`
- `probe/extra_generated_tokens`
- `probe/extra_total_tokens`
- `probe/extra_generated_token_ratio`
- `probe/extra_total_token_ratio`
- `probe/extra_token_ratio`
- `probe/raw_correctness_mean`
- `probe/shaped_reward_mean`
- `probe/truncation_false_negative_candidate_rate`
- `probe/truncation_high_confidence_recoverable_rate`
- `probe/recovery_rate_given_truncated_failure`

Success-rate, any-success, and all-success aggregates use attempted probes.
`context_overflow_rate` and `probe_coverage_rate` divide by all truncated
trajectories and return zero when none are truncated.

The key conditional metric is:

```text
probe/recovery_rate_given_truncated_failure =
P(
  forced-answer any-success
  | response-cap truncation,
    original raw verifier incorrect,
    probe successfully attempted
)
```

Its denominator excludes originally correct trajectories and skipped overflow
probes. By contrast, `truncation_false_negative_candidate_rate` retains all
truncated trajectories as its denominator; its numerator requires an attempted,
originally raw-incorrect trajectory with at least one correct probe branch.
The high-confidence rate has the same all-truncated denominator but requires
the configured branch success fraction.

Input-token overhead counts each grouped parent request's shared prefill once,
not once per sampled branch. Generated overhead sums every branch. The legacy
`probe/extra_token_ratio` is an alias for
`probe/extra_generated_token_ratio`.

## Recommended protocols

Online diagnostic:

```yaml
forced_answer_probe:
  enable: true
  num_samples: 4
  max_new_tokens: 64
  temperature: 1.0
  top_p: 1.0
  high_confidence_threshold: 0.75
```

Canonical paper diagnostic (not the online default):

```yaml
forced_answer_probe:
  enable: true
  num_samples: 4
  max_new_tokens: 128
  temperature: 0.7
  top_p: 0.95
  high_confidence_threshold: 0.75
```

For an H=2048 diagnostic run, append overrides such as:

```bash
data.max_response_length=2048 \
actor_rollout_ref.rollout.forced_answer_probe.enable=true \
actor_rollout_ref.rollout.forced_answer_probe.num_samples=4 \
actor_rollout_ref.rollout.forced_answer_probe.max_new_tokens=64
```

The explicit `+` Hydra prefix is only needed when applying the override to an
older composed config that does not yet contain the field.

## Interpretation and current scope

Forced-answer correctness is an **answer-availability proxy**. It shows that the
policy can produce a correct answer from a capped prefix under an intervention;
it does not show that a natural H=8192 continuation would have been correct.
Accordingly, the metrics use `candidate` and `recoverable` terminology and make
no claim that these samples are proven false negatives.

By default this remains auxiliary inference only. Enabling FA-TR changes the
effective terminal training reward for its strict activation subset, while
leaving original reward telemetry and all actor trajectory tensors unchanged.
Negative masking, sparse long continuation, and adaptive horizon control remain
outside this implementation.
