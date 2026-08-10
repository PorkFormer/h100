# Truncation-Aware Reward Recovery: Forced-Answer Probe

## Motivation

Short rollout horizons can censor reward: a trajectory may reach the response
cap before the policy naturally emits its final answer, and the verifier then
treats the incomplete response as unsuccessful. We call this experimental
failure mode **truncation-induced reward censoring**.

This Step 2 implementation asks a diagnostic question only: among trajectories
that hit the response cap, how often can the same current policy produce a
correct answer when explicitly required to answer immediately?

## Workflow

The normal rollout, reward, advantage, and actor update are unchanged. For each
single-turn trajectory, the rollout backend's finish reason is used to
distinguish natural EOS from maximum-length termination. Sequence length and
the EOS token provide a fallback when backend metadata is unavailable.

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
  num_samples: 2
  max_new_tokens: 64
  temperature: 1.0
  top_p: 1.0
  instruction: "\n\nNow stop reasoning and provide only the final answer in the required format."
  success_threshold: 0.0
  high_confidence_threshold: 0.5
  save_examples: false
  max_examples_per_step: 8
  examples_dir: null
```

`success_threshold` follows the existing verifier convention: a reward strictly
above it is a successful probe. `high_confidence_threshold` is applied to each
truncated trajectory's mean binary probe success. If qualitative examples are
enabled, at most `max_examples_per_step` records are written; each retains only
the original response tail rather than its full prefix.

The first online implementation supports single-turn vLLM rollouts. The model
context length must accommodate the original prompt, capped response, forced
instruction, and `max_new_tokens` without truncating the retained prefix.

## Metrics

- `probe/hit_cap_rate`
- `probe/num_truncated_trajectories`
- `probe/num_probe_generations`
- `probe/success_rate_mean`
- `probe/p_any_success`
- `probe/p_all_success`
- `probe/extra_generated_tokens`
- `probe/extra_token_ratio`
- `probe/reward_mean`
- `probe/truncation_false_negative_candidate_rate`
- `probe/truncation_high_confidence_recoverable_rate`

The two recoverability rates use truncated trajectories as their denominator.
The candidate rate requires an unsuccessful original reward and at least one
successful probe. The high-confidence rate additionally requires the configured
per-trajectory success-rate threshold.

## Example

For an H=2048 experiment, append overrides such as:

```bash
data.max_response_length=2048 \
actor_rollout_ref.rollout.forced_answer_probe.enable=true \
actor_rollout_ref.rollout.forced_answer_probe.num_samples=2 \
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

This implementation is auxiliary inference only. It does not change original
rewards, advantages, GRPO normalization, DAPO shaping, actor loss, response
masks, old/reference log probabilities, or training sequence lengths. Negative
masking, soft recovered rewards, sparse long continuation, and adaptive horizon
control are intentionally outside Step 2.
