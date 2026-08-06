# OBCF Gate D semantic equivalence protocol v2

Protocol version: `obcf-gate-d-semantic-equivalence-v2`

Preregistered: 2026-08-06, before Gate D v2 comparisons

## Status and amendment history

The original Gate D1 exact cross-launch comparison remains `FAIL`. Its decision
artifact at
`analysis/obcf_gate_cde_resume_20260805/gate_d_decision.json` is immutable and
must not be overwritten or relabelled. It established that independent baseline
launches themselves did not reproduce the same stochastic rollout and retained
batch. Gate D v2 amends the protocol by separating exact control-flow evidence,
same-batch numerical equivalence, and cross-launch statistical diagnostics.

This amendment does not weaken OBCF-specific correctness and does not authorize
any scientific recipe change. In particular, it does not make stochastic
sampling deterministic, add request seeds, disable sampling, change dynamic
filtering, or change the batch, rollout count, topology, reward, verifier,
optimizer, dtype, or OBCF objective.

## Object-specific equivalence standards

### Exact invariants

The following are compared without tolerance:

- resolved scientific configuration;
- initial actor checkpoint and initial optimizer/scheduler identities;
- frozen-batch prompt IDs, response token IDs, terminal reward labels, masks,
  discrete metadata, schemas, shapes, dtypes, and field names;
- rollout and actor-update counts;
- OBCF-off cache loads, prefix verifier calls, constraint observations, and
  lambda updates, all of which must be zero;
- absence of OBCF capability fields in off mode;
- shadow total advantage exactly equal to terminal advantage;
- shadow lambda before and after exactly zero and no constraint update;
- verifier error/timeout counts;
- cache, verifier, prefix protocol, seed protocol, and Gate D v2 protocol
  fingerprints.

No numerical tolerance, statistical interval, or baseline envelope can excuse
an exact-invariant failure.

### Numerical equivalence on one frozen batch

Float loss, old log probabilities, gradients, parameters, optimizer tensors,
and scalar training metrics may have low-order floating-point differences only
when all launches use the exact same frozen DataProto and the exact same initial
actor, optimizer, and scheduler state.

The preregistered `torch.testing.assert_close` tolerances are:

| dtype | rtol | atol |
|---|---:|---:|
| float64 | 1e-7 | 1e-9 |
| float32 | 1e-5 | 1e-7 |
| bfloat16 | 5e-3 | 5e-4 |
| float16 | 5e-3 | 5e-4 |

For every comparison the report also includes maximum absolute difference,
maximum relative difference, relative L2, mismatch fraction at the base
tolerance, and cosine:

```text
relative_l2 = ||a-b||_2 / (||a||_2 + 1e-12)
cosine = <a,b> / (||a||_2 ||b||_2 + 1e-12)
```

Parameter comparison uses `delta_theta = theta_after - theta_before`.

Baseline replicates A, B, and C are run first. Their pairwise comparisons form
the numerical-noise envelope tensor by tensor. A tensor that exceeds its dtype
base tolerance within duplicate baselines is labelled
`BASELINE_NONDETERMINISTIC`. This does not authorize a larger tolerance:
baseline-versus-off and off-versus-shadow may pass for that tensor only when all
reported discrepancies do not exceed the already measured baseline envelope.
Any exceedance or unexplained first divergence fails Gate D v2. Tolerances and
envelope rules are not changed after result inspection.

### Independent stochastic launch diagnostics

Independent full launches do not require exact response token IDs, reward
realization, retained prompt identities, actor loss, or final checkpoint hash.
They report pre-generation request identity/seed coverage, response and reward
distributions, group-success histograms, retained-set overlap, response lengths,
training metrics, timing, memory, effect sizes, and descriptive confidence
intervals. Baseline-versus-baseline variability is the descriptive reference for
baseline-versus-off variability. A p-value is not equivalence proof.

This layer blocks only a scientific-config difference, control-flow difference,
OBCF-off side effect, rollout/update-count difference, or a clear systematic
baseline-versus-off shift beyond baseline natural variability. Different
responses or retained sets alone do not fail Gate D-B.

## Four nondeterminism boundaries

Opt-in diagnostics, default disabled, save atomic per-run artifacts at:

1. dataloader output after request expansion and before generation;
2. completed rollout before reward;
3. reward completion before dynamic filtering;
4. dynamic filtering output and effective retained training batch.

The writer binds each manifest to resolved config, Git commit, run identity,
rank, and boundary. A rank/run-specific target prevents concurrent overwrite.
Failed writes leave no committed boundary or pass artifact. The writer never
adds a DataProto field and verifies the caller-visible DataProto hash before and
after serialization. Duplicate sample identities fail the audit. When an
explicit per-request sampling seed is absent, the saved value is JSON `null` and
the status is `MISSING_EXPLICIT_REQUEST_SEED`; instrumentation must not invent a
seed.

## Frozen-batch package and harness

The preferred source is an existing prior post-rollout dump. It must be audited
for complete actor inputs, source/config/commit/checkpoint identity, exact field
schema, and full DataProto hash. If it cannot meet this contract, one new
baseline one-step launch may freeze the unmodified post-filter batch immediately
before actor update. No rerun may be selected for a favorable realization.

The diagnostic harness:

- accepts `baseline`, `off`, or `shadow` only;
- verifies batch and initial-state manifests before worker initialization;
- does not call rollout or dynamic filtering;
- performs exactly one actor update;
- checks the frozen input hash again after the run;
- writes exact pre-update fields, scalar metrics, gradient summaries,
  `delta_theta`, optimizer states, scheduler/global step, counters, and resource
  usage to a unique atomic output;
- is not connected to the formal multi-step training entrypoint.

An invalid mode, hash mismatch, missing field, initial-state mismatch, rollout
request, duplicate actor update, or input mutation fails closed.

## Gate D-A: structural off no-op

Off must satisfy exactly:

```text
mode = off
cache_path = null
cache_loaded_count = 0
prefix_verifier_calls = 0
constraint_observation_count = 0
lambda_update_count = 0
capability_batch_field_count = 0
extra_rollout_request_count = 0
extra_actor_update_count = 0
```

The scientific resolved-config diff may contain only diagnostic entrypoint,
mode, output paths, experiment name, and diagnostic paths. Any other difference
fails Gate D-A.

## Gate D-B: baseline versus off

Baseline A/B/C and off A/B/C start independently from identical frozen inputs
and initial state. Before actor update, prompt IDs, response IDs, terminal
rewards, response and PPO masks, terminal advantages, returns, valid-token
count, actor fields, shapes, dtypes, and update count must be exact. Intermediate
values identify any first divergence; discrete mismatches never use tolerance.

Loss, policy-gradient loss, entropy, KL, clip fraction, gradient norm,
per-parameter gradients, `delta_theta`, optimizer tensor states, and
scheduler/global step use the preregistered numerical rules.

Gate D-B passes only when all discrete inputs and off side effects are exact and
all float differences remain inside base tolerance or the preregistered
baseline envelope, with no unexplained first divergence. It fails for any
discrete mismatch, OBCF path entry, update-count change, artifact/config
mismatch, or beyond-envelope difference. It is blocked when the real actor
update cannot be replayed, necessary inputs are missing, or duplicate baselines
cannot form a stable envelope.

## Gate D-C: stochastic descriptive diagnostics

Prior baseline, duplicate baseline, and off one-step runs are reused. At most one
additional baseline and two additional off launches may be run, serially and
with identical configuration. Reports include prompt multiset overlap, seed
coverage, response overlap, rewards, group success, retained overlap, loss,
entropy, KL, grad norm, length, rollout/update/step timing, and peak memory with
mean, standard deviation, range, standardized effect size, and descriptive
confidence intervals.

## Gate D2: off versus shadow

Gate D2 is authorized only after Gate D-A and D-B pass. Off A/B/C and shadow
A/B/C use the identical initial state and frozen batch. Shadow uses the formal
B=2048 cache, `mode=shadow`, lambda zero, no dual update, and no total-advantage
modification. Prompt/response IDs, rewards, masks, terminal advantage, optimizing
actor fields, update count, zero rollout requests, lambda, and update counts are
exact. Shadow may load cache, call the prefix verifier, observe capability and
record read-only diagnostics/fields. Actor float updates use the same numerical
rules and envelope as Gate D-B.

## Gate E and resume authorization

Only D-A, D-B, and D2 pass authorizes five dual actor updates under the unchanged
H4096 recipe. Gate D-C blocks only on its structural conditions. Gate E records
the requested training, coverage, constraint decomposition, advantage
perturbation, dual-state, timing, utilization, and memory fields every step.
There is no diagnostic second optimizer step.

Gate E stops on verifier error/timeout, fingerprint mismatch, nonfinite, OOM,
more than one actor update, abnormal rollout count, lambda maximum, unsavable
actor/optimizer state, terminal-reward contamination, all-zero-driven loss of
dual control, missing resume state, or numerical collapse.

After step 5, resume must restore global step, actor, optimizer, scheduler,
lambda, violation EMA, observation count, last constraint step, fingerprints,
seed protocol, and dual config exactly, then execute one verification update.
That update is excluded from the five-step effect summary.

## Engineering overhead

Frozen off-versus-shadow incremental time, independent prefix verifier time,
dual logic time, full step wall time, and GPU memory delta are reported
separately. Unmatched single stochastic steps cannot establish overhead.

- no more than 4%: provisional pass;
- greater than 4% and no more than 8%: `PROFILE_AND_REPEAT`;
- greater than 8%: engineering no-go unless initialization or diagnostic I/O
  contamination is demonstrated.

Scientific batch or protected-set reductions and disabled required validation
are forbidden as performance remedies.

## Final decision and limitations

The maximum decision is `GO TO 50-STEP REVIEW`; this protocol never launches
50-step or 200-step training. That recommendation requires Gate C v2, D-A, D-B,
D2, Gate E, and resume to pass; zero verifier errors; finite non-max lambda; at
least one nonzero capability-signal step; controlled all-zero residual; one
actor update per step; unchanged rollout counts; intact fingerprints; and
acceptable or explicitly profile-pending overhead.

The report must not claim accuracy improvement, cross-seed effectiveness,
all-zero recovery, or adaptive OBCF. Exact, numerical, and statistical criteria
are never substituted for each other.
