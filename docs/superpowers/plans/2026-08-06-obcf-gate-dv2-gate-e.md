# OBCF Gate D v2 / Gate E execution plan

Date: 2026-08-06

Branch: `fix/obcf-gate-d-semantic-equivalence-v2`

Reported starting commit: `698f8b70ab96ed7fbf5298e550b08da300bb992f`

New artifact root: `/workspace/rl/h100/analysis/obcf_gate_dv2_e_20260806`

The prior `analysis/obcf_gate_cde_resume_20260805` tree is immutable evidence. In
particular, the failed cross-launch exact Gate D decision remains failed and is
never overwritten. Gate D v2 is a protocol amendment, not a reinterpretation of
that result.

## Invariants and scientific scope

- Preserve the original H4096 recipe, data order, dynamic filtering, reward and
  verifier semantics, optimizer, learning rate, clipping, FSDP, dtype, TP=2,
  2x8 GPUs, train batch 256, generation batch 768, and `rollout.n=8`.
- Do not add per-request sampling seeds when none exist; record `null` and
  `MISSING_EXPLICIT_REQUEST_SEED` instead.
- Diagnostics are opt-in and default off. They do not mutate DataProto, ordering,
  synchronization, sampling, filtering, or actor inputs.
- The diagnostic frozen-batch harness is not a training entrypoint and cannot be
  used for 50/200-step training.
- Do not change the OBCF objective, capability floors, dual mathematics, reward,
  verifier, protected set, or scientific batch to obtain determinism or speed.
- This execution stops after a five-update Gate E smoke and one resumed update.
  It may recommend a 50-step review, but never starts 50-step, 200-step,
  multi-seed formal training, adaptive OBCF, Base witness replay, or new ablations.

## File-level change list

Planned code and tests (exact names may be refined only when repository routing
requires it):

- `docs/obcf_gate_d_semantic_equivalence_v2.md`: protocol amendment and exact,
  numerical, and statistical acceptance rules.
- `verl/verl/trainer/ppo/nondeterminism_diagnostics.py`: atomic, rank-safe,
  fail-closed boundary writer, identity schema, manifest binding, and duplicate
  identity audit.
- Existing DAPO trainer/task-runner integration files: opt-in calls at boundary 0
  through boundary 3 without mutating DataProto or control flow.
- Trainer configuration schema/YAML: diagnostics enable flag and output/run IDs,
  default disabled.
- `verl/verl/trainer/ppo/frozen_batch_harness.py` (or the nearest existing
  diagnostic package): manifest verification, immutable frozen-batch loading,
  mode validation, exactly-one-update orchestration, and result manifests.
- `verl/tests/trainer/ppo/test_nondeterminism_diagnostics.py`: disabled/no-file,
  four-boundary, write-failure, duplicate identity, DataProto immutability, and
  missing explicit seed tests.
- `verl/tests/trainer/ppo/test_frozen_batch_harness.py`: batch/checkpoint mismatch,
  off/shadow counters, exactly-one-update, no-rollout, input immutability, and
  invalid-mode tests.
- `analysis/obcf_gate_dv2_e_20260806/commands/`: immutable launch and aggregation
  scripts used for this run.
- `analysis/obcf_gate_dv2_e_20260806/gate_d_v2_protocol.json`: preregistered
  protocol and tolerances, written before comparisons.
- The remaining requested JSON, Parquet/CSV, Markdown, log, resource, hash, Git,
  bundle/patch, and archive artifacts under the new artifact root.

No unrelated refactor is permitted. Each independent component is committed
separately with the repository-required attribution trailer.

## Frozen-batch harness data flow

```text
prior post-filter DataProto dump + source manifest
                 |
                 v
        schema/hash/source audit  ----fail----> BLOCKED
                 |
                 v
 immutable frozen-batch package + initial model/optimizer/scheduler identity
                 |
          fresh independent launch
                 |
       +---------+---------+
       |         |         |
   baseline     off      shadow
       |         |         |
       +---- exactly one actor update
                 |
                 v
 pre-update exact snapshots + losses/gradients/delta-theta/optimizer tensors
                 |
                 v
 baseline envelope -> dtype-aware comparisons -> Gate D-B / Gate D2 decision
```

The harness does not call rollout or dynamic filtering. Every launch validates
the frozen DataProto full hash and initial state identity before loading inputs,
records input hashes before and after, and fails closed on mode/schema/hash/count
mismatch. The baseline, off, and shadow launches begin from the same actor,
optimizer, scheduler, and frozen DataProto. A launch performs exactly one actor
update and writes atomically to a unique run directory.

## Three gate classes

### Exact

Exact comparison covers resolved scientific config and initial-state identities;
frozen prompt/response token IDs; labels; masks; discrete advantage/return input
structure; update/rollout counts; cache and protocol fingerprints; verifier
errors/timeouts; all OBCF-off side-effect counters; shadow lambda/update counts;
shadow terminal/total-advantage identity; schemas, shapes, dtypes, and actor input
field names. No tolerance applies to these values.

### Numerical

Float loss, old logprobs, gradients, parameters, optimizer tensors, and scalar
training metrics use the preregistered `torch.testing.assert_close` tolerances:

| dtype | rtol | atol |
|---|---:|---:|
| float64 | 1e-7 | 1e-9 |
| float32 | 1e-5 | 1e-7 |
| bfloat16/float16 | 5e-3 | 5e-4 |

Every tensor also reports max absolute error, max relative error, relative L2,
mismatch fraction, and cosine. Three duplicate baseline launches define the
baseline numerical envelope. If baseline itself exceeds the base tolerance, the
tensor is labelled `BASELINE_NONDETERMINISTIC`; comparisons may pass only when
they remain inside the preregistered baseline envelope. Tolerances are never
relaxed after seeing a result. Parameter comparison is on
`delta_theta = theta_after - theta_before`, not only checkpoint hashes.

### Statistical

Independent stochastic launches compare request coverage, response/reward and
response-length distributions, retained-set overlap, training/timing/resource
metrics, baseline natural variability, effect sizes, and descriptive confidence
intervals. Response IDs, retained prompts, rewards, losses, and checkpoint hashes
are not exact gates across launches. This layer blocks only configuration,
control-flow, off-side-effect, count, or clear systematic-shift defects; a
non-significant p-value is not treated as equivalence proof.

## TDD and staged execution

1. Preregister the protocol JSON and documentation before result inspection.
2. Write failing diagnostics unit tests for disabled behavior, all four
   boundaries, atomic failure, duplicate identities, immutable DataProto, and
   missing explicit seed. Confirm the intended RED failures.
3. Implement the smallest opt-in writer/integration, run focused tests, then the
   relevant trainer test slice. Commit diagnostics independently.
4. Run two serial one-step duplicate baselines with instrumentation. Compare
   boundary 0 first and stop at the first divergent boundary; do not modify
   generator or seed rules. Commit only code/protocol, not reinterpretations.
5. Audit prior dumps and manifests. If incomplete, run one unselected baseline
   step and atomically freeze the post-filter/pre-update DataProto plus initial
   identities. Never resample to select a favorable batch.
6. Write failing frozen-harness tests for all eight required failure/invariant
   cases; confirm RED. Implement only diagnostic replay, verify focused tests,
   and commit the harness independently.
7. Run off once for Gate D-A and require exact structural/config invariants.
8. Run baseline A/B/C and off A/B/C serially, build the baseline numerical
   envelope, compare exact pre-update semantics and tensor updates, and decide
   Gate D-B.
9. Reuse prior stochastic runs and add at most one baseline and two off launches,
   serially, to produce Gate D-C descriptive diagnostics.
10. Only if D-A and D-B pass, run off A/B/C and shadow A/B/C on identical frozen
    inputs and decide Gate D2.
11. Only if D2 passes, execute exactly five dual actor updates using the original
    H4096 recipe, applying every Gate E stop condition. Save step-5 state.
12. Reload the step-5 checkpoint, exact-check resume state, execute one additional
    verification update, and decide resume integrity.
13. Generate the Chinese final report, decisions, inventory, SHA-256 listing,
    Git bundle/patch, archive, and push status. Do not run longer training.

## Stop conditions by phase

- **Preflight:** stop if HEAD differs from the reported commit, tracked changes
  overlap the work, prior Gate C/cache fingerprints fail, or required cluster
  topology/model/data are unavailable.
- **Instrumentation:** stop implementation on any default-on behavior, DataProto
  mutation, half-written artifact, concurrent overwrite, or sampling/filtering
  change.
- **Frozen acquisition:** mark `BLOCKED` if required actor inputs or a verifiable
  initial actor/optimizer/scheduler state cannot be reconstructed.
- **Gate D-A:** stop downstream gates and mark `FAIL` for any side effect/count or
  non-allowlisted scientific config difference.
- **Gate D-B:** stop D2/E for discrete mismatch, OBCF path entry, update-count
  change, beyond-envelope float mismatch, or unexplained first divergence;
  `BLOCKED` applies to unreplayable input or unstable baseline envelope.
- **Gate D-C:** block only for systematic structural/config/count/side-effect
  evidence, never merely different stochastic responses or retained sets.
- **Gate D2:** stop E for exact shadow invariant failure, verifier/cache failure,
  or beyond-envelope actor-update mismatch.
- **Gate E:** stop immediately for verifier error/timeout, fingerprint mismatch,
  nonfinite, OOM, duplicate actor update, abnormal rollout count, lambda max,
  unsavable state, terminal-reward contamination, all-zero-driven dual loss of
  control, missing resume state, or numerical collapse.
- **Resume:** stop and fail for any required state mismatch, manifest/hash defect,
  or invalid one-step transition.
- **Overhead:** <=4% is provisional pass, 4--8% is `PROFILE_AND_REPEAT`, and >8%
  is engineering no-go unless initialization/diagnostic I/O contamination is
  demonstrated. No recipe reduction is allowed.

## Expected outputs

The new root contains the four boundary phase-1 directories and comparisons;
the frozen package/manifests/configs/tests/resource usage; Gate D-A, D-B, D-C,
D2, Gate E, and resume JSON/Parquet/CSV/Markdown evidence; per-step dual,
coverage, constraint decomposition, advantage perturbation, timing, memory, and
state-integrity records; `final_report.md`; `final_decision.json`; immutable
commands and logs; `generated_files.txt`; `sha256sums.txt`; `git_state.txt`; a
tar.gz archive; and a Git bundle or patch when GitHub credentials prevent push.

The maximum successful recommendation is `GO TO 50-STEP REVIEW`. The report must
state that 50/200-step training was not run and that accuracy improvement,
cross-seed effectiveness, all-zero recovery, and adaptive OBCF remain unproven.
