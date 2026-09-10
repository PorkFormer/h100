# NCBR minimal reward hook

Base: `bfa08860fee9f4febaf7aa1041f0e7eb0ac09cd5`.
Source oracle: `d23da0e17830c296ae6e375793a7ccea98870bb3`.
The port uses committed Git objects; the active training checkout's uncommitted files are excluded.

`NCBRHook.apply` runs the original continuation runtime and reward adapter on an independent copy.
Its explicit inputs are raw token scores, verifier extras, boundary configuration, policy version,
sampling parameters, a continuation client, and a scoring callback. It receives no algorithm or
optimizer configuration. The output contains effective token scores, row boundary/changed masks,
metrics, and auxiliary correction evidence. The trainer owns writing rewards, group statistics,
filtering, replica sleep, advantage estimation, and actor updates.

`corrected_mask` means a row's effective token scores actually changed, including dtype rounding.
The original `boundary_changed` diagnostic keeps its original tolerance and is a separate field.
Short correctness, task score, and shaped token rewards remain distinct. All detector-selected cap
hits continue, including short-correct rows. Correction is added only to the last valid H-prefix
token. Long responses are constructed in separate reward-only batches and never enter actor inputs.

## Configuration

```yaml
ncbr:
  enable: null  # null: preserve legacy mode; false: identity; true: require shadow/replace
actor_rollout_ref:
  rollout:
    boundary_return:
      mode: "replace"
      long_response_length: 8192
```

For the standard entry point, keep `algorithm.adv_estimator=grpo` and select the existing actor
policy loss (`vanilla` or `gspo`). The shared preflight preserves the source v1 restrictions,
including the registered DAPO verifier, single-turn vLLM, no critic, and sufficient context.
The dedicated DAPO entry still enforces its replacement-mode dynamic-filter requirement.
`ncbr.enable=false` bypasses auxiliary client acquisition and verifier calls.

## Call graphs

Original DAPO:

```text
publish version → H rollout → identity/masks → original short reward
→ candidate processing [original continuation → chunked long verifier → reward adapter]
→ dynamic group filter / accumulation → complete-group selection → replica sleep
→ balance → old/ref logprobs → original GRPO → original actor/loss → publish
```

Extracted DAPO:

```text
publish version → H rollout → identity/masks → original short reward
→ NCBRHook.apply → trainer writes rewards / group labels / accumulator
→ original dynamic group filter / accumulation → complete-group selection → replica sleep
→ balance → original old/ref logprobs → original GRPO → original actor/loss → publish
```

Standard `main_ppo → RayPPOTrainer.fit`, enabled:

```text
publish → H rollout → identity/masks → original balance → original short reward
→ NCBRHook.apply → effective reward → replica sleep
→ original old/ref logprobs → original reward/advantage → original actor/loss → publish
```

Disabled standard runs retain the original sleep immediately after ordinary rollout. A failed
continuation/verification prevents old/ref and actor work. Unattested remote cleanup prevents sleep.

## Reproduction

From the repository root, with existing dependencies:

```bash
python verl/tests/experimental/ncbr_portable/verify.py
```

The script chooses a fresh `/tmp` output directory, disables CUDA, limits CPU threads, injects the
isolated source path inside each Python process, disables pytest's cache, and forbids `ray.init`
in the test suite. It installs nothing. An explicit `--output-dir` must have unused log filenames.

The original 12 fit characterizations were frozen before production edits and repeated byte-for-byte
against the source, port, and hook. They exercise real GRPO and registered vanilla/GSPO loss and
gradients inside the source synthetic fit harness. External model/scoring services are mocked.
The additional 15 boundary characterizations were generated later from the original source, then
checked against both the recorded port commit and final hook without changing the original golden.
They cover nonzero four-way score transitions, locking/unlocking, no-cap, metadata fallback,
EOS-at-limit, padding, retained row order, and actual loss/gradients.

Final CPU suite: **190 passed**, including original DAPO/dynamic sampling regressions, actual standard
fit integration, disabled comparison to the exact baseline fit, raw/actor tensor invariants, GSPO
tail isolation, and failure cleanup. Golden comparisons are byte-exact; tensor assertions use
`torch.equal`. No new approximate tolerance was needed; source internal tolerances are preserved.

## Scope and limitations

- `core_algos.py` is unchanged from the base, including GRPO, vanilla and GSPO mathematics.
- No training, GPU inference, real Ray connection, optimizer update, or deployment was run. Loss
  gradients are computed on synthetic CPU log-probability tensors. Distributed worker packing,
  actual vLLM/Ray cleanup behavior, throughput and memory requirements remain unverified.
- The hook deep-copies the candidate to protect caller tensors and metadata. This can add memory
  overhead; no performance claim is made.
- The detector retains the source metadata priority, including explicit length winning over EOS;
  it is not a new strict-length detector.
- The tracked grouped client now acknowledges release as in the NCBR source. Ordinary rollout's
  `generate` path is unchanged; other users of `generate_grouped` also inherit acknowledged release.
- Generated reference configs include already-existing Probe Credit defaults that were absent from
  the baseline generated files. No new Probe Credit algorithm or default activation was added.
- Source node placement changes, FA-TR/FA-CAC implementations, launch tooling, and actor diagnostics
  performance modifications are excluded. Unused source overhead tooling was removed.

AI assistance: OpenAI Codex. Each implementation commit carries an attribution trailer.
