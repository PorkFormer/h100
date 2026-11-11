# PPPO + DAPO Dynamic Sampling (NCBR-Matched)

This branch starts at PPPO d1f0bdd and preserves the historical NCBR workspace.
Formal target: fresh Qwen3-4B-Base, 300 outer steps, 4800 optimizer updates per
rank, 2 x 8 A100-SXM4-40GB, BF16 FSDP with FP32 master parameters/AdamW state.
Rollout TP=2, actor SP=1, shared actor/rollout GPU pool (rollout nnodes remains 0).

## Deployment and execution

Use the existing allocated NCBR container: Python 3.12.3, torch 2.10.0+cu129,
vLLM 0.18.0, Ray 2.54.1, Transformers 5.3.0, flash-attn 2.8.3.
No new environment or extra GPU allocation is requested. The head is
10.8.191.127:6395; the user manages the worker at 10.8.191.131. See
`../../verl/examples/pppo/RAY_NCBR_MATCHED.md`. Cluster membership alone is
not an FSDP/NCCL/algorithm acceptance result.

From this worktree, with both allocated nodes ready:

```bash
export PYTHONPATH="$PWD/verl"
export XDG_CACHE_HOME=/tmp/pppo-matched-cache
export FLASHINFER_WORKSPACE_BASE=/tmp/pppo-matched-cache/flashinfer
python experiments/pppo_ncbr_matched/run_all.py
```

The sequential runner stops on any failed gate. Every subprocess has an
independent persisted log. It never starts/restarts workers, kills unrelated
processes, lowers N/G/caps/batches, or resumes formal training from diagnostics.
Do not rerun over an existing failed GPU evidence directory: inspect and preserve
it, then use a separate attempt directory for diagnostics. Only actual passing
evidence for the current configuration and source fingerprint unlocks training.

The formal launcher is `verl/examples/pppo/run_pppo_ncbr_matched_16xa10040g.sh formal`.
For recovery, supply `--resume` with this run's committed formal checkpoint.
All formal artifacts are under
`/workspace/rl/h100/outputs/pppo_ncbr_matched_16xa10040g_s300`.
Checkpoint/data/token-level large artifacts are not tracked by Git.

## Frozen protocol and differences

`prepare.py` retains the historical 100-fold repeated Parquet order; the source
SHA256 is 816b8778751076fdbc882ce65045aa80bb59db1618ce36e8308903db90ae92f4.
Question identity normalizes whitespace only for splitting. All copies of the
256 holdout identities are removed. Ambiguous-answer source identities remain
unchanged in training but cannot enter holdout. Exact normalized overlap with
the three frozen final test prompt sets is checked; no semantic paraphrase
deduplication guarantee is claimed. Holdout manifest includes all source rows,
question hashes, split seed, file hashes and source identity.

Historical NCBR did not exclude this holdout, so training samples are not
strictly identical. Its formal continuation resumed from Step 20; this PPPO
run initializes model and optimizer from Base. Data seed=null is retained from
the historical sampler; actor/FSDP seeds and continuation seed are recorded
individually. Extra epoch capacity allows worst-case dynamic refill without
ending before 300 outer steps; this does not change the original repeat count.

The original PPPO equations are retained: floor(eta * original response length),
original token prompt plus prefix, total response budget 8192, original binary
correctness plus eight independently seeded continuations, advantage across
eight originals, prefix-only clipped PPO, and original-question-total-length
normalization. No continuation is an actor row. Dual clip, NCBR overlong shaping,
boundary reward correction and auxiliary objectives are disabled.

Dynamic sampling uses accumulated correctness after all scoring. This is an
explicit sampling modification, not an unmodified original-paper protocol.
The plateau completion rule is a reproduction choice: greedy 256-question
holdout at Step 0 and every 10 steps, patience 3, min_delta 0, cooldown 1,
eta 15/20/25/30/35%, effective from the next step. Final test scores never
control eta. Every config difference is categorized in `config_diff.json`.

## Acceptance and limits

Gate 0 freezes and checks the full resolved YAML and dataset/model provenance.
Gate 1 runs PPPO, NCBR and portable CPU regressions. Gate 2 uses the native
16-rank FSDP engine, BF16 mixed precision/FP32 master weights, original-length
denominators and actual AdamW updates. Fixed gates: prefix logprob absolute
error <=0.05, loss/gradient/update relative error <=2%, gradient cosine >=0.999.
Suffix logprob gradients must be zero. FP32/eager diagnostics cannot pass Gate 2.
Gate 3 requires G=1 diagnostics, G=8 smoke, every-rank finite nonzero updates,
and a new Ray job restoring a committed checkpoint. Gate 4 uses 256/768
questions, N=G=8, 8192 cap, mini16 and a 300-step schedule with a separate stop
hook; a one-step checkpoint measures save costs without becoming a formal seed.

Audit parameter scans are opt-in and disabled for formal/profile runs. The
lightweight optimizer counter reads actual AdamW state and fails if any rank
skips a step. Formal training requires 2048 actor rows and 16 updates per step.
Checkpoint commit requires exactly all 16 rank model/optimizer/extra shards,
data cursor, driver RNG and PPPO sidecar; restoration is tested separately.

Only Step 300 receives downstream AIME24/AIME25/AMC23 evaluation: 8192 cap,
32 samples/question, frozen historical prompt/tokenizer/verifier and seeds.
Pass@1 (c/32 estimator) equals Avg@32 numerically; Pass@32 is any-success per
question. Question-level uncertainty and mean response lengths are reported.
Validation, diagnostic and offline evaluation costs are separate from training.
Optimized-token fraction is never reported as a GPU-hours speedup.

Actual completion is determined by evidence in `validation_report.md` and
`training_status.json`, not by this README or successful CPU tests.
