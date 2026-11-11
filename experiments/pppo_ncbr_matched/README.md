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

All W&B logging is local/offline. Automatic approval review rejected online
configuration/metric disclosure to cartman/a100_verl; nothing is uploaded.
This operational difference is recorded in the frozen configuration audit.

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

## Historical status before the numerical repair (2026-11-11)

Gate 0 passed, and Gate 1 passed 205 tests. Native 16-rank BF16 Gate 2 failed:
prefix logprob error reached 0.291666 versus the fixed 0.05 limit, with local
gradient and AdamW update discrepancies exceeding the fixed 2% limits.
All subsequent gates and formal training are NOT RUN. See `reports/` for
committed logs, per-rank evidence, measured diagnostic costs and blocking status.

An independent diagnostic, explicitly ineligible for acceptance, can be run
after current Gate 0/1 receipts with:

```bash
python experiments/pppo_ncbr_matched/gpu_equivalence.py --trace-kernels \
  --output /path/to/new/diagnostic-directory
```

Its first-layer hooks show identical inputs but different BF16 Q/K/V linear
outputs across full/cropped matrix shapes, before attention. Deterministic
FlashAttention backward makes same-view repeated gradients and updates identical,
but does not make full/cropped equivalent. No precision change, threshold
relaxation, synthetic successful training status, or full-mask formal substitution
has been made. A validated shape-stable BF16 execution repair remains outstanding.

## Current status after the numerical repair

Gate 2 now passes all 16 ranks with the actual configured BF16 fixed-tile actor.
See `NUMERICAL_EXECUTION.md` for the execution repair and user-authorized logprob
tolerance revision, and `reports/validation_report.md` for current evidence.
Formal training remains NOT RUN pending Gates 3 and 4. Checkpoints remain every 50 steps.


## Automatic continuation and authorized email updates

`verl/examples/pppo/continue_ncbr_matched.py --run` attaches to the existing
G=1 driver without duplicating it, waits for successful exit, then sequentially
runs G=8, fresh-process resume, Gate 3 attestation, formal-scale profiling,
Gate 4 attestation, fresh Base S300 training and Step 300 evaluation. Every
training launcher retains its own gate/fingerprint checks. A source/config
change, stale G=1 heartbeat or failed child stops the chain. Only one coordinator
may hold its lock; existing automation evidence is never silently overwritten.

The user explicitly authorized progress email to `3531125084@qq.com`. The
coordinator uses the existing msmtp account and sends selected status summaries
at stage transitions, committed formal checkpoints, hourly while running and on
failure/completion. It sends no raw logs, model weights or credentials. Mail
receipts distinguish SMTP acceptance from inbox delivery. Existing gates and
G=1 are not rerun merely to install this orchestration-only wrapper.

Live state/logs/mail receipts: `outputs/pppo_ncbr_matched_16xa10040g_s300/automation/`.
Without `--run`, the wrapper checks prerequisites and prints status without
launching training or sending mail.

## Authorized repair boundary (2026-11-11)

The user authorizes engineering repairs and revalidation without repeated approval,
while preserving comparison with historical NCBR. Do not alter the frozen model /
tokenizer identity, source data and holdout split, question ordering/repetitions,
prompt/verifier/evaluation contract, response caps, N/G, train/candidate/mini/micro
batches, optimizer, schedule horizon, update counts, seeds, PPPO objective or eta
schedule to make a failed run succeed. Existing disclosed algorithm/data/numerical
execution differences remain disclosed; they do not authorize additional drift.

Prefer compatibility fixes, bounded request scheduling/chunking, communication,
metadata alignment and resource-management repairs. Preserve failed attempts,
review field-level differences and rerun affected gates before retrying. Runtime
or GPU-hour effects of execution changes must be reported separately. A required
change to a protected scientific condition is a blocker, not an automatic fallback.

The coordinator advances validated stages and reports failures; it does not
silently rewrite code/config or blindly retry a failed scientific experiment.
Engineering diagnosis and repair must precede restarting a stopped chain.


## Formal first-three-step monitoring

The user confirmed that the milestone is formal Steps 1–3, not smoke/profiling.
`verl/examples/pppo/monitor_first3.py` checks every 60 seconds and requires real
formal metrics for consecutive steps, 16 actual optimizer updates per step (48
cumulative), 2048 actor rows, 256 question groups and finite loss/gradient/timing
metrics, plus all prerequisite gate receipts and a live formal driver. It mails
the verified milestone and does not terminate the S300 process. Monitor state
and append-only observations live in `first3_monitor/` under the run root.

The first interactive-session driver and coordinator disappeared without a
successful outer step or logged Python exception. Their exact exit cause is not
established. The entire attempt is preserved in `interrupted/1794415797994109685`.
`start_detached_first3.py --run` verifies those processes are dead and no update
metrics exist before archival/restart. It launches driver, coordinator and monitor
with independent sessions, closed inherited descriptors and file-backed logs;
all three were verified reparented to PID 1. This lifecycle repair changes no
training/config fingerprint. Six monitor tests guard against counting smoke,
missing/duplicate steps, wrong update counts, nonfinite metrics and stale drivers.
