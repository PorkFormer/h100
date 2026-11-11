# PPPO reproduction on the current h100 fork

This is an independent implementation of the public PPPO equations from
*Well Begun, Half Done* (arXiv:2512.15274). The HPC4 reproduction uses
Qwen3-4B-Base and an 8192-token total response cap. It is **not** the original
paper's thinking-model/10240-token experiment, and no paper-level quality
reproduction or speedup is claimed.

## Algorithm contract

`python -m verl.experimental.pppo.main_pppo` uses `pppo_trainer.yaml`.
The deployment recipes are prepared in
`/fs1/home/zhushoufei/rlvr/experiments/pppo_4b_20261108/configs`.

For each question there are eight original full responses. Every original
branch gets eight additional independent n=1 continuations (G=1 is debug only).
The original prompt token IDs and integer-floor prefix are passed directly to
the server; the new tail budget is `response_cap - prefix_length`.
R is original binary correctness plus the eight new binary correctness values.
The sample-standard-deviation GRPO advantage is computed across the eight
original branches, not across the continuations.

The loss is the question mean of prefix surrogate sums divided by the sum of
original response lengths for that question. Row weights `8 / D_q` are computed
before shuffling and carried through DataProto/TensorDict to the actual FSDP
loss. Each microbatch contributes `dp_size / global_mini_rows` times its weighted
sum, without a second microbatch division. Dual clip, KL, entropy and importance
correction are rejected. Standard PPO/NCBR paths remain opt-out defaults.

`actor_view=full_mask_reference` keeps complete original inputs and masks suffix
loss. `actor_view=cropped` physically shortens inputs after filtering and computes
old log probabilities on those inputs. Any zero-length prefix makes that logical
actor batch fall back to the reference path; its loss remains zero. Empty original
responses are rejected. Continuations are never actor rows.

## Runtime and data

The independent PPPO runtime reuses only the bounded lifecycle scheduler. It does
not change NCBR's cap-hit detection, K=1 aggregation or reward replacement.
Generation chunks have at most512 requests, concurrency128, scoring chunks256.
Every request has an occurrence/member/branch identity, deterministic SHA256 seed
and policy version. Timeout, malformed results, version mismatch and verifier
exceptions stop the update. No automatic retries are performed. Release and drain
acknowledgements precede actor work. Scoring metadata is whitelisted to exclude
stale scores. Audit files store scalar results and token hashes, not all long tails.

`prepare.py` first applies the same template and1024 prompt limit. It chooses256
unique question groups using seed42/SHA256 ordering and excludes all occurrences
of those question groups from training. Whitespace-normalized question identity
is used only for splitting; original prompts are retained unchanged. The manifest
contains source/split hashes and original row indices. Historical NCBR arms that
trained on the holdout are not strict paired comparisons.

## Schedule and recovery

The plateau completion rule is a disclosed implementation choice: full-response
greedy holdout accuracy every10 outer steps; three failures to improve trigger a
5-point increase, with one validation cooldown; stages15,20,25,30,35. Step0 only
initializes the baseline. Fixed25 is an ablation. No test benchmark controls eta.

PPPO checkpoints contain the native model/optimizer/extra shards and `data.pt`,
plus driver RNG and `pppo_state.json`. The latter records actual data epoch and
consumption, occurrence counter, schedule, hashes and checkpoint policy version.
The independent `pppo_latest.json` is authoritative; the generic verl latest text
file is not used for PPPO auto resume. A model-save-incomplete checkpoint is
rejected. A validation-pending explicit resume restores matching weights and
completes validation before committing. Committed validation events are idempotent.
All data-loader state is restored, including the sampler at an epoch boundary.
Natural vLLM sampling is not claimed to be bitwise identical after process restart.

## Acceptance and execution

Run CPU tests on a scheduler CPU node, and GPU checks on allocated GPUs. Never
start model work on a login node. The deployment uses the existing SIF, the supplied
squashfuse PATH prefix, `/fs1` binding and an isolated PYTHONPATH/Ray directory.

Required checks are equation/gradient normalization, real padding/loss dispatch,
branch lifecycle failures, crop/oracle native FSDP equivalence, complete optimizer
updates and fresh-process restore. The frozen-batch GPU gate records logprob,
loss, every local FSDP gradient shard and an AdamW update, reduced across ranks.
The CPU FP64 tolerance is atol1e-12/rtol1e-10. BF16 thresholds are fixed before
execution: prefix logprob max absolute error0.05; loss/gradient/update relative
error2%; nonzero gradient cosine0.999. Failed checks are not relabeled passes.

Short smoke uses16 questions, N=G=8 and mini8, three outer steps plus a fresh
process's fourth step. A G=1 one-step run precedes it. Profiling uses the final
256-question configuration and300-step LR horizon, with a separate one-step
stop hook. Profiling states are not used to initialize any formal experiment.
Per-rank update evidence, fixed holdout results and generated/optimized token
costs are separate acceptance results. See the deployment `STATUS.md` for actual
passed gates and scheduler states; this document describes the contract only.

## Verified implementation status

The eight-A800 full-mask smoke (job3857125) completed G=1 one step,
G=8 three steps, and a fresh-process restore followed by step4. All eight
ranks recorded finite optimizer evidence and nonzero sampled parameter changes;
optimizer counters continued after restore. This is execution evidence, not
evidence of a quality improvement. Fixed holdout results varied across restart.

BF16 cropping has **not** passed the frozen equivalence gate. An FP32/eager
attention/no-remove-padding diagnostic passed, but does not isolate the BF16
discrepancy. Use `algorithm.pppo.actor_view=full_mask_reference` for the accepted
training path; the YAML's cropped mode remains experimental. Physical input
cropping is an engineering optimization, not a demonstrated paper requirement.

HPC deployment scripts, resolved experiment recipes and results referenced
above live outside this Git worktree and are not included in this commit.
