# Eight-GPU enlarged NCBR retest

This run responds to the user's request to increase test scale after the first run's
coverage gaps. Production source remains37a812d; oracle d23da0e and disabled baseline
bfa0886 are unchanged. The previous validation commit is98eb7d3; its artifacts remain
untouched at `/tmp/ncbr_8gpu_20260911/validation/evidence`.

The new isolated root is `/tmp/ncbr_8gpu_scale_20260911`. Read `scale_plan.json` for the
prespecified budget. The model and tokenizer/template are unchanged. The1024 unique
prompt pool preserves the prior first128 exactly, then extends without replacement
using the continuing seed42 RNG. Full original model/data hashes and prefix identity
were rechecked before generation. No result-driven supplementation is allowed.

Full-budget natural capture:512 prompts, n4, four sequential128-prompt blocks, H2048,
L8192, all strict cap hits continued, K1. Eight UUID-bound TP1 replicas; each block
starts original model/seed42. Request order and derived continuation seeds are fixed.
Both independent continuation repeats are saved; token determinism is not presumed.
Original reward shaping, verifier, detector and sampling parameters are unchanged.

Trainer: nine DAPO/standard vanilla/standard GSPO off/shadow/replace cases, original
model each time, standard train/minibatch64 prompts, DAPO train/minibatch2 prompts, microbatch1 per GPU,
four actual AdamW steps maximum, lr1e-6. DAPO generates128 candidate prompts per round,
retaining its original filter and maximum four rounds. Case timeout remains30 minutes;
timeout is never PASS. Step count, nonzero gradient, changed shards, publication and
actual serving of changed weights are measured separately. No formal checkpoints,
benchmark evaluation or shared dependency changes. Ray/cache/IPC remain private.

The actor diagnostic freezes all real captured outputs and checks original adapter
versus hook first. It selects the first two naturally corrected UID groups, filling
with earliest other groups if fewer exist, and preserves all four rows per group.
This explicit coverage diagnostic is not a population-rate sample. Shadow must equal
off exactly. Replace may change original GRPO advantages and GPU gradients; if those
advantages differ, at least one gradient shard must differ. Per-mode repeats stay exact.

Commands (from this clone; run GPU stages sequentially):

```bash
mkdir -p validation/evidence
python validation/prepare.py > validation/evidence/prepare.log
python validation/gate.py
# torchrun --standalone --nnodes=1 --nproc-per-node=8 validation/nccl_probe.py,
# with CUDA_VISIBLE_DEVICES set to the eight recorded UUIDs
python verl/tests/experimental/ncbr_portable/verify.py --output-dir validation/evidence/cpu_scale
python validation/scale_capture.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/scale_followthrough.py
python validation/cleanup_owned.py
```

The executed freeze and NCCL receipts are in evidence. The previous fault and reference
checks remain evidence for unchanged production code; this retest targets increased
natural correction/gradient/filter/version coverage. Previous-run RESULTS/WORKLOG files
will be superseded by the new scoped report at completion. Never infer current PASS
from those inherited historical files.

The initial proposal increased DAPO required groups to64 as well; this was revised
before any trainer launch to retain the original2-group target while increasing
candidates to128. Initial and revised plans are both retained. No results from a
64-group DAPO trainer exist or are claimed. The frozen prompt manifest is unchanged.

Default actor backward repeated-gradient equality failed on the enlarged frozen input.
Both attempts are preserved. The standalone actor diagnostic now enables PyTorch
deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8; all eight rank repeats
passed exactly afterward, with190 CPU/two characterizations repeated. This does not
assert bitwise repeatability for the default real trainer.

Executed continuation after preserving the initial failures:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/remaining_cases.py --resume --continue-timeouts
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/replay_trainer_natural.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/select_actual_actor.py
python validation/launch_actor.py --h 2048 --tag _natural --source validation/evidence/replay_trainer_natural/actor_actual_input.pt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/scale_actor_compare.py --tag _natural
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/analyze_cases.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/scale_coverage.py
python validation/check_receipts.py
python validation/scale_report.py
python validation/cleanup_owned.py
python validation/finalize.py
```

Final results are in `RESULTS.md`: seven trainer cases completed four updates;
DAPO shadow/replace timed out with two/three updates. Four natural0→1 events were
observed in completed hook batches (DAPO replace2, GSPO shadow1, GSPO replace1),
producing three effective replacement corrections. In the conditional replay of
the exact actual version0 actor batch, both vanilla and GSPO changed four advantage
rows and all eight gradient shards; shadow equaled off exactly. Natural1→0 and1→1
remain uncovered. Final cleanup found no owned processes and all eight GPUs idle.
