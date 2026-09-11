# Completed eight-GPU validation, 2026-09-11

User authorized adapting the full Qwen3-1.7B NCBR plan to all eight current GPUs.
Isolation root: `/tmp/ncbr_8gpu_20260911`; production source unchanged at `37a812d`.
Oracle `d23da0e`, disabled baseline `bfa0886`. Frozen model/data/128 prompts retained.

Final results: see `evidence/REPORT.md`, `case_analysis.json`, `receipt_checks.json`.
All environment, 190 CPU tests/two characterizations, frozen replay, eight-rank actor,
GSPO tail isolation, reference alignment, two live oracle diagnostics and six fault
barriers passed. Nine main trainer cases completed their configured budgets:
standard six cases each had two AdamW steps (first zero gradient, second changed
all eight shards); DAPO off/shadow had zero steps, replace one nonzero step before
candidate exhaustion. Standard final version2 publication is recorded, but no next
rollout at version2 ran. DAPO replace served version1 after its nonzero update.
Natural nonzero corrections and correctness transitions beyond0-to-0 remain uncovered.
No generated-token determinism, accuracy, scaling or speedup claim.

Private runner/environment corrections retained with original failure logs:
- vLLM UUID parsing: lazy process-local UUID-to-NVML-index shim.
- Early eager imports interfered with Ray GPU assignment; lazy loader resolved this.
- Private short Ray/IPC paths; CPU placement capacity48; seed in typed engine kwargs.
- Actor replay initial rollout_n omission corrected and CPU gates repeated.
- Transient occupancy snapshots blocked launches. Final release-fault first gate
  archived as `_gate_attempt1`; launch now requires two idle snapshots and an empty
  compute-process list. Fresh CUDA gate and190 CPU tests/two characterizations
  passed again; `fault_suite.py --resume` completed only the remaining release case.

All fault cases have no updates or new publication attempts. Release failure leaves
services awake until owned case teardown; no sleep after injection. Final inspection
found no owned processes and all eight GPUs idle; no forced cleanup was necessary.
Original checkout/shared model/dependencies/artifacts were not modified. No formal
checkpoint or benchmark evaluation. Raw evidence and private caches are untracked.
