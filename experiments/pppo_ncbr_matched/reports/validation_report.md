# PPPO + DAPO Dynamic Sampling (NCBR-Matched)

## Current acceptance status, 2026-11-11

| Gate | Result | Evidence |
|---|---|---|
| 0 | PASS | evidence/gate0_stable/result.json |
| 1 | PASS: 208 tests | evidence/gate1_stable/pytest.log |
| 2 | PASS: all 16 ranks | evidence/gate2_stable_acceptance/result.json |
| 3 | IN PROGRESS | G=1 Ray job 1a000000: dataset preprocessing; no completed update yet. G=8 and resume NOT RUN |
| 4 | NOT RUN | Full-scale 768/256, N=8/G=8 profiling required |

Formal training and Step 300 evaluation: **NOT RUN**. Checkpoint frequency remains 50.

Gate 2 used the actual configured actor backend, BF16 compute, FP32 masters,
world/DP size 16, and no diagnostic-only flags. All 16 ranks passed:

- old prefix logprob max error: 0
- current prefix logprob max error: 0.0539188385 (revised limit 0.06)
- loss relative error: 7.0758311e-7
- local gradient relative error: 0.0002877–0.0006521 (unchanged limit 0.02)
- AdamW update relative error: 0.0033433–0.0046195 (unchanged limit 0.02)
- gradient norm relative error: 9.2583178e-5
- suffix logprob gradient: 0; original-length denominator identical
- each path performed exactly one AdamW update; same-view repeats identical

Repairs: share frozen old log probabilities in the harness; use opt-in fixed
128-token BF16 linear tiles, FP32 inter-tile gradient accumulation and deterministic
FlashAttention backward in the actual actor. Full-mask still computes suffixes.
No algorithm, response cap, batch size, or model operand precision was changed.
Only the logprob tolerance was revised from 0.05 to 0.06 under explicit user
authorization; other numerical limits were retained. See NUMERICAL_EXECUTION.md.
The original 0.05 criterion would still fail; historical FAIL records are retained.

Code commit: `ecd383f64127cebc712a24004c04784105af0ff2`.

Config SHA256: `9db0a2d40765ecab2cb630c093b5e84e754f83b89f6294b23ee70a06f471ed33`.

All prior diagnostic failures and original analysis are in `evidence/validation_before_stable_repair.md` and their per-attempt evidence directories. No efficiency claim is made before Gate 4.


## Zero-prefix repair revalidation (2026-11-11 17:10 UTC)

Gate 0 PASS; Gate 1 210 tests PASS; Gate 2 all 16 ranks PASS on commit c811e8c7adcb47c3215c4a8e2adb68a43dba6f7e. New frozen batch original lengths include 2 and 6 (zero prefixes) and maximum 8192. No frozen config or numerical threshold changes. Max current logprob error 0.05391884; max gradient relative error 0.00063946; max AdamW update relative error 0.00473694. Zero-prefix guard tokens carry no PPO loss.

Previous successful G=1 and intentionally interrupted pre-update G=8 are preserved under revalidation_archive/zero_prefix_repair. Gate 3 is being repeated with corrected cropped execution. Formal training remains NOT RUN. Standalone scripted monitoring was disabled per user correction; current Codex session checks every five minutes.
