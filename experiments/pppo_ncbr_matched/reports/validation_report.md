# Validation report: PPPO-NCBR-Matched

Formal 300-step training is **NOT RUN**, blocked by BF16 cropped equivalence. No formal checkpoints or downstream PPPO scores exist.

| Check | Status | Evidence |
|---|---|---|
| Two-node Ray membership, idle 16 A100-40GB and selected shared files | PASS | evidence/cluster_preflight_before_gpu.json |
| Gate 0 resolved config, source/model/tokenizer provenance | PASS | evidence/gate0/result.json |
| Gate 1 CPU regression, 205 tests | PASS | evidence/gate1/result.json; evidence/gate1_pytest.log |
| Gate 2 native 16-rank BF16 FSDP | FAIL | evidence/gate2/result.json and rank_00..15.json |
| Gate 3 smoke and fresh-process resume | NOT RUN | Blocked by Gate 2 |
| Gate 4 full-scale profiling | NOT RUN | Blocked by Gate 2 |
| Formal 300 steps / 4800 optimizer updates | NOT RUN | training_status.json |
| Step 300 three-benchmark 8K evaluation | NOT RUN | evaluation_report.md |

## Fixed numerical thresholds and measured failure

BF16 logprob max abs <=0.05; loss/gradient/AdamW-update relative error <=0.02; nonzero gradient cosine >=0.999.
Measured old/current prefix logprob max abs: 0.255605698 / 0.291666031.
Local gradient relative errors: 0.052944–0.091506.
AdamW update relative errors: 0.301444–0.315170.
All 16 ranks used BF16 computation, FP32 master parameters, world size/DP size 16. Original denominators matched exactly; suffix logprob gradients were zero.
The frozen diagnostic covers original lengths 32,63,127,511,1024,2048,4096,8192 and eta=15%. It does not establish all-stage coverage or on-policy rollout readiness.

## Root-cause evidence and remaining work

First-layer prefix embedding and input-layernorm outputs match exactly. Q/K/V linear inputs match exactly, while Q/K/V outputs differ by up to 0.0009765625 / 0.00390625 / 0.001953125 respectively. The first observed divergence is before attention, consistent with shape-dependent BF16 GEMM rounding. Later layers amplify it. This is not evidence that attention contributes no additional error.
With FLASH_ATTENTION_DETERMINISTIC=1, same-view repeated full-mask forward, local gradients and AdamW deltas become identical on the recorded ranks; cropping still fails. This environment change is diagnostic-only, not silently applied to the formal recipe.
The optimizer mutation was postponed until all forward/backward passes completed; mismatch persisted, excluding intervening parameter updates as its cause.
A future repair must provide a shape-stable BF16 execution path for both reference and crop, then rerun the unchanged thresholds on all ranks and representative eta/length cases. No such repair is validated in this delivery. Full-mask remains a diagnostic reference only.

## Attempts and provenance

Initial Gate 2 failed before model computation due to the harness using OmegaConf structured dataclass construction; repaired to the native Hydra factory. That failure remains in evidence/gate2_attempt1_schema_error.
First numeric failure remains in evidence/gate2_attempt2_bf16_mismatch. The mutation-isolated repeat is evidence/gate2. Kernel tracing is evidence/gate2_kernel_trace and is not acceptance-eligible.
Formal config SHA256: a481187f0e81ff2ad90a6d0e113144cc946cd39066d7b2674b3ebc400675f58e
Numeric gate code commit: 770d4d5a5d67d3a8f713b7c73ededf8a1118b037
Kernel diagnostic code commit: 501aaf1f4cbbbad2f4ac4b8f782ed7c28e108003
Ray head was retained on 10.8.191.127:6395; worker startup belongs to the user. No worker was restarted. The existing head object store is 1 GiB versus historical 200 GB; full-scale capacity is unverified.
Automatic approval review rejected online W&B disclosure. All subsequent runs used offline/disabled W&B; no online experiment upload was performed.

Checkpoint restoration, vLLM policy synchronization, continuation capacity, 300-step execution and downstream evaluation remain unverified. CPU success and frozen FSDP updates do not replace these gates.


## BF16 reduction diagnostic (2026-11-11 16:02 UTC)

Real 16-rank FSDP rerun with BF16 operands, FP32 masters, deterministic FlashAttention backward and `allow_bf16_reduced_precision_reduction=False`: **FAIL**, diagnostic only, not acceptance eligible. Original thresholds unchanged. Configuration SHA is unchanged. Diagnostic source fingerprint and parent commit are in the receipt; code is committed in 57cb5b1.

Per-rank ranges:

```json
{
  "old_logprob_max_abs": [
    0.2198925018310547,
    0.2198925018310547
  ],
  "current_logprob_max_abs": [
    0.2315502166748047,
    0.2315502166748047
  ],
  "loss_relative_error": [
    0.00019407082456519912,
    0.00019407082456519912
  ],
  "grad_norm_relative_error": [
    0.007243936572212089,
    0.007243936572212089
  ],
  "gradient": [
    0.020919077567130823,
    0.04070539707008496
  ],
  "update": [
    0.1800437167686313,
    0.20111249402963605
  ]
}
```

Same-view repeated forward/gradients/updates are identical on every rank. Full/cropped Q/K/V projection differences remain before attention. Reduced-precision reduction contributes to the discrepancy but is not its sole cause. This setting is not promoted into formal training. No justified relaxed threshold has been established; the user permits reasonable relaxation, but this run does not establish equivalence. Gates 3/4, formal training and Step 300 evaluation remain NOT RUN.

Wall time: 135.5732250213623 seconds; allocated diagnostic GPU-hours: 0.6025476667616102.
