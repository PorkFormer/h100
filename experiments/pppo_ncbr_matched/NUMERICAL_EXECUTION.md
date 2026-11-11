# Gate 2 numerical repair

The frozen-batch harness now shares one immutable full-mask old-log-prob tensor
between views. Independently recomputed old probabilities are still compared.
Previously the two PPO ratios had different denominators, confounding the test.

Native BF16 GEMMs choose different reduction paths as the token dimension changes.
Trace evidence found identical first-layer projection inputs but different Q/K/V
outputs before attention. Disabling reduced-precision reduction alone was insufficient.

The opt-in actor backend uses fixed 128-token linear tiles, BF16 GEMM operands,
FP32 inter-tile weight-gradient accumulation, and deterministic FlashAttention
backward. Partial tiles are zero-padded locally and stripped before attention.
The full-mask reference computes all original suffixes with this same backend;
it is not cropped internally. Cropped inputs, token positions, attention masks,
original-length normalization and prefix-only PPO remain unchanged. This is an
execution difference from historical NCBR, explicitly listed in config_diff.json.
It is not a claim of bitwise equivalence to historical native cuBLAS kernels.

A single custom autograd boundary per linear is necessary for FSDP compatibility.
Its output has independent storage because verl scales logits in-place. Parameters
are not replaced; FSDP state-dict names and shapes remain unchanged. Actual fresh-
process checkpoint restoration remains a separate required Gate 3 check.

On the diagnostic 16-rank run at commit 2478c2c, all first-layer traced prefixes
and old log probabilities matched exactly. Current logprob maximum error was
0.05391884; gradient relative errors were 0.0002877–0.0006521 and AdamW update
relative errors 0.0033433–0.0046195. Same-view repeats matched exactly.

The user explicitly authorized modest threshold relaxation. Before the new
acceptance run, only logprob max-absolute tolerance is revised from 0.05 to 0.06.
Loss, gradient norm, gradient vector and AdamW-update limits remain 0.02; gradient
cosine remains at least 0.999. The change accommodates the measured residual
while retaining the stricter update/gradient checks. It is not a general bound
on BF16 error or proof for every future batch. Historical FAIL receipts retain
original thresholds and are never relabeled. Diagnostic receipts cannot unlock
training. A fresh run using the actual configured engine must pass.

Fixed tiling adds projection padding and launch overhead. Gate 4 must measure
actual runtime/memory on the unchanged formal workload. No throughput or GPU-hour
saving is claimed from this repair or from the optimized-token fraction.
