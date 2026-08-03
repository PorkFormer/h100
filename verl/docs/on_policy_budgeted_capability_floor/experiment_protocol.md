# OBCF Experiment Protocol

No GPU performance or scientific-effectiveness claim is valid until its corresponding real-data gate is completed and recorded.

## Pre-training gates

1. Build the Base floor cache from existing Base rollout/score artifacts. Record the cache fingerprint, Base weight hash, protected count, Base success-count histogram, floor distribution, verifier fingerprint, and source artifact hashes.
2. Run the offline signal simulator for Base, H2048 step 200, and H4096 step 200 artifacts. Record q/floor/deficit means, active/mixed/all-zero/all-one fractions, nonzero-gradient coverage, active-without-gradient coverage, deficit by Base success count, and expected prefix verifier calls. The 5% nonzero-gradient threshold is an engineering gate, not a theoretical threshold. Do not proceed when nearly every active group is all-zero.
3. Run one-step `off` equivalence from identical checkpoint, data, and seeds. Compare rollout, reward, advantage, actor checksum, optimizer checksum, and verify zero prefix calls.
4. Run two-step `shadow` equivalence and require actor and optimizer checksums to match `off`; only prefix metrics may differ.

## Active gates

5. Run a five-step dual smoke. Inspect deficit, mixed/all-zero coverage, lambda, EMA, residual, verifier time, total step time, entropy, and gradient norm.
6. Run a 50-step comparison of Vanilla DAPO, shadow OBCF, dual OBCF, and fixed-lambda OBCF (`dual_lr=0`).
7. Run the 200-step experiment only after all previous gates pass.

Failed smoke results must be reported as observed. Do not add Base trajectory replay, witness teacher forcing, continuation generation, forced answers, length reward, global KL, or rescue components to repair them. Do not claim GPU validation, throughput, memory, measured overhead, or effectiveness from CPU tests or unexecuted launchers.

## Reporting

Report the exact cache/config fingerprints and seed. Separate protected from Base-unsolved prompt results. Report all-zero active groups and active-without-gradient groups explicitly. Directional changes in the operational prefix event and terminal correctness are empirical results, not evidence of internal state preservation or a universal causal mechanism. One seed does not support a universal claim.
