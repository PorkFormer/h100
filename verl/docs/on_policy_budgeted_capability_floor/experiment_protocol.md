# OBCF Experiment Protocol

No GPU performance or scientific-effectiveness claim is valid until its corresponding real-data gate is completed and recorded.

## Pre-training gates

The strict order is: event equivalence → floor/actionability simulation → full Base floor cache → off equivalence → shadow equivalence → five-step dual smoke → 50-step method smoke → 200-step multi-seed experiment. A later gate cannot repair or waive an earlier failure.

1. On the matched 512-prompt Base artifacts, run exact event equivalence through the current `RewardLoopWorker`. Require all 4096 identities, zero mismatch in either direction, and zero historical/recomputed errors. Resolve every mismatch before building any training cache.
2. Build a temporary schema-v2 cache with `B=2048`, `n0=8`, `support_threshold=2`, and `reference_tolerance_count=0`. Require zero structurally inert protected prompts for current `rollout.n=8`.
3. Run the offline signal simulator for Base, H2048 step 200, and H4096 step 200 with tolerance 0 and tolerance 1. Record q/floor/deficit means, active/mixed/all-zero/all-one fractions, nonzero-gradient coverage, active-without-gradient coverage, structurally inert coverage, retained/delayed/lost, and all metrics by Base success count. The 5% nonzero-gradient threshold is an engineering gate, not a theoretical threshold. Proceed only when tolerance 0 has zero inert floors, at least 5% global nonzero-gradient coverage, no errors/non-finite values, and H4096 shows more expected deficit/delay signal than H2048.
4. Only after the matched audit passes, roll out all training prompts with eight Base samples and build the full schema-v2 cache. Archive the manifest, prompts, audit report, hashes, event-equivalence attestation, and resolved reward configuration.
5. Run one-step `off` equivalence from identical checkpoint, data, and seeds. Compare rollout identities/token IDs, rewards, old log probabilities, advantages, actor and optimizer checksums, global step, and dataloader state. Verify zero prefix calls.
6. Run two-step `shadow` equivalence and require actor parameters, optimizer state, and advantages to match `off`; lambda and dual observation count remain zero. Prefix metrics and verifier wall time may differ, but rollout count and actor forward/backward/update counts may not.

## Active gates

7. Run a five-step dual smoke with tolerance 0. Inspect deficit, mixed/all-zero coverage, nonzero-gradient and active-without-gradient coverage, lambda, EMA, residual, verifier time, total step time, memory, entropy, and gradient norm. Step 1 composes the actor advantage with the pre-update `lambda=0`; lambda changes only after that protected observation.
8. Run a 50-step comparison of Vanilla DAPO, shadow OBCF, dual OBCF, and fixed-lambda OBCF (`dual_lr=0`). Do not activate tolerance 1 unless its simulator actionability gate passes.
9. Run the 200-step Vanilla-versus-dual experiment with at least three seeds per method only after all previous gates pass. Use prompt-level paired bootstrap and keep all rollouts for a prompt together.

Failed smoke results must be reported as observed. Do not add Base trajectory replay, witness teacher forcing, continuation generation, forced answers, length reward, global KL, or rescue components to repair them. Do not claim GPU validation, throughput, memory, measured overhead, or effectiveness from CPU tests or unexecuted launchers.

## Reporting

Report the exact cache/config fingerprints and seed. Separate protected from Base-unsolved prompt results. Report all-zero active groups and active-without-gradient groups explicitly. Directional changes in the operational prefix event and terminal correctness are empirical results, not evidence of internal state preservation or a universal causal mechanism. One seed does not support a universal claim.
