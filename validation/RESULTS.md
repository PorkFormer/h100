# Enlarged Qwen3-1.7B eight-GPU NCBR retest

Overall verdict: NOT ALL PASS. The default actor repeated-gradient gate failed twice; the explicitly deterministic diagnostic is reported separately. Trainer failures or incomplete cases: train_dapo_vanilla_replace_v6: FAIL_TIMEOUT; train_dapo_vanilla_shadow_v6: FAIL_TIMEOUT.

Production code remains `37a812d`, oracle `d23da0e`, disabled baseline `bfa0886`. Model `/workspace/models/Qwen3-1.7B-Base`; all eight UUIDs and shared model/data hashes were rechecked. Previous artifacts at `/tmp/ncbr_8gpu_20260911/validation/evidence` are unchanged. See `scale_plan.json`, `freeze_checks.json`, `data_identity.json`, `manifest.json` and per-case commands/configurations.

## Scale and interpretation

The frozen pool contains1024 unique prompts, preserving the previous first128 exactly. Capture uses first512 prompts ×4 short rollouts, in four fixed128-prompt blocks, H2048/L8192. Sampling, verifier, reward shaping and detector are unchanged. All cap hits receive K1 continuation; two independent repeats are retained.

Trainer cases use four optimizer steps maximum. Standard train/minibatch64 prompts; DAPO retains original train/minibatch2 prompts but increases each candidate round to128 prompts, maximum four rounds. The initial proposal of64 required DAPO groups was revised before any trainer launch after the first capture block showed sparse correct rewards. Both plans are retained; no64-group DAPO result is claimed. Data/reward/filter mathematics were not changed.

Previous standard GSPO replace first rollout had one unique trajectory among four branches for each of its two prompts; the second had two unique trajectories per prompt. This is evidence of small-batch diversity limitations, not proof that size is the only cause. Current group diversity and natural verifier availability are reported separately.

## Frozen natural capture

- Rows: 2048; cap hits: 78; short verifier-correct rows: 20; invalid extracted answers: 947.
- Natural cap transitions: {'0->0': 78, '0->1': 0, '1->0': 0, '1->1': 0}. Nonzero effective corrections: 0.
- Frozen original runtime/adapter versus hook: PASS, exact scores, masks, request calls and retained row order. Independent repeat differences: 14; no generated-token determinism claim.

## Trainer cases

| Case | Execution | Steps | Nonzero changed-weight versions | Served after nonzero update | Natural task deltas | Effective corrected rows | Mixed short-reward groups | Seconds |
|---|---|---:|---|---|---:|---:|---:|---:|
| train_dapo_vanilla_off_v6 | PASS | 4 | [1, 2, 3, 4] | [1, 2, 3] | 0 | 0 | 43 | 324.4 |
| train_dapo_vanilla_replace_v6 | FAIL_TIMEOUT | 3 | [1, 2, 3] | [1, 2, 3] | 2 | 2 | 36 | 1809.0 |
| train_dapo_vanilla_shadow_v6 | FAIL_TIMEOUT | 2 | [1, 2] | [1, 2] | 0 | 0 | 22 | 1809.0 |
| train_standard_gspo_off_v6 | PASS | 4 | [1, 2, 3, 4] | [1, 2, 3] | 0 | 0 | 20 | 373.8 |
| train_standard_gspo_replace_v6 | PASS | 4 | [1, 2, 3, 4] | [1, 2, 3] | 1 | 1 | 15 | 1566.2 |
| train_standard_gspo_shadow_v6 | PASS | 4 | [1, 2, 3, 4] | [1, 2, 3] | 1 | 0 | 18 | 1361.7 |
| train_standard_vanilla_off_v6 | PASS | 4 | [1, 2, 3, 4] | [1, 2, 3] | 0 | 0 | 17 | 371.1 |
| train_standard_vanilla_replace_v6 | PASS | 4 | [1, 2, 3, 4] | [1, 2, 3] | 0 | 0 | 23 | 1474.1 |
| train_standard_vanilla_shadow_v6 | PASS | 4 | [1, 2, 3, 4] | [1, 2, 3] | 0 | 0 | 17 | 1141.8 |

Off bypasses the hook, so its hook-based correction counts are not measured. Short reward groups use hook inputs when present, otherwise full DAPO candidate batches or standard actor batches; the source is recorded per case. All-case token diversity is in `scale_coverage.json`. DAPO exhaustion is UNCOVERED, not a passed full-update case. A published version counts as served only when normal rollout metadata records it. Final publication alone does not prove serving.

## GPU actor diagnostic

Observed comparisons: `{'vanilla': {'status': 'PASS', 'shadow_exact': True, 'replace_changed_advantage_rows': 0, 'replace_changed_gradient_shards': 0, 'natural_gradient_effect': 'UNCOVERED'}, 'gspo': {'status': 'PASS', 'shadow_exact': True, 'replace_changed_advantage_rows': 0, 'replace_changed_gradient_shards': 0, 'natural_gradient_effect': 'UNCOVERED'}}`.
Default GPU actor replay failed twice: repeated off-mode gradients differed beyond the fixed tolerance (maximum observed absolute difference0.000732421875), while inspected full loss inputs were exact. Both failed attempts remain saved. Explicit PyTorch deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 in the standalone diagnostic restored exact repeats. This is not a claim of default trainer bitwise reproducibility; production code/configuration was not changed by that diagnostic fix.
Actor selection is explicitly coverage-directed within the fixed natural capture: first two corrected UID groups, filling with earliest others; all four rows per group retained. Selection is recorded in `actor_h2048/selection.json`. Registered vanilla/GSPO losses and gradients repeat exactly. Shadow must match off. Replacement is allowed to change original GRPO advantages and gradients; an observed advantage change must change at least one gradient shard. No synthetic verifier outputs are used in this retest.

## Actual natural correction and actor credit

Selected real service trace: `{'hook_output': 'validation/evidence/train_dapo_vanilla_replace_v6/worker_audit/345720_1789113334118124304_hook_output.pt', 'policy_version': 0, 'corrections': [{'row': 82, 'uid': 'b8a2d358-a0cc-5f16-ba73-54b3845b434b', 'trajectory_id': 'b8a2d358-a0cc-5f16-ba73-54b3845b434b:2', 'short_acc': 0.0, 'long_acc': 1.0, 'short_task_score': -1.0, 'long_task_score': 1.0, 'delta': 2.0, 'changed_token_positions': [2047], 'valid_H_tokens': 2048, 'group_entered_actor': True, 'trajectory_entered_actor': True}]}`. Original verifier replay, oracle/hook scores and filtering passed exactly. Actual actor prefix tokens, effective scores and recomputed original GRPO advantages are exact.
Natural-event GPU counterfactual: `{'vanilla': {'status': 'PASS', 'shadow_exact': True, 'replace_changed_advantage_rows': 4, 'replace_changed_gradient_shards': 8, 'natural_gradient_effect': 'COVERED'}, 'gspo': {'status': 'PASS', 'shadow_exact': True, 'replace_changed_advantage_rows': 4, 'replace_changed_gradient_shards': 8, 'natural_gradient_effect': 'COVERED'}}`.
This event was observed in the prespecified trainer run, not fabricated or added to the512-prompt capture cohort. The exact actually retained8-row version0 actor batch, including original old_log_probs, is evaluated with original weights; all input tensors were matched before replay. No new generation or altered verifier is used for the conditional diagnostic. Other unobserved transitions remain uncovered.

## Limits and artifacts

Unobserved natural transitions remain uncovered. An absence of corrections does not establish that the hook was never called: cap-hit/continuation, nonzero verifier delta, filtering and gradient effects are distinct gates. These are fixed-sample implementation diagnostics, not an accuracy benchmark or training-effectiveness/capacity/performance claim.
Original190 CPU tests and both frozen characterizations passed again in `cpu_final/`; fresh CUDA/NCCL gates, raw token/verifier/actor/gradient data and source provenance are retained. Previous reference and fault-barrier validation applies to unchanged production code and is not represented as newly executed here. No formal checkpoint, benchmark evaluation, shared dependency change or production fix.

Final process/GPU state: `final_state.json`; source/model/runner hashes: `final_manifest.json`; execution timings and CUDA allocation metrics: `case_analysis.json`; sampled inference NVML peaks: `h2048_processes.json`. First failures remain preserved.
