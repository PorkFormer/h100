"""Report observed scale effects without equating missing coverage to a pass."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'evidence'
def read(s):return json.loads((E/s).read_text())
cases=read('case_analysis.json');coverage=read('scale_coverage.json');replay=read('replay_h2048_v2/summary.json');actors=read('scale_actor_comparison.json');natural=read('replay_trainer_natural/summary.json');natural_gpu=read('scale_actor_comparison_natural.json')
assert len(cases)==len(coverage)==9
bycase={x['case']:x for x in coverage}
failures=[x['case']+': '+x['status'] for x in cases if x['status']!='PASS']
t=read('replay_h2048_v2/replace_transitions.json');transition={f'{a}->{b}':sum(bool(h) and x==a and y==b for h,x,y in zip(t['hit_cap'],t['short'],t['long'])) for a in [0,1] for b in [0,1]}
verifier=read('replay_h2048_v2/short_verifier.json')
lines=['# Enlarged Qwen3-1.7B eight-GPU NCBR retest','',
'Overall verdict: NOT ALL PASS. The default actor repeated-gradient gate failed twice; the explicitly deterministic diagnostic is reported separately. Trainer failures or incomplete cases: '+('; '.join(failures) if failures else 'none')+'.','',
'Production code remains `37a812d`, oracle `d23da0e`, disabled baseline `bfa0886`. Model `/workspace/models/Qwen3-1.7B-Base`; all eight UUIDs and shared model/data hashes were rechecked. Previous artifacts at `/tmp/ncbr_8gpu_20260911/validation/evidence` are unchanged. See `scale_plan.json`, `freeze_checks.json`, `data_identity.json`, `manifest.json` and per-case commands/configurations.', '',
'## Scale and interpretation','',
'The frozen pool contains1024 unique prompts, preserving the previous first128 exactly. Capture uses first512 prompts ×4 short rollouts, in four fixed128-prompt blocks, H2048/L8192. Sampling, verifier, reward shaping and detector are unchanged. All cap hits receive K1 continuation; two independent repeats are retained.', '',
'Trainer cases use four optimizer steps maximum. Standard train/minibatch64 prompts; DAPO retains original train/minibatch2 prompts but increases each candidate round to128 prompts, maximum four rounds. The initial proposal of64 required DAPO groups was revised before any trainer launch after the first capture block showed sparse correct rewards. Both plans are retained; no64-group DAPO result is claimed. Data/reward/filter mathematics were not changed.', '',
'Previous standard GSPO replace first rollout had one unique trajectory among four branches for each of its two prompts; the second had two unique trajectories per prompt. This is evidence of small-batch diversity limitations, not proof that size is the only cause. Current group diversity and natural verifier availability are reported separately.', '',
'## Frozen natural capture','',
f"- Rows: {replay['rows']}; cap hits: {replay['cap_hits']}; short verifier-correct rows: {sum(verifier['acc'])}; invalid extracted answers: {sum(x=='[INVALID]' for x in verifier['pred'])}.",
f"- Natural cap transitions: {transition}. Nonzero effective corrections: {replay['modes']['replace']['changed_rows']}.",
f"- Frozen original runtime/adapter versus hook: {replay['status']}, exact scores, masks, request calls and retained row order. Independent repeat differences: {len(replay['independent_repeat_differences'])}; no generated-token determinism claim.", '',
'## Trainer cases','',
'| Case | Execution | Steps | Nonzero changed-weight versions | Served after nonzero update | Natural task deltas | Effective corrected rows | Mixed short-reward groups | Seconds |',
'|---|---|---:|---|---|---:|---:|---:|---:|']
for x in cases:
 c=bycase[x['case']]
 lines.append(f"| {x['case']} | {x['status']} | {x['logical_optimizer_steps']} | {c['nonzero_weight_change_versions']} | {c['served_after_nonzero_update_versions']} | {c['natural_nonzero_task_deltas']} | {c['natural_corrected_rows']} | {c['mixed_short_reward_groups']} | {x['seconds']:.1f} |")
lines += ['', 'Off bypasses the hook, so its hook-based correction counts are not measured. Short reward groups use hook inputs when present, otherwise full DAPO candidate batches or standard actor batches; the source is recorded per case. All-case token diversity is in `scale_coverage.json`. DAPO exhaustion is UNCOVERED, not a passed full-update case. A published version counts as served only when normal rollout metadata records it. Final publication alone does not prove serving.', '', '## GPU actor diagnostic','',f'Observed comparisons: `{actors}`.',
'Default GPU actor replay failed twice: repeated off-mode gradients differed beyond the fixed tolerance (maximum observed absolute difference0.000732421875), while inspected full loss inputs were exact. Both failed attempts remain saved. Explicit PyTorch deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 in the standalone diagnostic restored exact repeats. This is not a claim of default trainer bitwise reproducibility; production code/configuration was not changed by that diagnostic fix.',
'Actor selection is explicitly coverage-directed within the fixed natural capture: first two corrected UID groups, filling with earliest others; all four rows per group retained. Selection is recorded in `actor_h2048/selection.json`. Registered vanilla/GSPO losses and gradients repeat exactly. Shadow must match off. Replacement is allowed to change original GRPO advantages and gradients; an observed advantage change must change at least one gradient shard. No synthetic verifier outputs are used in this retest.', '',
'## Actual natural correction and actor credit','',
f"Selected real service trace: `{natural['source_record']}`. Original verifier replay, oracle/hook scores and filtering passed exactly. Actual actor prefix tokens, effective scores and recomputed original GRPO advantages are exact.",
f'Natural-event GPU counterfactual: `{natural_gpu}`.',
'This event was observed in the prespecified trainer run, not fabricated or added to the512-prompt capture cohort. The exact actually retained8-row version0 actor batch, including original old_log_probs, is evaluated with original weights; all input tensors were matched before replay. No new generation or altered verifier is used for the conditional diagnostic. Other unobserved transitions remain uncovered.', '',
'## Limits and artifacts','',
'Unobserved natural transitions remain uncovered. An absence of corrections does not establish that the hook was never called: cap-hit/continuation, nonzero verifier delta, filtering and gradient effects are distinct gates. These are fixed-sample implementation diagnostics, not an accuracy benchmark or training-effectiveness/capacity/performance claim.',
'Original190 CPU tests and both frozen characterizations passed again in `cpu_final/`; fresh CUDA/NCCL gates, raw token/verifier/actor/gradient data and source provenance are retained. Previous reference and fault-barrier validation applies to unchanged production code and is not represented as newly executed here. No formal checkpoint, benchmark evaluation, shared dependency change or production fix.', '',
'Final process/GPU state: `final_state.json`; source/model/runner hashes: `final_manifest.json`; execution timings and CUDA allocation metrics: `case_analysis.json`; sampled inference NVML peaks: `h2048_processes.json`. First failures remain preserved.']
(E/'REPORT.md').write_text('\n'.join(lines)+'\n')
(R/'RESULTS.md').write_text((E/'REPORT.md').read_text())
(E/'runner_sha256.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in R.rglob('*.py') if 'evidence' not in p.parts and not any('cache' in x for x in p.parts)},indent=2))
print(E/'REPORT.md')
