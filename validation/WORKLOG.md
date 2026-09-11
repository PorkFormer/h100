# Active enlarged retest

User asked to rerun at larger scale to investigate missing mechanism coverage.
New clone `/tmp/ncbr_8gpu_scale_20260911`, branch `validation/scale`, based on98eb7d3.
Production remains37a812d, oracle d23da0e, baseline bfa0886. No production changes.
Previous outputs at `/tmp/ncbr_8gpu_20260911` are untouched. No subagents.

Frozen pool1024 unique prompts: old first128 byte-exact, extend with continuing RNG42
without replacement. Model/data hashes and1024 Parquet/JSON pairs verified. CUDA8,
NCCL36 and190 CPU/two frozen characterizations PASS. Exact plan in scale_plan.json.
Initial proposal of DAPO train/minibatch64 was revised BEFORE any trainer launch to
retain original2-group target and enlarge candidates128. Standard train/minibatch64.
Four optimizer steps per case max; nine cases. Same n4/H2048/L8192/seed/shaping/loss.

Capture active tool session18614 executes scale_capture.py, four sequential blocks
at offsets0,128,256,384, each128 prompts. Blocks0/128/256 finished in394/357/491s;
last block running. First two block replays passed exact frozen oracle/hook checks,
27+13 cap hits, no nonzero corrections. Third block replay active session23350.
Full capture aggregate expected at evidence/h2048_processes.json. Individual tokens
under infer_h2048_bNNNN_replicaR; final aggregate dirs omit block tag.

After capture exits successfully run:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/scale_followthrough.py
This sequentially runs full512-prompt frozen replay v2, GPU actor8 rows from first two
natural corrected groups (fill earliest if none), scale_actor_compare, nine trainers,
analyze_cases and scale_coverage. It stops on unexpected failure. Trainer main cases
have four steps and standard batch64; DAPO candidate128/train2/max4 candidate rounds.
Each launcher has1800s timeout and two idle snapshots/no compute processes + fresh CUDA.
No checkpoint/evaluation. All worker audits remain private. Case suffix remainsv6
inside this new clone, so no old artifact collision.

Check pending scripts/configs: scale config standard/DAPO dry runs passed.
scale_coverage records per-group token diversity, short rewards, task deltas, effective
corrections, changed-weight versions and actual normal serving versions. check_receipts
adapted to four steps without requiring the old zero-gradient first step. actor cross-
mode comparison allows natural replacement effects but shadow must equal off.

Important observation from old artifacts: standard GSPO replace first rollout had
4 identical branches for each of2 prompts; second rollout had2 unique branches each.
This supports a small-batch diversity limitation but not a causal claim about seed.
First128 new capture had5 verifier-correct rows/512, many invalid extracted answers.
Of first40 continuations35 stopped and5 hit L budget, so no claim that all need more L.

Remaining: finish capture and followthrough, inspect all errors, CPU/diff/syntax as
appropriate, check_receipts, scale_report.py, cleanup_owned.py inspect only after suites
exit, finalize.py, commit only validation scripts/docs/manifest with attribution.
scale_report.py writes evidence/REPORT.md and validation/RESULTS.md. finalize requires
runner_sha256.json from scale_report and no owned processes. Finalize again after commit
so manifest records correct validation revision. Old RESULTS/report.py are inherited
historical until superseded; do not use them as new evidence. Final response Chinese,
scoped comparison and gaps; memory citation MEMORY.md:4 used for isolation guidance.

Update: capture completed successfully,2048 short rows/78 cap hits,total1599.4s. Followthrough active tool session79190, log evidence/scale_followthrough.log, individual scale_stage_* logs. Full frozen replay then actor then nine trainers. No capture process remains.

Actor gate failed twice due backward numerical nondeterminism (off also affected), maxabs0.0007324, fixed tolerance failed. Both attempts and tensors preserved. Added deterministic algorithms and CUBLAS workspace ONLY to standalone actor diagnostic; eight-rank exact repeats now PASS;190 CPU/two golden rerun PASS. Followthrough resumed at stage2 via active tool session (see current tool result), log scale_followthrough_resume.log; it compares actor modes then starts nine trainers.

Active resumed followthrough session51128: stage2 actor comparison passed, no natural replacement effect (all78 cap transitions0-to-0); stage3 nine trainers now running, first DAPO off has started. Check per-case run.log and remaining_cases.json. Default actor repeatability failure must remain explicit in final report even though deterministic diagnostic passed.


Latest trainer status: DAPO off completed4 nonzero updates, all8 shards changed each
step, actual normal serving versions0/1/2/3.512 candidate groups:511 had4 distinct
trajectories,1 had3;43 mixed-acc groups. Scope coverage improves versus prior zero steps.
DAPO shadow active, has2 complete updates/16 optimizer receipts, third continuation
in progress. First27-request continuation cost579.2s,68062 tail tokens, max individual
request92.2s; all27 release acks total0.042s. Next26-request batch also complete. It may
hit the unchanged1800s case limit before4 updates. Do not relax timeout or concurrency4.

The currently executing old remaining_cases.py will STOP on timeout. New file now has
--resume --continue-timeouts. If active followthrough session51128 exits on a timeout:
inspect process.json and owned resources; run cleanup_owned.py --apply ONLY after the
case exits if leftovers exist. Then run remaining_cases.py --resume --continue-timeouts
to execute untouched remaining cases, skipping all recorded outcomes (including failed
ones), preserving their FAIL_TIMEOUT status. Subsequent timeout cases clean only owned
marked processes before fresh gates for next cases. Never rerun completed data/cases.
Followthrough may stop at stage3, so manually run final analyze_cases, scale_coverage,
check_receipts, scale_report, cleanup inspection, finalize and commit afterward.
Analyzer/check_receipts/coverage were adapted to RECORD timeout failure, not assert it
passed; incomplete rank counts must not be represented as full updates. Other unexpected
failures still stop and require diagnosis. No production source changes or external
process cleanup allowed. All commands/run logs in this isolated clone.

Update: shadow hit1800s limit and exited after1809s (driver code-6 due timeout teardown).
Owned cleanup inspection found no processes; all8 GPUs were0MiB/0%. Overall FAIL_TIMEOUT.
Two complete updates changed all8 ranks; normal service versions1/2 observed. Two complete
hook batches had49 cap transitions, all0-to-0. Earlier26-second-batch inference from
release count was inaccurate because it included in-progress next batch; complete total
is27+22=49.73 total remote release acks include partial third batch, not53 complete requests.

Active dispatcher now tool session22673 runs remaining_cases.py --resume --continue-timeouts,
log evidence/remaining_cases_resume.log. DAPO replace is running, ~5min elapsed at latest check.
It will continue subsequent timeout failures only after owned cleanup, then fresh GPU gates.
After dispatcher completes, manually run analyzers/report/finalization; old followthrough
is stopped. Off PASS4 steps; shadow FAIL_TIMEOUT2 steps; remaining7 cases still to finish.
Analysis after shadow timeout succeeded. Do not rerun or relabel original failed case.


CRITICAL NEW COVERAGE: DAPO replace FIRST step at policy version0 produced a real0-to-1
correction. Candidate row82, UID b8a2d358-a0cc-5f16-ba73-54b3845b434b, trajectory suffix:2;
task score-1 to+1, delta+2 at token2047 only, H length2048. Group AND trajectory entered
actual actor batch. See natural_correction_first.json. First replace hook has17 requests,
1 effective correction; first update completed in350.5s (may fit30min unlike shadow).

replay_trainer_natural.py PASSED on this frozen actual service batch. Independently
rerun original verifier matches actual saved short and long scalars; d23 runtime/adapter
and hook have exact requests, scores and filter ordering; actual actor responses,
token scores and original GRPO advantages match exactly. Artifacts at
replay_trainer_natural/, log replay_trainer_natural.log. No synthetic verifier outputs.

MANDATORY after all main GPU cases finish: run natural counterfactual GPU actor test:
python validation/launch_actor.py --h2048 (use separate args: --h 2048) --tag _natural
  --source validation/evidence/replay_trainer_natural/actor_input.pt
then OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python validation/scale_actor_compare.py --tag _natural.
CLI source/tag support was added and syntax checked; it preserves existing actor_h2048
and writes actor_h2048_natural + scale_actor_comparison_natural.json. Uses deterministic
settings already validated. First two corrected UID groups (fill earliest other group)
are selected from this real version0 batch, with all4 trajectories each. Must verify
replacement changes advantages and gradient shards, while shadow/off remain exact.
This is a selected natural-event diagnostic, not a change to population capture/data.
Update scale_report.py to include this real event + GPU result; existing direct512-prompt
capture still has78 neutral cap transitions and must remain separate. Do not claim all
natural transition types covered. Main dispatcher22673 remains active, DAPO replace.

Natural GPU source refined to EXACT actual retained8-row actor batch (not candidate fill):
select_actual_actor.py PASSED, all6 input tensor keys exact against first actual actor batch,
actual old_log_probs included. Use --source validation/evidence/replay_trainer_natural/actor_actual_input.pt
for --tag _natural. actor_replay now uses frozen actual old_log_probs when present (otherwise
old self-logprob behavior unchanged). Full512-row candidate replay remains preserved in
actor_input.pt. Main DAPO replace has reached version3 normal rollout; fourth step pending.

Continuation update: DAPO replace timed out at1809s with3 complete eight-rank updates and2 effective corrections; preserve FAIL_TIMEOUT. Standard vanilla off completed4 steps; standard vanilla shadow completed4 steps and exited0 at1141.8s. Dispatcher22673 remains active, now standard vanilla replace. cpu_final passed190 tests and both frozen characterizations. Report generator now explicitly states NOT ALL PASS and default-gradient failure separately. Natural GPU follow-up remains required after main dispatcher.

09:44 UTC update: standard vanilla off/shadow/replace and GSPO off/shadow all exited0 with4 actual actor updates. Dispatcher22673 now runs final GSPO replace. Natural actor GPU follow-up still pending. Partial scale_coverage confirms all completed standard cases had changed weights v1-v4 and served v1-v3; vanilla shadow51 neutral caps, replace55 neutral caps. DAPO corrections remain2.

Final execution: all nine cases finished. Seven PASS with4 steps; DAPO shadow/replace FAIL_TIMEOUT with2/3 steps. GSPO shadow and replace each observed one additional0->1; replace effective count1. Total completed hook0->1 events4; effective corrected rows3. Natural actor replay and comparison PASS for vanilla and GSPO: shadow=off exact,4 changed advantage rows,8 changed gradient shards. analyze_cases,scale_coverage,check_receipts passed; all actor rows trace to short rollouts. cpu_final190/two frozen PASS; syntax28 scripts and diff-check PASS. Cleanup/finalize: no owned processes, all8 GPUs0MiB/0%,5532 evidence files indexed before final commit. Production diff against37a812d empty. No production fix.
