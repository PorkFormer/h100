# Step20 pause and aligned offline evaluation

This is a separate evaluation extension. It does not change training code,
optimizer, schedule, sampling, or the previously accepted gates. Training remains
paused after completion. The active conversation checks training progress; there
is no training-monitor daemon in this directory.

## Frozen experiment

| Model | Actual checkpoint | Evaluation questions |
|---|---:|---|
| PPPO + DAPO Dynamic Sampling (NCBR-Matched) | 20 | AIME24 (30), AIME25 (30), AMC23 (40), MATH500 (500) |
| DAPO+NCBR H2048/L8192 | 300 | MATH500 (500) |
| Vanilla DAPO H2048 | 300 | MATH500 (500) |
| Vanilla DAPO H4096 | 300 | MATH500 (500) |
| Vanilla DAPO H8192 | 300 | MATH500 (500) |

Every question gets 32 independent samples, response cap 8192, temperature/top-p
1, top-k -1, normal EOS, historical request seeds and Answer: prompt wrapper.
The original 100 questions retain IDs 512–611; MATH500 uses 612–1111. The
historical BF16 vLLM 0.18 TP1 worker runs on eight head GPUs, eight logical shards,
ordered batches of eight. No extra GPU allocation. The four historical runs use
actual Step300 exports even when their run directory names contain `s600`.

There are 83,200 formal responses plus a separate 256-response diagnostic.
All generations are independent; no truncating previously generated responses
to simulate an 8K run. The original `math_dapo.compute_score` receives full
responses with default arguments. It implements normalized string equivalence,
not general symbolic algebra. Errors/timeouts fail the arm instead of scoring 0.

Pass@1 is mean per-question c/32 and equals Avg@32; Pass@32 is the fraction of
questions with c>0. SE and bootstrap intervals use questions as sampling units,
not training seeds. Reports include lengths and truncation, paired MATH500
differences, and evaluation costs separately from training costs.

## Execution

Run from the independent worktree with the existing training interpreter:

```bash
export PYTHONPATH=verl
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q verl/examples/pppo/offline_step20/test_evaluate.py
python verl/examples/pppo/offline_step20/evaluate.py prepare
python verl/examples/pppo/offline_step20/evaluate.py preflight
```

`prepare` is one-time and refuses overwriting inputs. It freezes source hashes,
token IDs, protocol, extension source SHA and exact normalized-question overlap
audit. This audit does not establish absence of semantic or pretraining leakage.
The protocol must be refrozen with preserved old evidence if `evaluate.py`
changes before inference. Never change it silently after generation.

After Step20 metrics and committed checkpoint exist, invoke the one-shot pause:

```bash
python verl/examples/pppo/offline_step20/pause.py
```

It checks 320 optimizer updates, 256 retained groups, 2048 actor trajectories,
finite metrics, holdout evaluation and all 50 recovery files. It verifies the
driver identity before terminating only that training driver. Independently
verify Ray workers released resources and both nodes are idle after teardown.
Do not stop the Ray cluster or kill unrelated processes. The receipt is not
proof of fresh-process recovery; that is covered by the earlier acceptance gate.

Re-run preflight to fully hash Step20, then:

```bash
python -u verl/examples/pppo/offline_step20/execute.py
```

This runs the diagnostic and five formal arms sequentially with exclusive
locking and idle-head checks. It sends authorized milestone/failure emails using
the existing mail transport and updated Step20 status text. It does not poll
training, schedule future checks, or resume training. Existing arm directories
cause refusal: preserve failed generations and validate/rescore where possible
before manually selecting a new attempt. No automatic destructive retry.

Artifacts are under
`outputs/pppo_ncbr_matched_16xa10040g_s300_ckpt1_v2/offline_eval_step20_h8192_math500_v1/`.
Large response files and weights stay outside Git. Commit small receipts/reports
after actual execution; never mark preparation or CPU tests as inference PASS.

NCBR's 628.6497 GPU-hours process-lineage estimate differs in scope from its
614.3770 step-plus-testing GPU-hours. Fixed PPPO Step20 is not automatically
compute matched. Preserve the Step20 cost and any trailing generation separately.
