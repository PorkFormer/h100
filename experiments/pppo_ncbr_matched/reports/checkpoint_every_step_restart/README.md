# Fresh restart with every-step checkpoints

User authorized stopping the original formal job and restarting with revised checkpoint saving on 2026-11-12. Original job 22000000 completed three outer steps (48 optimizer updates) and was stopped during Step 4. Its logs remain under `outputs/pppo_ncbr_matched_16xa10040g_s300`; no formal checkpoint existed. Its expenditure is discarded-run cost, not part of the new model lineage.

New output: `outputs/pppo_ncbr_matched_16xa10040g_s300_ckpt1_v2`.
Frozen full configuration: `verl/examples/pppo/configs/pppo_ncbr_matched_16xa10040g_ckpt1.yaml`.

Only `trainer.save_freq=1` and four output/name fields differ from the original frozen recipe. All 300 steps, N=8, G=8, candidate batch 768, retained batch 256, 8K cap, 16 optimizer updates/step, Base initialization and validation cadence remain unchanged. Model/optimizer/extra-state shards, DataLoader cursor, RNG and PPPO sidecar are saved through the existing collective checkpoint transaction; a checkpoint is usable only after sidecar status becomes `committed`. HF model exports are included. All checkpoints are retained; no automatic deletion.

Gate 0 is a strict field-delta audit against the original validated config. CPU Gate 1 is rerun. GPU Gates 2–4 are explicitly scoped reuse, NOT newly executed GPU tests: GPU/algorithm source is byte-identical to the passing source, and save_freq=1 already ran in the multi-step smoke, fresh-process resume and full-scale profile. Original receipts are preserved as parent_result.json; scoped reuse receipts retain the original evidence paths and hashes. The only modified infrastructure module adds environment-selected run/config paths. No numerical thresholds change.

Launch (on the existing allocated head, with the original two-node Ray cluster):

```bash
bash verl/examples/pppo/run_pppo_ncbr_matched_ckpt1.sh pipeline
```

The pipeline refuses an existing training history and holds an exclusive lock. It runs formal fresh-Base training, then the same Step-300 evaluation; it is an execution coordinator, not a periodic monitoring script. To resume a committed checkpoint explicitly, use the existing launch.py with PPPO_MATCHED_RUN and PPPO_MATCHED_CONFIG set to the new absolute paths, plus `formal --resume <new-run>/checkpoints/global_step_N`. Never use a smoke or profiling checkpoint.

Budget selection is a one-shot read-only report, not a background monitor:

```bash
python verl/examples/pppo/budget_checkpoint_report.py /workspace/rl/h100/outputs/pppo_ncbr_matched_16xa10040g_s300_ckpt1_v2
```

The historical NCBR full-lineage target is 628.649734220604 GPU-hours (Step 1–20 plus resumed Step 21–300). Every-step saving preserves both the last complete model below and the first model above that budget. Extra checkpoint time is included in the new process-based cost, not silently subtracted. Compare exact timer scopes before claiming equal cost: checkpoint file mtime gives a boundary estimate, whereas the historical receipt includes process-end overhead. Separately retain step-timer totals. Validation, offline evaluation, old aborted-run and acceptance costs must be distinguished. Equal-budget evaluation is additional to the original Step-300 evaluation and uses the same 8K/32-sample benchmark contract; a matching checkpoint has not yet been selected or evaluated.
