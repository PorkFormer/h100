# NCBR fixed-workload scheduler benchmark

This directory records the 2026-09-12 eight-A100 / Qwen3-1.7B-Base experiment. At identical request inputs and per-request decode lengths, work-conserving scheduling passed four paired repetitions at C64 (1.9136×, 95% CI 1.9011–1.9261) and C128 (1.5183×, 1.5096–1.5270). Capacity-aware admission remains **NO-GO**: C128 was still best and stable, and a steady-state saturation knee was not established. The production default remains `fixed_wave`.

Read [RESULTS.md](RESULTS.md) for the complete findings and [PROTOCOL.md](PROTOCOL.md) for the frozen protocol. Historical natural-generation timing in `../scheduler_validation` remains `DIAGNOSTIC_ONLY`; the new fixed-workload qualification does not change that historical result.

## What is versioned

- The exact benchmark, collector, profiling, reporting, and targeted-test Python sources from the completed experiment. Production scheduler/runtime files were not changed in this experiment.
- The report, protocol, preflight/model hashes, selection, paired statistics, and full original [evidence hash index](final_manifest.json).
- [results/instances.json](results/instances.json): all 27 measured instances (five sweep, sixteen paired, six natural), including metrics, topology and cleanup receipts, without raw generated tokens.
- [results/frozen_inputs.tar.gz](results/frozen_inputs.tar.gz): the exact frozen workload and predeclared prompt pool. [results/artifact_binding.json](results/artifact_binding.json) records their hashes.
- The source collection configuration, test log, initial source-audit discrepancy, and cleanup receipt. Rebuildable caches and exited private Ray directories were cleaned only after their ownership and inactivity were checked; runtime logs were archived locally.

Raw generation/capture events, GPU time series, and first-failure evidence remain unchanged in `/tmp/ncbr_scheduler_20260911/scheduler_validation_next/evidence/`. They are deliberately not Git blobs. The 6,274 originally sealed files, their paths, and the original report are preserved. `results/` and this README are subsequent publication artifacts, not retroactively added to that historical seal. Full raw evidence has not been uploaded to an external artifact store.

## CPU-only checks

From the repository root, using the existing experiment Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s scheduler_validation_next -p 'test_*.py'
git diff --check
```

The 13 targeted tests cover frozen-workload validation, replay sampling overrides, and paired statistics. The validated production scheduler/runtime hashes are in `preflight.json`; unchanged production code does not require repeating the historical CPU/actor/GPU correctness suite for this publication.

## Inputs and historical reproduction

To materialize the exact inputs in a fresh checkout:

```bash
tar -xzf scheduler_validation_next/results/frozen_inputs.tar.gz -C scheduler_validation_next
```

Check the extracted files against `results/artifact_binding.json` and `workload_manifest.json` before use. Do not rerun `prepare_prompts.py` or recollect trajectories to reproduce this report.

These are **environment-bound experiment scripts**, not a portable production launcher. They require the original model `/workspace/models/Qwen3-1.7B-Base`, the pinned dependencies/UUID order in `preflight.json`, the repository compatibility shim, and historical paths under `/tmp/ncbr_8gpu_scale_20260911/validation/evidence`. `results/collection_config.yaml` is a byte-exact copy of the external `train_dapo_vanilla_replace_v6/config.yaml` used by `collect.py` and `perf_case.py`; the checked-in prompt archive contains the complete predeclared pool. Data selection additionally depended on the original DAPO Parquet path.

`analyze.py` and `report.py` require the full locally retained raw evidence for recomputation. `final_checks.py` and `seal.py` document the original pre-publication sealing procedure; they intentionally bind the measured pre-publication HEAD `d7a0f81` and existing evidence. They are not post-commit CI commands and must not be used to overwrite the original seal. A Git-only checkout can inspect the report, compact receipts and hashes, and run the targeted tests, but cannot independently recompute all GPU metrics without the raw evidence.

GPU launchers require fresh exclusive instance directories, eight idle healthy devices, and explicit warmup. They never run automatically as part of the CPU checks. No training, capacity-aware implementation, default promotion, or additional benchmark execution was performed for this commit.
