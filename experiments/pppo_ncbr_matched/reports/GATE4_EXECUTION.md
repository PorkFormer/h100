# Gate 4 execution evidence and measurement limits

Gate 4 completed on 2026-11-11, Ray job 21000000. The frozen source fingerprint and configuration hash match Gates 0–3. Formal fresh-Base job 22000000 subsequently started; formal successful steps were still zero at this snapshot. Formal resume_mode=disable, total_training_steps=300, save_freq=50. No profile checkpoint is used for formal initialization.

Measured profiling counts: 6144 original responses, 49152 continuations, 768 candidate groups, 258 zero-variance filtered groups, 254 valid excess groups, 256 retained groups, 2048 actor rows, 16 actual optimizer updates. Zero-prefix fallback remained disabled. Gradient norm was finite (0.2880800543).

Profiling wall time was 8224.57 seconds, or 36.55364 allocated GPU-hours for 16 GPUs, including initialization and validation. Recorded outer-step wall time was 7174.34 seconds; actor update 106.02 seconds, original generation 157.17 seconds, continuation generation 6736.01 seconds, continuation reward 34.19 seconds, checkpoint 76.40 seconds, weight sync 6.29 seconds. Timers can overlap and must not be blindly summed.

Measurement limitations: actor/perf/max_memory_* metrics describe PyTorch actor allocator memory, not the peak physical memory of colocated vLLM plus actor. An interactive nvidia-smi sample during continuation showed roughly 25.5–26.5 GiB on head GPUs; it is not a measured whole-job physical peak. Initial and final acceptance holdout validation times are included in total job time but were not separately instrumented; missing testing timers must not be reported as zero validation cost. Offline Step-300 evaluation is NOT RUN.

The legacy cost_report.json aggregator uses the latest resume attempt duration for the whole smoke-g8 run. Therefore its smoke-g8 GPU-hours are incomplete: use the per-attempt measurements in cost_attempts.json instead. Copied canonical Gate 2 receipts are not separate executions and must not be double-counted. These reporting limitations do not change the measured model updates or algorithm, and no GPU-hour efficiency advantage is claimed.
