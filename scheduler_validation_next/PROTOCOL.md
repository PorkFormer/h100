# Frozen systems benchmark protocol

Authorized checkout: /tmp/ncbr_scheduler_20260911, HEAD d7a0f810c678f2b20c55085dca182a109fdd500a.
Validated scheduler/runtime hashes are checked against implementation/audit manifests. Only this directory is added. No production runtime edits, training, optimizer, verifier or default changes.

Collection: frozen Qwen3-1.7B-Base, production eight TP1 vLLMHttpServer services/config. Predeclared 3072 unique prompt pool (existing 1024 followed by first eligible unseen prompts from unique data block), exact saved tokenizer IDs, n=4, H2048, L8192, temperature=1, top_p=1, top_k=-1, seed42, natural EOS. Complete 128-prompt / 512-trajectory batches are saved. Hit-cap trajectories are passed to unchanged build/run_boundary_continuations at policy_version0; first256 in batch/branch order frozen after complete natural continuation batch. No actor/optimizer. Full batch surplus retained, not discarded from evidence.

Benchmark: exact frozen input/prefix/request/routing/seed IDs, replay length=observed natural tail length, client-boundary max_tokens override and ignore_eos=true. No runtime monkeypatch. Same request_batch_size=256 in both arms, necessary because historical batch_size8 limits fixed_wave to8 even at larger C. Config remains identical otherwise, including previously enabled prefix caching.

Sweep first128 requests, WC C8/16/32/64/128, one formal instance each after fresh restart and identical per-service warmup. Inspect data for anomalies; repeat anomalous instances with distinct IDs, never overwrite. Select lowest stable C at >=95% observed maximum decode throughput. No default assumption C128 is saturated. Inspect sweep before paired runs.

Formal: 256 requests, four AB/BA alternating pairs at C_sat and stable high C (prefer128); if same C, reuse same four pairs explicitly rather than counting duplicate experiments, and add the adjacent lower successful sweep C as a supplemental second comparison. This refinement was recorded after sweep and before any paired measurements (observed C_sat=high=128; supplemental C64). Every instance fresh service/cache/runtime, eight UUID-bound CUDA-gated GPUs, same warmup. Startup outside continuation timing. Natural external validity follows formal: first128 input prefixes, original6144 max and EOS, one instance/arm/C.

Acceptance: exact effective inputs/request count/per-request decode lengths/totals/config/topology/concurrency, successful clean release/drain and owned-process teardown, mean fixed/WC >=1.05 and paired10000-bootstrap lower95%CI>1. Generated token content is not an eligibility condition.

Telemetry: nvidia-smi every~2s perGPU memory/utilization; dispatch/acquire/generation_complete/release_start/release_ack events; production engine timing fields when present. No private backend API. Active means request not yet generation-complete; slots include waiting release. Areas are request-seconds or slot-seconds, explicitly distinct from wall durations.

Stop after RESULTS.md and final artifact/source/GPU checks. No capacity-aware implementation or default promotion.

Natural supplement, declared before natural measurements: retain N128/C128 as a one-wave control; add one fresh fixed/WC N256/C128 pair to exercise pending-work refill. Normalized external-validity only, no extra statistical repetitions.
