# On-Policy Probe Credit Redistribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental synchronous DAPO trainer that preserves official Dynamic Sampling and optionally redistributes standard GRPO terminal advantage using on-policy Immediate Answer Prefix Probes.

**Architecture:** A dedicated experimental trainer subclasses `RayPPOTrainer` and migrates the official `verl-recipe` DAPO loop without changing the shared trainer. Focused modules own typed configuration, official filtering parity, Probe protocol/runtime, and pure credit math. vLLM gains an opt-in grouped-generation RPC that reuses the active rollout engine and is called only after the final retained batch is fixed.

**Tech Stack:** Python 3.10+, PyTorch, TensorDict/DataProto, Hydra/OmegaConf dataclasses, Ray, vLLM async server, pytest.

## Global Constraints

- Preserve every pre-existing unstaged/untracked user file; never reset, clean, or stage unrelated paths.
- Do not modify `RayPPOTrainer.fit()` or Dynamic Sampling selection semantics.
- Source Dynamic Sampling from `verl-project/verl-recipe:dapo/dapo_ray_trainer.py@e477deff97b15d067f8b4e71f75b80ae58ad64c4`.
- Probe only the final complete retained prompt groups and finish before actor update.
- All candidate rollouts and Probes use the same synchronous `global_step` policy version.
- Do not modify terminal reward, `token_level_scores`, `token_level_rewards`, clipping, KL, or loss aggregation.
- First version supports GRPO, synchronous training, and vLLM only.
- Run no GPU allocation, Slurm command, formal DAPO training, dependency installation, or destructive operation.

---

### Task 1: Typed Probe Configuration

**Files:**
- Modify: `verl/verl/trainer/config/algorithm.py`
- Modify: `verl/verl/trainer/config/ppo_trainer.yaml`
- Create: `verl/tests/trainer/config/test_probe_credit_config_on_cpu.py`

**Interfaces:**
- Produces: `ProbeCreditConfig.validate()`, `AlgoConfig.probe_credit`.

- [ ] Write failing tests for canonical defaults and every invalid rho/position/coef/n/max_tokens combination.
- [ ] Run `conda run -n pytorch pytest -q tests/trainer/config/test_probe_credit_config_on_cpu.py` and verify missing-type failures.
- [ ] Add a `ProbeCreditConfig(BaseConfig)` dataclass with the exact approved defaults and validation, export it, and add the registered YAML node.
- [ ] Run the focused tests and existing config tests.

### Task 2: Pure Probe Protocol and Runtime Mapping

**Files:**
- Create: `verl/verl/experimental/probe_credit/__init__.py`
- Create: `verl/verl/experimental/probe_credit/probe_runtime.py`
- Create: `verl/tests/experimental/probe_credit/test_probe_runtime_on_cpu.py`

**Interfaces:**
- Produces: `relative_horizons(response_length, positions)`, `first_nonempty_line(text)`, `immediate_verifier_text(candidate)`, `aggregate_probe_successes(values, n, strict)`, `derive_grouped_request_seed(...)`, `build_probe_requests(...)`, and `aggregate_probe_results(...)`.
- Request records include request ID, policy version, UID, trajectory index, position indices, absolute horizon, raw token IDs, grouped seed, and branch count.

- [ ] Write failing tests for floor boundaries, odd/short lengths, duplicate-prefix reuse, prompt-level V0 reuse, exact raw-token concatenation, stable seeds, strict context overflow, invalid branches, and shuffled results.
- [ ] Add parity tests that dynamically load the current offline Probe and compare candidate extraction, verifier wrapping, prefix construction, and success aggregation.
- [ ] Run the focused tests and observe feature-missing failures.
- [ ] Implement immutable request/result dataclasses and pure request construction/aggregation without pandas, CLI, file I/O, decode/re-encode, or chat templates.
- [ ] Run tests until all runtime mapping and parity cases pass.

### Task 3: Pure Probe Credit Mathematics

**Files:**
- Create: `verl/verl/experimental/probe_credit/probe_credit.py`
- Create: `verl/tests/experimental/probe_credit/test_probe_credit_on_cpu.py`

**Interfaces:**
- Produces: `compute_probe_pseudo_rewards`, `compute_probe_temporal_returns`, `compute_group_relative_probe_advantages`, `build_probe_token_correction`, and `apply_probe_credit_redistribution`.
- Calls: `verl.utils.as_torch_index` and `verl.utils.group_mean_std` for exact GRPO std parity.

- [ ] Write failing tests for the required exact example, rho 0/1, negative progress, group sizes 2/8, degeneracy, zero-mass correction, exact tail/padding zero, final advantage, disabled behavior, and coefficient zero.
- [ ] Run the focused tests and verify missing-module failures.
- [ ] Implement vectorized, no-grad PyTorch math, preserving `terminal_advantages` and setting GRPO `returns == advantages` only after correction.
- [ ] Add metrics for terminal, correction, final advantage, degeneracy, and zero-mean residual.
- [ ] Run the focused math tests until green.

### Task 4: Official Dynamic Sampling Parity Helper

**Files:**
- Create: `verl/verl/experimental/probe_credit/dynamic_sampling.py`
- Create: `verl/tests/experimental/probe_credit/test_dynamic_sampling_on_cpu.py`

**Interfaces:**
- Produces: `filter_dapo_generation_batch(batch, metric_name)` and `select_complete_prompt_groups(batch, prompt_count, rollout_n)`.

- [ ] Write failing DataProto tests reproducing upstream positive-std, degenerate, singleton, multi-generation accumulation, max-batch, and exact complete-group selection behavior.
- [ ] Include a literal reference implementation in tests matching the upstream block and compare retained IDs numerically.
- [ ] Implement the helper with upstream `np.std` filtering semantics and explicit complete-group validation/truncation.
- [ ] Verify Probe config is never read by this helper and run parity tests.

### Task 5: Grouped vLLM Probe Generation RPC

**Files:**
- Modify: `verl/verl/workers/rollout/llm_server.py`
- Modify: `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- Modify: `verl/verl/experimental/probe_credit/probe_runtime.py`
- Test: `verl/tests/experimental/probe_credit/test_probe_runtime_on_cpu.py`

**Interfaces:**
- Produces: `LLMServerClient.generate_grouped(...) -> list[TokenOutput]` and vLLM server `generate_grouped(...)`.
- Consumes: raw prefix IDs plus one grouped SamplingParams request containing `n`, seed, newline stop, and Probe-only decode settings.

- [ ] Write fake-server tests asserting one RPC/prefix, `n=4`, independent sampling dict copies, stable output-index branch mapping, and no normal rollout config mutation.
- [ ] Add a vLLM-only grouped RPC that collects every `RequestOutput.outputs` item without changing the existing `generate()` method.
- [ ] Add a synchronous-compatible Probe generator using `auto_await` and `asyncio.gather`, preserving explicit request IDs and arbitrary response order.
- [ ] Run fake runtime tests and Python compilation for both modified rollout files.

### Task 6: Experimental DAPO Trainer and Ordering

**Files:**
- Create: `verl/verl/experimental/probe_credit/dapo_trainer.py`
- Create: `verl/tests/experimental/probe_credit/test_dapo_trainer_on_cpu.py`

**Interfaces:**
- Produces: `RayDAPOProbeCreditTrainer`, `_probe_final_retained_batch`, and `_compute_probe_credit_advantage` hook points.
- Consumes: Tasks 1-5 modules plus standard `compute_advantage`.

- [ ] Write a CPU/mock event-log test for terminal reward -> filtering/accumulation -> final selection -> Probe -> standard GRPO -> redistribution -> actor update.
- [ ] Test zero Probe calls for filtered/cropped trajectories, identical retained IDs feature-off/on, same policy version, feature-off official parity, and independent configs.
- [ ] Migrate the official DAPO loop, documenting only current-API compatibility edits.
- [ ] Keep replicas awake through final Probe, then sleep before old-log-prob/model work; reject async/non-vLLM/non-GRPO configurations before training.
- [ ] Preserve `terminal_advantages`, apply correction after standard GRPO, and expose required metrics/timers.
- [ ] Run the trainer CPU/mock tests.

### Task 7: Entry Point, Config, Launcher, and Documentation

**Files:**
- Create: `verl/verl/experimental/probe_credit/main_dapo_probe_credit.py`
- Create: `verl/verl/trainer/config/probe_credit_dapo_trainer.yaml`
- Create: `verl/examples/probe_credit/train_dapo_qwen3_8b_h100x8_probe_credit_smoke.sh`
- Create: `verl/docs/algo/probe_credit.md`
- Create: `verl/tests/experimental/probe_credit/test_probe_credit_integration_on_cpu.py`

**Interfaces:**
- Entry point selects only `RayDAPOProbeCreditTrainer`.
- Launcher requires an explicit positive `PROBE_CREDIT_COEF` when enabled and does not contain `sbatch`.

- [ ] Write failing config composition and 2-prompts x 4-rollouts end-to-end CPU integration tests.
- [ ] Add TaskRunner/main, config inheritance, an executable shell launcher that only invokes Python, and equations/invariants/source documentation.
- [ ] Verify shell syntax, config composition, shape conservation, raw masked advantage-mass conservation, tail/padding zeros, and unchanged rewards.

### Task 8: Regression Verification and Delivery

**Files:**
- Review every task file; stage only these files and the two already committed spec/plan documents.

- [ ] Run `python -m py_compile` on core algorithms, shared trainer, experimental trainer/runtime, and modified rollout adapters.
- [ ] Run all new Probe Credit tests, existing groupwise/core algorithm/config tests, and current offline Probe tests in the suitable existing Conda environment.
- [ ] Run `bash -n` on the launcher, `git diff --check`, `git status --short`, and inspect the complete task diff.
- [ ] Confirm no terminal reward/Dynamic Sampling/shared trainer semantics changed and no active GPU/Slurm work was started.
- [ ] Commit only task files with an imperative subject and AI-assistance trailer.
- [ ] Push `agent/onpolicy-probe-credit` with upstream tracking and report exact commands, counts, warnings, source commit, compatibility edits, risks, and a non-executed H100 smoke recommendation.
