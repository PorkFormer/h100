# Readiness Dominance DAPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experimental direct Readiness Dominance analysis and optional trajectory-level positive-advantage reweighting to the existing synchronous DAPO trainer without changing ProbeCredit.

**Architecture:** Add pure dominance/reweighting math, a separately namespaced absolute-horizon Probe planner, an independent typed config, and a thin `RayDAPOProbeCreditTrainer` subclass. Reuse the parent `fit()` while overriding validation, `_prepare_final_retained_batch`, absolute probing, and post-GRPO processing.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Hydra/OmegaConf, pytest, existing VERL `DataProto`, grouped vLLM Probe runtime.

## Global Constraints

- Work directly on local `main`; create no local or remote branch and no worktree.
- Do not reset, stash, clean, force-push, or overwrite unrelated changes.
- Preserve all existing relative Probe, ProbeCredit config, five-position math, Dynamic Sampling, trainer, entrypoint, launcher, and tests.
- Use fixed positive strictly increasing absolute horizons and active common support with `min_common_positions=2`.
- Use authoritative binary finite reward extra-info `acc`; score-sum disagreement is diagnostic and nonfatal.
- Compute direct readiness dominance only; never compute a transitive closure.
- Support actor `loss_agg_mode=token-mean` only.
- Complete and test `off` and `shadow` before implementing `reweight`.
- Keep the smoke launcher fixed to `shadow`; do not run formal or expensive GPU training.
- Every commit includes `Co-authored-by: OpenAI Codex <codex@openai.com>`.

---

### Task 1: Pure Direct Readiness Dominance Mathematics

**Files:**
- Create: `verl/experimental/probe_credit/readiness_dominance.py`
- Create: `tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py`

**Interfaces:**
- Produces:
  - frozen `DominanceResult(dominance_matrix, eligible_mask, frontier_mask, dominated_mask, group_has_dominance)`
  - `compute_readiness_dominance(probe_values, valid_mask, terminal_success, positive_trajectory_mask, group_ids, *, n, strict_branch_margin=1, min_common_positions=2) -> tuple[DominanceResult, dict[str, float]]`
- Consumes only PyTorch tensors and group IDs; it has no trainer or `DataProto` dependency.

- [ ] **Step 1: Write validation and direct-edge tests**

Add tests with concrete tensors for:

```python
def test_direct_dominance_uses_shared_active_horizons():
    values = torch.tensor([[0.50, 0.75, 1.00, 0.00], [0.25, 0.75, 0.75, 1.00]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    result, metrics = compute_readiness_dominance(
        values,
        valid,
        torch.tensor([True, True]),
        torch.tensor([True, True]),
        ["p", "p"],
        n=4,
    )
    assert result.dominance_matrix.tolist() == [[False, True], [False, False]]
    assert result.frontier_mask.tolist() == [True, False]
    assert result.dominated_mask.tolist() == [False, True]
    assert metrics["dominance/comparable_pair_count"] == 1.0
    assert metrics["dominance/pair_coverage_rate"] == 1.0
```

Add separate tests for crossing profiles, identical profiles, different UIDs, terminal mismatch, nonpositive trajectory masks, fewer than two common valid cells, a three-trajectory directly nondominated set, pair-specific support without transitive closure, deterministic repeated calls, and shape/dtype/NaN/range/parameter failures.

- [ ] **Step 2: Run the new math tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py
```

Expected: collection fails because `verl.experimental.probe_credit.readiness_dominance` does not exist.

- [ ] **Step 3: Implement strict tensor validation and direct pairwise comparison**

Implement a frozen dataclass and explicit nested unordered-pair loop. For each same-group eligible pair, compute:

```python
common = valid_mask[i] & valid_mask[j]
if int(common.sum().item()) >= min_common_positions:
    delta = probe_values[i, common] - probe_values[j, common]
    i_dominates = bool((delta >= 0).all() and (delta >= strict_branch_margin / n).any())
    j_dominates = bool((delta <= 0).all() and (-delta >= strict_branch_margin / n).any())
```

Write only those direct edges. Set `dominated_mask = dominance_matrix.any(dim=0)`, `frontier_mask = eligible_mask & ~dominated_mask`, and group dominance flags from direct edges. Count crossing only when a comparable pair has both positive and negative deltas. Validate floating `probe_values`, boolean masks, exact shapes, finite values, `[0,1]`, positive parameters, and group ID length.

Define metric denominators exactly as the design table. Report zero for an empty denominator and report comparable-pair common-position mean/min/max.

- [ ] **Step 4: Run the math tests and existing pure ProbeCredit tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py \
  tests/experimental/probe_credit/test_probe_credit_on_cpu.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the direct math**

```bash
git add \
  verl/experimental/probe_credit/readiness_dominance.py \
  tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py
git commit -m "feat: add readiness dominance mathematics" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 2: Separately Namespaced Absolute-Horizon Probe Runtime

**Files:**
- Modify: `verl/experimental/probe_credit/probe_runtime.py`
- Create: `tests/experimental/probe_credit/test_absolute_probe_runtime_on_cpu.py`
- Verify unchanged: `tests/experimental/probe_credit/test_probe_runtime_on_cpu.py`

**Interfaces:**
- Produces:
  - `ProbePositionKind` values `"relative"` and `"absolute"`
  - `AbsoluteProbePlan(requests, valid_mask, absolute_horizons)`
  - `derive_absolute_grouped_request_seed(policy_version, uid, trajectory_id, absolute_horizon, ordered_branch_ids)`
  - `build_absolute_probe_requests(trajectories, *, trajectory_mask, policy_version, absolute_horizons, answer_prefix_token_ids, n, max_tokens, max_model_len, strict=True) -> AbsoluteProbePlan`
- Extends `ProbeRequest` with `position_kind` while preserving all existing relative request values and seed behavior.

- [ ] **Step 1: Write absolute planner and relative-regression tests**

Create tests asserting:

```python
plan = build_absolute_probe_requests(
    [
        ProbeTrajectory("p", "long", (10,), (20, 21, 22, 23, 24)),
        ProbeTrajectory("p", "short", (10,), (30, 31, 32)),
    ],
    trajectory_mask=[True, True],
    policy_version=7,
    absolute_horizons=[1, 3, 4],
    answer_prefix_token_ids=(99,),
    n=4,
    max_tokens=2,
    max_model_len=32,
)
assert plan.valid_mask == ((True, True, True), (True, False, False))
assert all(request.position_kind == "absolute" for request in plan.requests)
assert all(request.relative_position is None for request in plan.requests)
assert {request.absolute_horizon for request in plan.requests} == {1, 3, 4}
```

Also assert both trajectories use exactly one response token at horizon one, inactive cells create no request, false `trajectory_mask` rows create no request, input concatenation is raw-token exact, request IDs/seeds are stable but distinct from relative namespaces, routing stays UID/policy based, aggregation restores `[B,K]`, mixed policy versions fail, missing strict branches fail, and context overflow fails closed.

Add a regression assertion that existing relative requests have `position_kind=="relative"`, unchanged `relative_position`, and the same known grouped seed value calculated by the existing function.

- [ ] **Step 2: Run absolute and relative runtime tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_absolute_probe_runtime_on_cpu.py \
  tests/experimental/probe_credit/test_probe_runtime_on_cpu.py
```

Expected: the absolute test fails to import the new interfaces while all existing relative tests still collect.

- [ ] **Step 3: Implement the absolute position representation and planner**

Keep `derive_grouped_request_seed`, `_request_id`, `_make_request`, `relative_horizons`, and `build_probe_requests` behavior unchanged. Extend `ProbeRequest` so existing relative construction sets:

```python
position_kind="relative"
relative_position=<existing float>
```

Absolute construction sets:

```python
position_kind="absolute"
relative_position=None
absolute_horizon=horizon
```

Use JSON payloads prefixed with `"absolute"` for absolute IDs and seeds. Validate positive strictly increasing integer horizons, positive `n`/`max_tokens`, trajectory-mask length and booleans, consistent prompt tokens within UID, and `h < len(response_token_ids)`. Build one request per active cell; do not fill or request inactive cells.

- [ ] **Step 4: Run runtime tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_absolute_probe_runtime_on_cpu.py \
  tests/experimental/probe_credit/test_probe_runtime_on_cpu.py
```

Expected: all absolute and existing relative runtime tests pass.

- [ ] **Step 5: Commit the runtime**

```bash
git add \
  verl/experimental/probe_credit/probe_runtime.py \
  tests/experimental/probe_credit/test_absolute_probe_runtime_on_cpu.py
git commit -m "feat: add absolute-horizon probe runtime" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 3: Independent Readiness Dominance Configuration

**Files:**
- Modify: `verl/trainer/config/algorithm.py`
- Create: `tests/trainer/config/test_readiness_dominance_config_on_cpu.py`
- Verify unchanged: `tests/trainer/config/test_probe_credit_config_on_cpu.py`

**Interfaces:**
- Produces `ReadinessDominanceConfig` and `AlgoConfig.readiness_dominance`.
- Keeps `ProbeCreditConfig` defaults and validation byte-for-byte unchanged.

- [ ] **Step 1: Write typed-config tests**

Assert the exact defaults:

```python
config = ReadinessDominanceConfig()
assert config.mode == "off"
assert config.absolute_horizons == [256, 512, 1024, 2048]
assert config.n == 4
assert config.max_tokens == 32
assert config.strict_branch_margin == 1
assert config.min_common_positions == 2
config.validate()
```

Add parametrized failures for invalid modes, empty/noninteger/nonpositive/nonincreasing horizons, nonpositive `n`/`max_tokens`/margin/common positions/request limits, margin greater than `n`, invalid temperature/top-p/top-k/prefix/stop, non-strict operation, and batch size below concurrency. Assert two `AlgoConfig` objects own distinct dominance and ProbeCredit config instances.

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/trainer/config/test_readiness_dominance_config_on_cpu.py \
  tests/trainer/config/test_probe_credit_config_on_cpu.py
```

Expected: import fails for `ReadinessDominanceConfig`.

- [ ] **Step 3: Add the independent dataclass and AlgoConfig field**

Add the exact fields from the design with independent factories:

```python
@dataclass
class ReadinessDominanceConfig(BaseConfig):
    mode: str = "off"
    absolute_horizons: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    n: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 32
    stop: list[str] = field(default_factory=lambda: ["\n"])
    answer_prefix: str = "\n\nAnswer:"
    strict: bool = True
    strict_branch_margin: int = 1
    min_common_positions: int = 2
    max_concurrent_requests: int = 128
    request_batch_size: int = 512

readiness_dominance: ReadinessDominanceConfig = field(default_factory=ReadinessDominanceConfig)
```

Export the class from `algorithm.__all__`. Validation must not read or mutate `probe_credit`.

- [ ] **Step 4: Run both config suites**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/trainer/config/test_readiness_dominance_config_on_cpu.py \
  tests/trainer/config/test_probe_credit_config_on_cpu.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the config**

```bash
git add \
  verl/trainer/config/algorithm.py \
  tests/trainer/config/test_readiness_dominance_config_on_cpu.py
git commit -m "feat: add readiness dominance configuration" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 4: Trainer Off and Shadow Modes

**Files:**
- Create: `verl/experimental/probe_credit/dapo_dominance_trainer.py`
- Create: `tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py`
- Verify unchanged: `verl/experimental/probe_credit/dapo_trainer.py`
- Verify unchanged: `tests/experimental/probe_credit/test_dapo_trainer_on_cpu.py`

**Interfaces:**
- Produces `RayDAPOReadinessDominanceTrainer(RayDAPOProbeCreditTrainer)`.
- Overrides `_dominance_config`, `_validate_probe_credit_mode`, `_prepare_final_retained_batch`, `_probe_final_retained_batch`, and `_compute_probe_credit_advantage`.
- Inherits `fit()` unchanged; the post-GRPO parent hook name is overridden to preserve dynamic dispatch without editing the parent.

- [ ] **Step 1: Write validation, correctness, off, and shadow tests**

Use `object.__new__` and mock methods following the existing trainer tests. Assert:

```python
result = trainer._prepare_final_retained_batch(batch, metrics, timing)
assert events == ["absolute_probe", "sleep"]  # shadow
```

and:

```python
result = off_trainer._prepare_final_retained_batch(batch, {}, {})
assert events == ["sleep"]
assert "dominance_probe_values" not in result.batch
```

Cover:

- parent `_prepare_final_retained_batch` is not called;
- policy mismatch fails before Probe and sleep;
- mode validation rejects non-GRPO, non-vLLM, non-token-mean, unsupported parent modes, disabled filtering, filter metric other than `acc`, and horizons `>= rollout.response_length`;
- `acc` is required, finite, binary, length-matched, and authoritative;
- score-sum disagreement records `dominance/terminal_success_score_disagreement_rate` but does not fail;
- only terminal-success retained trajectories generate planned requests;
- shadow attaches Probe values/valid mask/horizons/terminal-success and computes direct dominance after standard GRPO;
- shadow preserves standard `advantages` and `returns` bitwise;
- score/reward tensors, IDs, and ordering stay bitwise/equal;
- strict aggregation failures propagate.

Mock event order must be:

```text
terminal_reward
filter
complete_group_selection
absolute_probe
sleep_replicas
old_log_prob
standard_grpo
dominance
actor_update
```

- [ ] **Step 2: Run trainer tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py \
  tests/experimental/probe_credit/test_dapo_trainer_on_cpu.py
```

Expected: import fails for the new trainer.

- [ ] **Step 3: Implement typed config and startup validation**

Convert Hydra nodes with `omega_conf_to_dataclass`. Call `ReadinessDominanceConfig.validate()`, then apply the same verified synchronous GRPO/vLLM restrictions as the parent without calling the parent's config-reading validation. Require:

```python
config.algorithm.filter_groups.enable is True
config.algorithm.filter_groups.metric == "acc"
config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
max(config.absolute_horizons) < config.actor_rollout_ref.rollout.response_length
```

- [ ] **Step 4: Implement correctness extraction and `_prepare_final_retained_batch`**

Validate policy version first. In `off`, sleep and return. Otherwise validate `acc`, calculate diagnostic score-sum disagreement, call the absolute Probe hook, then sleep exactly once. Never read standard advantages here.

```python
def _prepare_final_retained_batch(self, batch, metrics, timing_raw):
    self._validate_rollout_policy_version(batch)
    if self._dominance_config().mode != "off":
        terminal_success = self._terminal_success_from_acc(batch, metrics)
        batch = self._probe_final_retained_batch(batch, terminal_success, metrics, timing_raw)
    self.checkpoint_manager.sleep_replicas()
    return batch
```

- [ ] **Step 5: Implement absolute retained probing**

Build all retained `ProbeTrajectory` objects but pass authoritative terminal success as `trajectory_mask`. Generate and aggregate with existing grouped runtime functions. In strict mode assert aggregate validity equals the plan. Attach:

```text
dominance_probe_values
dominance_probe_valid_mask
dominance_absolute_horizons
dominance_terminal_success
```

Record request/branch/input/output metrics under `dominance/`.

The Probe call and aggregation are:

```python
plan = build_absolute_probe_requests(
    trajectories,
    trajectory_mask=terminal_success.tolist(),
    policy_version=rollout_policy_version,
    absolute_horizons=config.absolute_horizons,
    answer_prefix_token_ids=prefix_ids,
    n=config.n,
    max_tokens=config.max_tokens,
    max_model_len=max_model_len,
    strict=config.strict,
)
results = generate_grouped_probe_results(
    self.llm_server_manager.get_client(),
    plan.requests,
    sampling_params=sampling_params,
    score_candidate=lambda request, text: self._score_probe_candidate(batch, request, text),
    max_concurrent_requests=config.max_concurrent_requests,
    request_batch_size=config.request_batch_size,
)
```

- [ ] **Step 6: Implement shadow post-GRPO analysis**

Clone standard advantages and returns. Define positive trajectories from positive masked standard advantage mass. Call `compute_readiness_dominance`; attach direct-set masks and metrics. In shadow, assert and retain exact equality of both standard tensors. Do not create `terminal_advantages` or `dominance_weights`.

```python
standard_advantages = batch.batch["advantages"].clone()
standard_returns = batch.batch["returns"].clone()
positive_mask = (
    (standard_advantages.clamp_min(0) * batch.batch["response_mask"]).sum(-1) > 0
)
dominance, dominance_metrics = compute_readiness_dominance(
    batch.batch["dominance_probe_values"],
    batch.batch["dominance_probe_valid_mask"],
    batch.batch["dominance_terminal_success"],
    positive_mask,
    batch.non_tensor_batch["uid"],
    n=config.n,
    strict_branch_margin=config.strict_branch_margin,
    min_common_positions=config.min_common_positions,
)
if config.mode == "shadow":
    assert torch.equal(batch.batch["advantages"], standard_advantages)
    assert torch.equal(batch.batch["returns"], standard_returns)
```

- [ ] **Step 7: Run new trainer tests and all old ProbeCredit tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py \
  tests/experimental/probe_credit
```

Expected: all selected tests pass, including old ProbeCredit.

- [ ] **Step 8: Commit off and shadow**

```bash
git add \
  verl/experimental/probe_credit/dapo_dominance_trainer.py \
  tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py
git commit -m "feat: integrate readiness dominance shadow mode" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 5: Trajectory-Level Frontier Reweighting

**Files:**
- Modify: `verl/experimental/probe_credit/readiness_dominance.py`
- Extend: `tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py`

**Interfaces:**
- Produces `apply_frontier_reweighting(advantages, response_mask, group_ids, dominance) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]`.
- Requires standard GRPO active-token advantages to be constant per trajectory and padding advantages to be zero.

- [ ] **Step 1: Write reweight RED tests**

Use two independent prompt groups with explicit lengths and advantages:

```python
advantages = torch.tensor(
    [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [-1.0, -1.0, 0.0]]
)
response_mask = torch.tensor(
    [[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool
)
```

Construct a `DominanceResult` where row zero directly dominates row one. Assert row-one weight zero, row-zero weight three, negative row weight one, padding unchanged, and masked positive mass exactly conserved.

Add tests for bitwise no-dominance behavior, multiple groups, singleton positives, crossing profiles, constant-row validation, zero padding validation, NaN/Inf advantages, zero/invalid frontier mass, finite scales, and p50/p90/p99 metrics.

- [ ] **Step 2: Run reweight tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py
```

Expected: import or attribute failure for `apply_frontier_reweighting`.

- [ ] **Step 3: Implement trajectory weights and strict mass checks**

Start with `weights = torch.ones(B)`. For each direct-dominance group, calculate eligible and directly nondominated masked positive masses. Fail for nonfinite inputs, nonconstant active rows, nonzero padding, nonpositive/nonfinite frontier mass, nonfinite scale, or nonfinite residual. Set dominated weights to zero and directly nondominated weights to scale. Compute:

```python
new_advantages = advantages * weights.unsqueeze(-1)
```

Do not touch groups without direct dominance. Report the required mass, scale mean/max/p50/p90/p99, and skipped-invalid count metrics.

- [ ] **Step 4: Run math/reweight tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py \
  tests/experimental/probe_credit/test_probe_credit_on_cpu.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit reweight math**

```bash
git add \
  verl/experimental/probe_credit/readiness_dominance.py \
  tests/experimental/probe_credit/test_readiness_dominance_on_cpu.py
git commit -m "feat: add frontier advantage reweighting" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 6: Trainer Reweight Mode

**Files:**
- Modify: `verl/experimental/probe_credit/dapo_dominance_trainer.py`
- Extend: `tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py`

**Interfaces:**
- Consumes `apply_frontier_reweighting`.
- Produces reweight batch tensors and exact event order with `frontier_reweight` between direct dominance and actor update.

- [ ] **Step 1: Write reweight integration RED tests**

Assert event order:

```text
terminal_reward
filter
complete_group_selection
absolute_probe
sleep_replicas
old_log_prob
standard_grpo
dominance
frontier_reweight
actor_update
```

Assert `terminal_advantages` equals standard GRPO, eligible dominated positives change, weights have expected values, `returns==advantages`, no-dominance output is bitwise baseline, rewards/scores/IDs/order are unchanged, filtered groups are absent from requests, incomplete groups fail, and reweight still rejects non-token-mean aggregation or token-varying GRPO rows.

- [ ] **Step 2: Run trainer tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py
```

Expected: reweight assertions fail because shadow processing does not apply weights.

- [ ] **Step 3: Wire reweight after direct dominance**

In mode `reweight`, save:

```python
batch.batch["terminal_advantages"] = standard_advantages.clone()
new_advantages, weights, reweight_metrics = apply_frontier_reweighting(
    standard_advantages,
    batch.batch["response_mask"],
    batch.non_tensor_batch["uid"],
    dominance,
)
batch.batch["advantages"] = new_advantages
batch.batch["returns"] = new_advantages
batch.batch["dominance_weights"] = weights
```

Attach Probe/direct-set diagnostic tensors and assert scores, rewards, IDs, and order are unchanged.

- [ ] **Step 4: Run trainer and old ProbeCredit suites**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py \
  tests/experimental/probe_credit
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit reweight integration**

```bash
git add \
  verl/experimental/probe_credit/dapo_dominance_trainer.py \
  tests/experimental/probe_credit/test_dapo_dominance_trainer_on_cpu.py
git commit -m "feat: enable readiness dominance reweighting" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 7: Independent Entrypoint, Hydra Config, and Shadow-Only Launcher

**Files:**
- Create: `verl/experimental/probe_credit/main_dapo_readiness_dominance.py`
- Create: `verl/trainer/config/readiness_dominance_dapo_trainer.yaml`
- Create: `examples/readiness_dominance/train_dapo_qwen3_8b_h100x8_dominance_smoke.sh`
- Create: `tests/experimental/probe_credit/test_readiness_dominance_integration_on_cpu.py`

**Interfaces:**
- Entrypoint instantiates `RayDAPOReadinessDominanceTrainer`.
- Hydra config defaults `algorithm.readiness_dominance.mode=off`.
- Launcher explicitly sets mode `shadow` and cannot select reweight through a default environment variable.

- [ ] **Step 1: Write composition and static launcher tests**

Assert config composition uses GRPO, DAPO `acc` filtering, vLLM, token-mean, ProbeCredit disabled, and dominance off. Read files as text and assert the entrypoint names only the new trainer. Assert launcher contains:

```text
algorithm.readiness_dominance.mode=shadow
actor_rollout_ref.rollout.n=2
algorithm.readiness_dominance.n=2
algorithm.readiness_dominance.absolute_horizons=[512,1024,2048]
algorithm.readiness_dominance.max_tokens=32
trainer.total_training_steps=1
```

Assert it has a distinct experiment name, keeps project name untouched, and contains no `reweight`, `sbatch`, or `srun`.

- [ ] **Step 2: Run integration test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_readiness_dominance_integration_on_cpu.py
```

Expected: files are missing.

- [ ] **Step 3: Add entrypoint, config, and launcher**

Create `ReadinessDominanceTaskRunner(TaskRunner)` whose `run()` calls, in order, `add_actor_rollout_worker`, `add_critic_worker`, `add_reward_model_resource_pool`, `add_teacher_model_resource_pool`, `add_ref_policy_worker`, `validate_config`, tokenizer/processor loading, dataset/sampler creation, resource-pool initialization, the `RayDAPOReadinessDominanceTrainer` constructor with the same named config/tokenizer/processor/worker/dataset/sampler arguments used by the existing dedicated entrypoint, `init_workers()`, and `fit()`. Register:

```python
@hydra.main(
    config_path="../../trainer/config",
    config_name="readiness_dominance_dapo_trainer",
    version_base=None,
)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    runner_class = ray.remote(num_cpus=1)(ReadinessDominanceTaskRunner)
    run_ppo(config, task_runner_class=runner_class)
```

The YAML body is:

```yaml
defaults:
  - ppo_trainer
  - _self_

algorithm:
  adv_estimator: grpo
  filter_groups:
    _target_: verl.trainer.config.FilterGroupsConfig
    enable: true
    metric: acc
    max_num_gen_batches: 0
  probe_credit:
    enable: false
    coef: 0.0
  readiness_dominance:
    mode: off

actor_rollout_ref:
  actor:
    loss_agg_mode: token-mean
  rollout:
    name: vllm
```

The launcher requires `MODEL_PATH`, `TRAIN_FILE`, and `VAL_FILE`, invokes Python directly, requests eight existing GPUs only through config, uses one step and small batches, sets the exact shadow overrides from Step 1, and never submits a job.

- [ ] **Step 4: Run integration/config/old entrypoint tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit/test_readiness_dominance_integration_on_cpu.py \
  tests/experimental/probe_credit/test_probe_credit_integration_on_cpu.py \
  tests/trainer/config/test_readiness_dominance_config_on_cpu.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit entrypoint delivery**

```bash
git add \
  verl/experimental/probe_credit/main_dapo_readiness_dominance.py \
  verl/trainer/config/readiness_dominance_dapo_trainer.yaml \
  examples/readiness_dominance/train_dapo_qwen3_8b_h100x8_dominance_smoke.sh \
  tests/experimental/probe_credit/test_readiness_dominance_integration_on_cpu.py
git commit -m "feat: add readiness dominance trainer entrypoint" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 8: Experimental Protocol Documentation

**Files:**
- Create: `docs/algo/readiness_dominance.md`

**Interfaces:**
- Documents direct readiness dominance and the directly nondominated set.
- Explicitly limits reweight use pending offline stability validation.

- [ ] **Step 1: Write the protocol README**

Document terminal-equivalent `acc` success, fixed absolute horizons, active common support, pair-specific support, no transitivity assumption, crossing/identical incomparability, direct-edge set construction, token-mean trajectory weights, immutable verifier rewards, modes, metrics and denominators, and experimental validation boundaries.

Include the formula:

```text
i directly readiness-dominates j iff:
same prompt
AND both acc-success
AND both positive standard-GRPO trajectories
AND at least min_common_positions shared active absolute horizons
AND V_i(h) >= V_j(h) on every shared active horizon
AND one shared horizon improves by strict_branch_margin / n
```

State explicitly that the method does not optimize AUC, random budget distributions, or token-level potentials, and does not claim novelty for the forced-answer Probe. Require shadow before reweight and split-half/repeated-Probe/`n` sensitivity validation before formal reweight training.

- [ ] **Step 2: Check documentation text and diff**

Run:

```bash
rg -n "direct readiness|directly nondominated|active common|split-half|shadow|reweight" \
  docs/algo/readiness_dominance.md
git diff --check
```

Expected: all required concepts are found and diff check is clean.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/algo/readiness_dominance.md
git commit -m "docs: document readiness dominance protocol" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>"
```

---

### Task 9: Full Verification, Main Synchronization, and Push

**Files:**
- Verify all changed files; make no opportunistic refactors.

**Interfaces:**
- Produces a tested clean local `main` and pushes only `origin/main`.

- [ ] **Step 1: Run the complete requested CPU regression**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/experimental/probe_credit \
  tests/trainer/config/test_probe_credit_config_on_cpu.py \
  tests/trainer/config/test_readiness_dominance_config_on_cpu.py
```

Expected: zero failures.

- [ ] **Step 2: Compile and lint**

Run:

```bash
.venv/bin/python -m compileall verl/experimental/probe_credit
.venv/bin/ruff check \
  verl/experimental/probe_credit \
  tests/experimental/probe_credit
```

If `.venv/bin/ruff` is absent, use the existing `ruff` executable. Do not add a formatter dependency.

- [ ] **Step 3: Verify requirements and repository state**

Run:

```bash
git branch --show-current
git branch --list
git status --short
git log --oneline --decorate -15
git diff origin/main...HEAD --stat
git ls-remote --heads origin
```

Expected: current branch is `main`, local and remote contain no newly created branch, worktree is clean, and only intended commits are ahead.

- [ ] **Step 4: Fetch and reconcile origin**

Run:

```bash
git fetch origin
git rev-list --left-right --count origin/main...main
```

If origin is ahead, run `git rebase origin/main`, resolve conflicts without force, and repeat Steps 1–3. If origin is not ahead, continue.

- [ ] **Step 5: Push only main**

Run:

```bash
git push origin main
```

Then verify:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
git ls-remote --heads origin
```

Expected: clean `main`, equal local/remote HEADs, and no remote branch other than the pre-existing `main`.
