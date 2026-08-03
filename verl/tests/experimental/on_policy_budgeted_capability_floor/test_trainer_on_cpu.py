from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.capability_constraints.identity import canonical_prompt_key
from verl.experimental.capability_constraints.identity import reference_model_fingerprint
from verl.experimental.on_policy_budgeted_capability_floor.dapo_trainer import (
    RayDAPOOnPolicyBudgetedCapabilityFloorTrainer,
)
from verl.experimental.on_policy_budgeted_capability_floor.math import CapabilityAdvantageResult
from verl.experimental.on_policy_budgeted_capability_floor.prefix_batch import ProtectedGroupSelection
from verl.experimental.on_policy_budgeted_capability_floor.reward_adapter import (
    NormalizedRewardOutput,
)
from verl.trainer.config import OnPolicyBudgetedCapabilityFloorConfig


class _Algorithm(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _trainer(mode="shadow", *, update_interval=1):
    obcf = OnPolicyBudgetedCapabilityFloorConfig(
        mode=mode,
        cache_path="cache" if mode != "off" else None,
        reference_budget=2,
        update_interval=update_interval,
    )
    algorithm = _Algorithm(
        on_policy_budgeted_capability_floor=obcf,
        probe_credit=SimpleNamespace(enable=False),
        readiness_dominance=SimpleNamespace(mode="off"),
        success_support_floor=SimpleNamespace(mode="off"),
        adv_estimator="grpo",
        gamma=1.0,
        lam=1.0,
        norm_adv_by_std_in_grpo=True,
        use_kl_in_reward=False,
        rollout_correction=None,
        filter_groups=SimpleNamespace(enable=True, metric="acc"),
    )
    rollout = SimpleNamespace(
        name="vllm",
        n=2,
        response_length=4,
        multi_turn=SimpleNamespace(enable=False),
    )
    trainer = object.__new__(RayDAPOOnPolicyBudgetedCapabilityFloorTrainer)
    trainer.config = SimpleNamespace(
        algorithm=algorithm,
        actor_rollout_ref=SimpleNamespace(
            rollout=rollout,
            actor=SimpleNamespace(use_kl_loss=False),
        ),
        distillation=SimpleNamespace(enabled=False),
        trainer=SimpleNamespace(resume_mode="disable"),
    )
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.use_teacher_policy = False
    trainer.processor = None
    trainer.global_steps = 1
    trainer._lambda = 0.0
    trainer._violation_ema = 0.0
    return trainer


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda t: setattr(t.config.algorithm.probe_credit, "enable", True), "ProbeCredit"),
        (lambda t: setattr(t.config.algorithm.readiness_dominance, "mode", "shadow"), "Readiness"),
        (lambda t: setattr(t.config.algorithm.success_support_floor, "mode", "dual"), "BSSF"),
        (lambda t: setattr(t.config.algorithm, "adv_estimator", "gae"), "GRPO"),
        (lambda t: setattr(t.config.actor_rollout_ref.rollout, "name", "hf"), "vLLM"),
        (lambda t: setattr(t.config.actor_rollout_ref.rollout.multi_turn, "enable", True), "single-turn"),
        (lambda t: setattr(t, "processor", object()), "text-only"),
        (lambda t: setattr(t, "use_reference_policy", True), "reference"),
        (lambda t: setattr(t, "use_critic", True), "critic"),
        (lambda t: setattr(t, "use_teacher_policy", True), "teacher"),
        (lambda t: setattr(t.config.algorithm.filter_groups, "metric", "score"), "metric acc"),
    ],
)
def test_active_mode_rejects_unsupported_combinations(mutation, message):
    trainer = _trainer()
    trainer._load_obcf_cache = MethodType(lambda self: None, trainer)
    mutation(trainer)
    with pytest.raises(ValueError, match=message):
        trainer._validate_probe_credit_mode()


def _retained_batch():
    prompts = torch.tensor([[0, 10, 11], [0, 10, 11]])
    responses = torch.tensor([[20, 21, 22, 23], [30, 31, 32, 33]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1]] * 2)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": mask,
            "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4, 5]] * 2),
            "response_mask": torch.ones((2, 4), dtype=torch.long),
        },
        non_tensors={
            "uid": np.asarray(["occurrence", "occurrence"], dtype=object),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "x"}] * 2, dtype=object),
        },
    )


def test_shadow_observes_only_protected_exact_prefix_and_reports_metrics():
    trainer = _trainer(update_interval=2)
    key = canonical_prompt_key("tok", "tmpl", [10, 11])
    row = {"prompt_key": key, "prompt_token_ids": [10, 11], "capability_floor": 0.75}
    trainer._obcf_cache = SimpleNamespace(
        manifest={"tokenizer_fingerprint": "tok", "chat_template_fingerprint": "tmpl"},
        get=lambda candidate: row if candidate == key else None,
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    scored_batches = []

    def score(self, batch):
        scored_batches.append(batch)
        return NormalizedRewardOutput(torch.full((2, 2), 99.0), {"acc": [0, 1]})

    trainer._score_batch_with_existing_reward_pipeline = MethodType(score, trainer)
    metrics = {}
    result = trainer._observe_capability(_retained_batch(), metrics)

    assert result is not None
    assert scored_batches[0].batch["responses"].tolist() == [[20, 21], [30, 31]]
    assert metrics["obcf/protected_prompt_occurrences"] == 1.0
    assert metrics["obcf/prefix_verifier_calls"] == 2.0
    assert metrics["obcf/deficit_mean"] == pytest.approx(0.25)
    assert metrics["obcf/mixed_group_fraction"] == 1.0
    assert metrics["obcf/update_applied"] == 0.0


def test_no_protected_prompt_skips_prefix_scoring_and_preserves_dual_state():
    trainer = _trainer()
    trainer._lambda = 0.4
    trainer._violation_ema = 0.2
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._obcf_cache = SimpleNamespace(
        manifest={"tokenizer_fingerprint": "tok", "chat_template_fingerprint": "tmpl"},
        get=lambda _key: None,
    )
    trainer._score_batch_with_existing_reward_pipeline = lambda _batch: pytest.fail("must skip")
    metrics = {}

    assert trainer._observe_capability(_retained_batch(), metrics) is None
    assert trainer._lambda == 0.4
    assert trainer._violation_ema == 0.2
    assert metrics["obcf/prefix_verifier_calls"] == 0.0
    assert metrics["obcf/q_current_mean"] == 0.0
    assert metrics["obcf/all_zero_group_fraction"] == 0.0


def test_retained_multimodal_inputs_fail_closed_before_scoring():
    trainer = _trainer()
    batch = _retained_batch()
    batch.non_tensor_batch["multi_modal_inputs"] = np.asarray([{}, {}], dtype=object)
    with pytest.raises(ValueError, match="text-only"):
        trainer._observe_capability(batch, {})


@pytest.mark.parametrize("with_path", [False, True])
def test_fresh_run_requires_matching_base_weight_hash(monkeypatch, tmp_path, with_path):
    from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer

    trainer = _trainer("dual")
    trainer.global_steps = 99
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"actual")
    expected_hash = reference_model_fingerprint(model_dir)
    trainer._obcf_cache = SimpleNamespace(
        manifest={"reference_model_hash": expected_hash if not with_path else "a" * 64}
    )
    if with_path:
        trainer._obcf_base_model_local_path = str(model_dir)
    monkeypatch.setattr(
        RayDAPOProbeCreditTrainer,
        "_load_checkpoint",
        lambda self: setattr(self, "global_steps", 0),
    )

    message = "weight hash mismatch" if with_path else "local Base model path"
    with pytest.raises(ValueError, match=message):
        trainer._load_checkpoint()


def test_shadow_orders_terminal_advantage_observation_then_one_actor_update(monkeypatch):
    from verl.experimental.on_policy_budgeted_capability_floor import dapo_trainer as module

    trainer = _trainer()
    events = []
    batch = _retained_batch()

    def advantage(input_batch, **_kwargs):
        events.append("terminal_advantage")
        input_batch.batch["advantages"] = torch.ones((2, 4))
        return input_batch

    monkeypatch.setattr(module, "compute_advantage", advantage)
    trainer._compute_probe_credit_advantage = MethodType(
        lambda self, value, _metrics: events.append("probe_disabled") or value,
        trainer,
    )
    trainer._observe_capability = MethodType(
        lambda self, value, _metrics: events.append("prefix_observation")
        or (pytest.fail("advantages missing") if "advantages" not in value.batch else None),
        trainer,
    )
    actor_keys = []

    def update(self, value):
        events.append("actor_update")
        actor_keys.append(set(value.batch.keys()))
        return DataProto(meta_info={"metrics": {}})

    trainer._update_actor = MethodType(update, trainer)
    before_keys = set(batch.batch.keys())
    trainer._compute_advantage_and_actor_update(batch, {}, {})

    assert events == ["terminal_advantage", "probe_disabled", "prefix_observation", "actor_update"]
    assert actor_keys == [before_keys | {"advantages"}]


def _dual_observation():
    selection = ProtectedGroupSelection(
        rollout_indices=torch.tensor([0, 1]),
        group_ids=torch.tensor([0, 0]),
        prompt_keys=("protected",),
        capability_floors=torch.tensor([0.75]),
        rollout_count_per_group=2,
    )
    result = CapabilityAdvantageResult(
        q_current=torch.tensor([0.5]),
        deficit=torch.tensor([0.25]),
        active_group=torch.tensor([True]),
        centered_prefix_reward=torch.tensor([-0.5, 0.5]),
        token_advantage=torch.tensor([[-0.5, -0.5], [0.5, 0.5]]),
        observed_constraint=torch.tensor(0.25),
        mixed_group_fraction=torch.tensor(1.0),
        all_zero_group_fraction=torch.tensor(0.0),
        all_one_group_fraction=torch.tensor(0.0),
        nonzero_gradient_group_fraction=torch.tensor(1.0),
    )
    return selection, result


def test_dual_uses_preupdate_lambda_for_one_actor_step_then_updates_controller(monkeypatch):
    from verl.experimental.on_policy_budgeted_capability_floor import dapo_trainer as module

    trainer = _trainer("dual")
    trainer._ema_initialized = False
    trainer._constraint_observation_count = 0
    trainer._last_constraint_step = -1
    batch = _retained_batch()
    batch.batch["old_log_probs"] = torch.full((2, 4), -3.0)
    old_before = batch.batch["old_log_probs"].clone()
    events = []

    def advantage(value, **_kwargs):
        events.append("terminal_advantage")
        value.batch["advantages"] = torch.ones((2, 4))
        return value

    monkeypatch.setattr(module, "compute_advantage", advantage)
    trainer._compute_probe_credit_advantage = MethodType(lambda self, value, _metrics: value, trainer)
    trainer._observe_capability = MethodType(
        lambda self, _value, _metrics: events.append("prefix_observation") or _dual_observation(),
        trainer,
    )
    actor_advantages = []

    def update(self, value):
        events.append("actor_update")
        actor_advantages.append(value.batch["advantages"].clone())
        return DataProto(meta_info={"metrics": {}})

    trainer._update_actor = MethodType(update, trainer)
    metrics = {}
    trainer._compute_advantage_and_actor_update(batch, metrics, {})

    assert events == ["terminal_advantage", "prefix_observation", "actor_update"]
    assert torch.equal(actor_advantages[0], torch.ones((2, 4)))
    assert torch.equal(batch.batch["terminal_advantages"], torch.ones((2, 4)))
    assert batch.batch["capability_advantages"].tolist() == [
        [-0.5, -0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ]
    assert torch.equal(batch.batch["old_log_probs"], old_before)
    assert trainer._violation_ema == pytest.approx(0.25)
    assert trainer._lambda == pytest.approx(0.002)
    assert trainer._constraint_observation_count == 1
    assert metrics["obcf/update_applied"] == 1.0


def test_dual_without_protected_prompt_keeps_state_and_updates_actor_once(monkeypatch):
    from verl.experimental.on_policy_budgeted_capability_floor import dapo_trainer as module

    trainer = _trainer("dual")
    trainer._lambda = 0.4
    trainer._violation_ema = 0.2
    trainer._ema_initialized = True
    trainer._constraint_observation_count = 2
    trainer._last_constraint_step = 1
    def advantage(value, **_kwargs):
        value.batch["advantages"] = torch.ones((2, 4))
        return value

    monkeypatch.setattr(module, "compute_advantage", advantage)
    trainer._compute_probe_credit_advantage = MethodType(lambda self, value, _metrics: value, trainer)
    trainer._observe_capability = MethodType(lambda self, _value, _metrics: None, trainer)
    calls = []
    trainer._update_actor = MethodType(
        lambda self, value: calls.append(value) or DataProto(meta_info={"metrics": {}}),
        trainer,
    )

    trainer._compute_advantage_and_actor_update(_retained_batch(), {}, {})

    assert len(calls) == 1
    assert trainer._lambda == 0.4
    assert trainer._violation_ema == 0.2
    assert trainer._constraint_observation_count == 2
