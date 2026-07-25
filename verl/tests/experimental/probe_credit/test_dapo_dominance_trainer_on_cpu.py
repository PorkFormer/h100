import sys
import types
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

if "cachetools" not in sys.modules:
    cachetools = types.ModuleType("cachetools")

    class _LRUCache(dict):
        def __init__(self, maxsize):
            super().__init__()
            self.maxsize = maxsize

    cachetools.LRUCache = _LRUCache
    sys.modules["cachetools"] = cachetools

rollout_utils = types.ModuleType("verl.workers.rollout.utils")
rollout_utils.update_prometheus_config = lambda *_args, **_kwargs: None
sys.modules.setdefault("verl.workers.rollout.utils", rollout_utils)

checkpoint_package = types.ModuleType("verl.utils.checkpoint")
checkpoint_package.__path__ = []
checkpoint_manager = types.ModuleType("verl.utils.checkpoint.checkpoint_manager")
checkpoint_manager.find_latest_ckpt_path = lambda *_args, **_kwargs: None
checkpoint_manager.should_save_ckpt_esi = lambda *_args, **_kwargs: False
sys.modules.setdefault("verl.utils.checkpoint", checkpoint_package)
sys.modules.setdefault("verl.utils.checkpoint.checkpoint_manager", checkpoint_manager)

from verl import DataProto  # noqa: E402
from verl.experimental.probe_credit import (  # noqa: E402
    dapo_dominance_trainer as dapo_dominance_trainer_module,
)
from verl.experimental.probe_credit import dapo_trainer as dapo_trainer_module  # noqa: E402
from verl.experimental.probe_credit.dapo_dominance_trainer import (  # noqa: E402
    RayDAPOReadinessDominanceTrainer,
)
from verl.experimental.probe_credit.dynamic_sampling import (  # noqa: E402
    filter_dapo_generation_batch,
    select_complete_prompt_groups,
)
from verl.trainer.config import ProbeCreditConfig, ReadinessDominanceConfig  # noqa: E402
from verl.workers.rollout.replica import TokenOutput  # noqa: E402


def _config(
    *,
    mode="shadow",
    horizons=None,
    adv_estimator="grpo",
    rollout_name="vllm",
    loss_agg_mode="token-mean",
    filter_enable=True,
    filter_metric="acc",
):
    dominance = ReadinessDominanceConfig(
        mode=mode,
        absolute_horizons=[1, 2] if horizons is None else horizons,
        n=4,
        max_tokens=2,
        max_concurrent_requests=4,
        request_batch_size=8,
    )
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            adv_estimator=adv_estimator,
            probe_credit=ProbeCreditConfig(),
            readiness_dominance=dominance,
            use_kl_in_reward=False,
            rollout_correction=None,
            filter_groups=SimpleNamespace(enable=filter_enable, metric=filter_metric),
        ),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(loss_agg_mode=loss_agg_mode),
            rollout=SimpleNamespace(
                name=rollout_name,
                mode="async",
                n=4,
                response_length=4,
                prompt_length=2,
                max_model_len=16,
                multi_turn=SimpleNamespace(enable=False),
            ),
        ),
        distillation=SimpleNamespace(enabled=False),
        global_profiler=SimpleNamespace(steps=None),
    )
    config.algorithm.get = lambda name, default=None: getattr(config.algorithm, name, default)
    return config


def _version_batch(ids=("keep-a", "keep-b"), versions=(3, 3)):
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros(len(ids), 1)},
        non_tensors={
            "trajectory_id": np.asarray(ids, dtype=object),
            "rollout_policy_version": np.asarray(versions, dtype=object),
        },
    )


def _probe_batch(acc=(1.0, 0.0), versions=(3, 3)):
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.long)
    scores = torch.zeros(2, 4)
    scores[0, 2] = 1.0
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[10, 11], [10, 11]]),
            "responses": torch.tensor([[20, 21, 22, 0], [30, 31, 32, 33]]),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
            ),
            "response_mask": response_mask,
            "token_level_scores": scores,
            "token_level_rewards": scores.clone(),
        },
        non_tensors={
            "uid": np.asarray(["p", "p"], dtype=object),
            "trajectory_id": np.asarray(["success", "failure"], dtype=object),
            "rollout_policy_version": np.asarray(versions, dtype=object),
            "acc": np.asarray(acc, dtype=object),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray(
                [{"ground_truth": "1"}, {"ground_truth": "1"}], dtype=object
            ),
            "extra_info": np.asarray([{}, {}], dtype=object),
        },
    )


def _trainer(config=None):
    trainer = object.__new__(RayDAPOReadinessDominanceTrainer)
    trainer.config = _config() if config is None else config
    trainer.global_steps = 4
    trainer._rollout_policy_version = 3
    trainer.use_critic = False
    trainer.use_teacher_policy = False
    return trainer


def test_validation_accepts_typed_shadow_configuration():
    trainer = _trainer()

    trainer._validate_probe_credit_mode()

    assert trainer._dominance_config().mode == "shadow"


def test_validation_instantiates_hydra_dominance_node_as_typed_config():
    trainer = _trainer()
    trainer.config.algorithm.readiness_dominance = OmegaConf.create(
        {
            "_target_": "verl.trainer.config.ReadinessDominanceConfig",
            "mode": "off",
            "absolute_horizons": [1, 2],
        }
    )

    trainer._validate_probe_credit_mode()

    assert isinstance(trainer._dominance_config(), ReadinessDominanceConfig)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(adv_estimator="gae"), "GRPO"),
        (_config(rollout_name="sglang"), "vLLM"),
        (_config(loss_agg_mode="seq-mean-token-mean"), "token-mean"),
        (_config(filter_enable=False), "filter_groups.enable=true"),
        (_config(filter_metric="score"), "filter_groups.metric=acc"),
        (_config(horizons=[1, 4]), "response_length"),
    ],
)
def test_validation_rejects_unsupported_training_protocol(config, message):
    trainer = _trainer(config)

    with pytest.raises(ValueError, match=message):
        trainer._validate_probe_credit_mode()


def test_off_mode_overrides_parent_prepare_and_skips_probe_and_acc():
    trainer = _trainer(_config(mode="off"))
    events = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append("sleep")
    )
    trainer._probe_final_retained_batch = MethodType(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("absolute Probe must not run")
        ),
        trainer,
    )
    batch = _version_batch()

    result = trainer._prepare_final_retained_batch(batch, {}, {})

    assert events == ["sleep"]
    assert result is batch
    assert "acc" not in batch.non_tensor_batch


def test_policy_version_mismatch_fails_before_probe_or_sleep():
    trainer = _trainer()
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: (_ for _ in ()).throw(AssertionError("must fail first"))
    )
    trainer._probe_final_retained_batch = MethodType(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must fail first")),
        trainer,
    )

    with pytest.raises(ValueError, match="policy version"):
        trainer._prepare_final_retained_batch(
            _version_batch(versions=(3, 4)), {}, {}
        )


@pytest.mark.parametrize(
    ("acc", "message"),
    [
        (None, "missing authoritative acc"),
        ([1.0], "length"),
        ([1.0, float("nan")], "finite"),
        ([1.0, float("inf")], "finite"),
        ([1.0, 0.5], "binary"),
        ([1.0, "correct"], "numeric"),
    ],
)
def test_authoritative_acc_fails_closed_when_missing_or_invalid(acc, message):
    trainer = _trainer()
    batch = _probe_batch()
    if acc is None:
        batch.non_tensor_batch.pop("acc")
    else:
        batch.non_tensor_batch["acc"] = np.asarray(acc, dtype=object)

    with pytest.raises(ValueError, match=message):
        trainer._terminal_success_from_acc(batch, {})


def test_acc_score_sum_disagreement_is_metric_only_and_acc_remains_authoritative():
    trainer = _trainer()
    batch = _probe_batch(acc=(0.0, 1.0))
    metrics = {}

    terminal_success = trainer._terminal_success_from_acc(batch, metrics)

    assert terminal_success.tolist() == [False, True]
    assert metrics["dominance/terminal_success_score_disagreement_rate"] == 1.0


def test_shadow_prepare_probes_before_sleep_with_authoritative_success_mask():
    trainer = _trainer()
    events = []
    observed_success = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append("sleep")
    )

    def probe(self, batch, terminal_success, metrics, timing_raw):
        observed_success.extend(terminal_success.tolist())
        events.append("absolute_probe")
        return batch

    trainer._probe_final_retained_batch = MethodType(probe, trainer)
    batch = _probe_batch()

    result = trainer._prepare_final_retained_batch(batch, {}, {})

    assert events == ["absolute_probe", "sleep"]
    assert observed_success == [True, False]
    assert result is batch


def test_real_absolute_probe_requests_only_terminal_success_trajectories():
    trainer = _trainer()
    trainer.tokenizer = lambda *_args, **_kwargs: {"input_ids": [99]}
    calls = []

    class FakeClient:
        async def generate_grouped(
            self, request_id, *, prompt_ids, sampling_params, routing_key=None
        ):
            calls.append((request_id, tuple(prompt_ids), dict(sampling_params), routing_key))
            return [
                TokenOutput(
                    token_ids=[branch],
                    extra_fields={
                        "text": str(branch),
                        "branch_id": branch,
                        "global_steps": 3,
                    },
                )
                for branch in range(4)
            ]

    trainer.llm_server_manager = SimpleNamespace(get_client=lambda: FakeClient())
    trainer._score_probe_candidate = MethodType(
        lambda self, batch, request, text: True, trainer
    )
    batch = _probe_batch()
    metrics = {}
    terminal_success = trainer._terminal_success_from_acc(batch, metrics)

    result = trainer._probe_final_retained_batch(
        batch, terminal_success, metrics, {}
    )

    assert len(calls) == 2
    assert [call[1] for call in calls] == [(10, 11, 20, 99), (10, 11, 20, 21, 99)]
    assert result.batch["dominance_terminal_success"].tolist() == [True, False]
    assert result.batch["dominance_probe_valid_mask"].tolist() == [
        [True, True],
        [False, False],
    ]
    assert result.batch["dominance_absolute_horizons"].tolist() == [[1, 2], [1, 2]]
    assert metrics["dominance/probe_valid_cell_rate"] == 1.0
    assert metrics["dominance/request_count"] == 2.0
    assert metrics["dominance/branch_count"] == 8.0


def test_strict_missing_probe_branch_fails_before_sleep():
    trainer = _trainer()
    trainer.tokenizer = lambda *_args, **_kwargs: {"input_ids": [99]}
    events = []

    class MissingBranchClient:
        async def generate_grouped(
            self, request_id, *, prompt_ids, sampling_params, routing_key=None
        ):
            return [
                TokenOutput(
                    token_ids=[branch],
                    extra_fields={
                        "text": str(branch),
                        "branch_id": branch,
                        "global_steps": 3,
                    },
                )
                for branch in range(3)
            ]

    trainer.llm_server_manager = SimpleNamespace(
        get_client=lambda: MissingBranchClient()
    )
    trainer._score_probe_candidate = MethodType(
        lambda self, batch, request, text: True, trainer
    )
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append("sleep")
    )

    with pytest.raises(ValueError, match="missing Probe branches"):
        trainer._prepare_final_retained_batch(_probe_batch(), {}, {})
    assert events == []


def test_success_trajectory_with_no_active_horizon_needs_no_server_version():
    trainer = _trainer(_config(horizons=[1]))
    trainer.tokenizer = lambda *_args, **_kwargs: {"input_ids": [99]}
    trainer.llm_server_manager = SimpleNamespace(
        get_client=lambda: (_ for _ in ()).throw(
            AssertionError("an empty absolute plan must not call the server")
        )
    )
    batch = _probe_batch()
    batch.batch["response_mask"][0] = torch.tensor([1, 0, 0, 0])
    batch.batch["token_level_scores"][0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    batch.batch["token_level_rewards"][0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metrics = {}
    terminal_success = trainer._terminal_success_from_acc(batch, metrics)

    result = trainer._probe_final_retained_batch(batch, terminal_success, metrics, {})

    assert result.batch["dominance_probe_values"].tolist() == [[0.0], [0.0]]
    assert result.batch["dominance_probe_valid_mask"].tolist() == [[False], [False]]
    assert result.batch["dominance_absolute_horizons"].tolist() == [[1], [1]]
    assert metrics["dominance/request_count"] == 0.0
    assert metrics["dominance/branch_count"] == 0.0
    assert metrics["dominance/probe_valid_cell_rate"] == 0.0


def _shadow_advantage_batch():
    response_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.long
    )
    advantages = torch.tensor(
        [[1.0, 1.0, 0.0], [0.5, 0.5, 0.0], [-1.0, -1.0, 0.0]]
    )
    scores = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    return DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "advantages": advantages,
            "returns": advantages.clone(),
            "token_level_scores": scores,
            "token_level_rewards": scores.clone(),
            "dominance_probe_values": torch.tensor(
                [[0.75, 1.0], [0.25, 0.5], [0.0, 0.0]]
            ),
            "dominance_probe_valid_mask": torch.tensor(
                [[True, True], [True, True], [False, False]]
            ),
            "dominance_terminal_success": torch.tensor([True, True, False]),
        },
        non_tensors={
            "uid": np.asarray(["p", "p", "p"], dtype=object),
            "trajectory_id": np.asarray(["a", "b", "c"], dtype=object),
        },
    )


def test_shadow_computes_direct_dominance_but_preserves_grpo_bitwise():
    trainer = _trainer()
    batch = _shadow_advantage_batch()
    advantages_before = batch.batch["advantages"].clone()
    returns_before = batch.batch["returns"].clone()
    scores_before = batch.batch["token_level_scores"].clone()
    rewards_before = batch.batch["token_level_rewards"].clone()
    ids_before = batch.non_tensor_batch["trajectory_id"].copy()
    metrics = {}

    result = trainer._compute_probe_credit_advantage(batch, metrics)

    assert torch.equal(result.batch["advantages"], advantages_before)
    assert torch.equal(result.batch["returns"], returns_before)
    assert torch.equal(result.batch["token_level_scores"], scores_before)
    assert torch.equal(result.batch["token_level_rewards"], rewards_before)
    assert result.non_tensor_batch["trajectory_id"].tolist() == ids_before.tolist()
    assert result.batch["dominance_frontier_mask"].tolist() == [True, False, False]
    assert result.batch["dominance_dominated_mask"].tolist() == [False, True, False]
    assert "terminal_advantages" not in result.batch
    assert "dominance_weights" not in result.batch
    assert metrics["dominance/group_with_dominance_rate"] == 1.0


def test_mock_event_order_places_shadow_dominance_after_standard_grpo(monkeypatch):
    trainer = _trainer()
    trainer.config.algorithm.gamma = 1.0
    trainer.config.algorithm.lam = 1.0
    trainer.config.algorithm.norm_adv_by_std_in_grpo = True
    trainer.config.actor_rollout_ref.rollout.n = 4
    events = ["terminal_reward", "filter", "complete_group_selection"]
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append("sleep_replicas")
    )
    trainer._probe_final_retained_batch = MethodType(
        lambda self, batch, success, metrics, timing: events.append("absolute_probe")
        or batch,
        trainer,
    )
    trainer._compute_probe_credit_advantage = MethodType(
        lambda self, batch, metrics: events.append("dominance") or batch, trainer
    )
    trainer._update_actor = MethodType(
        lambda self, batch: events.append("actor_update")
        or SimpleNamespace(meta_info={"metrics": {}}),
        trainer,
    )
    monkeypatch.setattr(
        dapo_trainer_module,
        "compute_advantage",
        lambda batch, **kwargs: events.append("standard_grpo") or batch,
    )
    batch = _probe_batch()

    batch = trainer._prepare_final_retained_batch(batch, {}, {})
    events.append("old_log_prob")
    trainer._compute_advantage_and_actor_update(batch, {}, {})

    assert events == [
        "terminal_reward",
        "filter",
        "complete_group_selection",
        "absolute_probe",
        "sleep_replicas",
        "old_log_prob",
        "standard_grpo",
        "dominance",
        "actor_update",
    ]


def test_reweight_saves_terminal_grpo_and_applies_trajectory_weights():
    trainer = _trainer(_config(mode="reweight"))
    batch = _shadow_advantage_batch()
    standard_advantages = batch.batch["advantages"].clone()
    scores_before = batch.batch["token_level_scores"].clone()
    rewards_before = batch.batch["token_level_rewards"].clone()
    ids_before = batch.non_tensor_batch["trajectory_id"].copy()
    metrics = {}

    result = trainer._compute_probe_credit_advantage(batch, metrics)

    assert torch.equal(result.batch["terminal_advantages"], standard_advantages)
    torch.testing.assert_close(
        result.batch["dominance_weights"], torch.tensor([1.5, 0.0, 1.0])
    )
    torch.testing.assert_close(
        result.batch["advantages"],
        torch.tensor([[1.5, 1.5, 0.0], [0.0, 0.0, 0.0], [-1.0, -1.0, 0.0]]),
    )
    assert torch.equal(result.batch["returns"], result.batch["advantages"])
    assert torch.equal(result.batch["token_level_scores"], scores_before)
    assert torch.equal(result.batch["token_level_rewards"], rewards_before)
    assert result.non_tensor_batch["trajectory_id"].tolist() == ids_before.tolist()
    assert metrics["dominance/positive_mass_before"] == 3.0
    assert metrics["dominance/positive_mass_after"] == 3.0
    assert metrics["dominance/mass_residual_max"] == 0.0


def test_reweight_without_direct_dominance_is_bitwise_baseline():
    trainer = _trainer(_config(mode="reweight"))
    batch = _shadow_advantage_batch()
    batch.batch["dominance_probe_values"][:2] = torch.tensor(
        [[0.25, 1.0], [0.50, 0.75]]
    )
    standard_advantages = batch.batch["advantages"].clone()
    standard_returns = batch.batch["returns"].clone()

    result = trainer._compute_probe_credit_advantage(batch, {})

    assert torch.equal(result.batch["terminal_advantages"], standard_advantages)
    assert torch.equal(result.batch["advantages"], standard_advantages)
    assert torch.equal(result.batch["returns"], standard_returns)
    assert torch.equal(result.batch["dominance_weights"], torch.ones(3))
    assert not bool(result.batch["dominance_dominated_mask"].any())


def test_reweight_rejects_token_varying_standard_grpo_advantages():
    trainer = _trainer(_config(mode="reweight"))
    batch = _shadow_advantage_batch()
    batch.batch["advantages"][0, :2] = torch.tensor([1.0, 2.0])
    batch.batch["returns"] = batch.batch["advantages"].clone()

    with pytest.raises(ValueError, match="constant"):
        trainer._compute_probe_credit_advantage(batch, {})


def test_mock_event_order_places_frontier_reweight_before_actor(monkeypatch):
    trainer = _trainer(_config(mode="reweight"))
    trainer.config.algorithm.gamma = 1.0
    trainer.config.algorithm.lam = 1.0
    trainer.config.algorithm.norm_adv_by_std_in_grpo = True
    trainer.config.actor_rollout_ref.rollout.n = 4
    events = ["terminal_reward", "filter", "complete_group_selection"]
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: events.append("sleep_replicas")
    )
    trainer._probe_final_retained_batch = MethodType(
        lambda self, batch, success, metrics, timing: events.append("absolute_probe")
        or batch,
        trainer,
    )
    trainer._update_actor = MethodType(
        lambda self, batch: events.append("actor_update")
        or SimpleNamespace(meta_info={"metrics": {}}),
        trainer,
    )
    monkeypatch.setattr(
        dapo_trainer_module,
        "compute_advantage",
        lambda batch, **kwargs: events.append("standard_grpo") or batch,
    )
    original_dominance = dapo_dominance_trainer_module.compute_readiness_dominance
    original_reweight = dapo_dominance_trainer_module.apply_frontier_reweighting

    def record_dominance(*args, **kwargs):
        events.append("dominance")
        return original_dominance(*args, **kwargs)

    def record_reweight(*args, **kwargs):
        events.append("frontier_reweight")
        return original_reweight(*args, **kwargs)

    monkeypatch.setattr(
        dapo_dominance_trainer_module,
        "compute_readiness_dominance",
        record_dominance,
    )
    monkeypatch.setattr(
        dapo_dominance_trainer_module,
        "apply_frontier_reweighting",
        record_reweight,
    )
    batch = _shadow_advantage_batch()
    batch.non_tensor_batch["rollout_policy_version"] = np.asarray(
        [3, 3, 3], dtype=object
    )
    batch.non_tensor_batch["acc"] = np.asarray([1.0, 1.0, 0.0], dtype=object)

    batch = trainer._prepare_final_retained_batch(batch, {}, {})
    events.append("old_log_prob")
    trainer._compute_advantage_and_actor_update(batch, {}, {})

    assert events == [
        "terminal_reward",
        "filter",
        "complete_group_selection",
        "absolute_probe",
        "sleep_replicas",
        "old_log_prob",
        "standard_grpo",
        "dominance",
        "frontier_reweight",
        "actor_update",
    ]


def test_subclass_reuses_parent_fit_without_copying_training_loop():
    assert (
        RayDAPOReadinessDominanceTrainer.fit
        is dapo_trainer_module.RayDAPOProbeCreditTrainer.fit
    )


def test_only_final_retained_complete_groups_reach_absolute_probe():
    trainer = _trainer()
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None)
    observed_ids = []
    observed_success = []

    def probe(self, batch, terminal_success, metrics, timing_raw):
        observed_ids.extend(batch.non_tensor_batch["trajectory_id"].tolist())
        observed_success.extend(terminal_success.tolist())
        return batch

    trainer._probe_final_retained_batch = MethodType(probe, trainer)
    uids = ["keep-a"] * 4 + ["filtered"] * 4 + ["keep-b"] * 4
    acc = [0.0, 1.0, 0.0, 1.0] + [0.0] * 4 + [1.0, 0.0, 1.0, 0.0]
    scores = torch.zeros(12, 2)
    scores[:, -1] = torch.tensor(acc)
    candidate = DataProto.from_dict(
        tensors={
            "token_level_scores": scores,
            "token_level_rewards": scores.clone(),
        },
        non_tensors={
            "uid": np.asarray(uids, dtype=object),
            "trajectory_id": np.asarray([f"t-{index}" for index in range(12)], dtype=object),
            "rollout_policy_version": np.asarray([3] * 12, dtype=object),
            "acc": np.asarray(acc, dtype=object),
        },
    )

    filtered = filter_dapo_generation_batch(candidate, "acc")
    retained = select_complete_prompt_groups(filtered, prompt_count=2, rollout_n=4)
    result = trainer._prepare_final_retained_batch(retained, {}, {})

    assert result.non_tensor_batch["uid"].tolist() == ["keep-a"] * 4 + ["keep-b"] * 4
    assert observed_ids == [f"t-{index}" for index in range(4)] + [
        f"t-{index}" for index in range(8, 12)
    ]
    assert observed_success == [False, True, False, True, True, False, True, False]
    assert not any("filtered" == uid for uid in result.non_tensor_batch["uid"])


def test_incomplete_prompt_group_fails_before_absolute_probe():
    trainer = _trainer()
    trainer._probe_final_retained_batch = MethodType(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection must fail before Probe")
        ),
        trainer,
    )
    candidate = DataProto.from_dict(
        tensors={"dummy": torch.zeros(7, 1)},
        non_tensors={
            "uid": np.asarray(["a"] * 4 + ["b"] * 3, dtype=object),
            "trajectory_id": np.asarray([f"t-{index}" for index in range(7)], dtype=object),
            "acc": np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=object),
        },
    )

    with pytest.raises(ValueError, match="complete"):
        select_complete_prompt_groups(candidate, prompt_count=2, rollout_n=4)
