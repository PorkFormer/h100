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
from verl.experimental.probe_credit import dapo_trainer as dapo_trainer_module  # noqa: E402
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer  # noqa: E402
from verl.trainer.config import ProbeCreditConfig  # noqa: E402


def _config(*, enable=True, coef=0.5, adv_estimator="grpo", rollout_name="vllm"):
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            adv_estimator=adv_estimator,
            probe_credit=ProbeCreditConfig(enable=enable, coef=coef),
            use_kl_in_reward=False,
            rollout_correction=None,
        ),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(name=rollout_name, mode="async", multi_turn=SimpleNamespace(enable=False))
        ),
        distillation=SimpleNamespace(enabled=False),
        global_profiler=SimpleNamespace(steps=None),
    )
    config.algorithm.get = lambda name, default=None: getattr(config.algorithm, name, default)
    return config


def _batch(ids=("keep-a", "keep-b"), versions=(3, 3)):
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros(len(ids), 1)},
        non_tensors={
            "trajectory_id": np.asarray(ids, dtype=object),
            "rollout_policy_version": np.asarray(versions, dtype=object),
        },
    )


def test_validation_rejects_non_grpo_non_vllm_and_silent_zero_coefficient():
    for config, message in [
        (_config(adv_estimator="gae"), "GRPO"),
        (_config(rollout_name="sglang"), "vLLM"),
        (_config(enable=True, coef=0.0), "positive"),
    ]:
        trainer = object.__new__(RayDAPOProbeCreditTrainer)
        trainer.config = config
        with pytest.raises(ValueError, match=message):
            trainer._validate_probe_credit_mode()


def test_validation_instantiates_hydra_probe_node_as_typed_config():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
                "probe_credit": {
                    "_target_": "verl.trainer.config.ProbeCreditConfig",
                    "enable": False,
                    "coef": 0.0,
                },
            },
            "actor_rollout_ref": {"rollout": {"name": "vllm"}},
            "distillation": {"enabled": False},
            "global_profiler": {"steps": None},
        }
    )

    trainer._validate_probe_credit_mode()

    assert isinstance(trainer._probe_config(), ProbeCreditConfig)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda trainer: setattr(trainer.config.algorithm, "use_kl_in_reward", True), "use_kl_in_reward=false"),
        (lambda trainer: setattr(trainer, "use_critic", True), "no critic"),
        (
            lambda trainer: setattr(trainer.config.actor_rollout_ref.rollout.multi_turn, "enable", True),
            "single-turn",
        ),
        (lambda trainer: setattr(trainer.config.distillation, "enabled", True), "distillation"),
        (lambda trainer: setattr(trainer, "use_teacher_policy", True), "teacher policy"),
        (
            lambda trainer: setattr(
                trainer.config.algorithm,
                "rollout_correction",
                SimpleNamespace(rollout_is="sequence", rollout_rs=None, bypass_mode=False),
            ),
            "rollout correction",
        ),
        (lambda trainer: setattr(trainer.config.global_profiler, "steps", [1]), "profiling"),
    ],
)
def test_validation_rejects_unverified_training_modes(mutate, message):
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=False, coef=0.0)
    trainer.use_critic = False
    trainer.use_teacher_policy = False
    mutate(trainer)

    with pytest.raises(ValueError, match=message):
        trainer._validate_probe_credit_mode()


def test_validation_allows_async_vllm_engine_with_synchronous_optimizer_updates():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=False, coef=0.0)
    trainer.use_critic = False
    trainer.use_teacher_policy = False

    trainer._validate_probe_credit_mode()


def test_timing_accumulator_adds_generation_batches_instead_of_overwriting():
    timing = {"gen": 2.0, "agent_loop/generate_sequences/mean": 1.5}

    dapo_trainer_module._accumulate_timing(
        timing,
        {"gen": 3.0, "agent_loop/generate_sequences/mean": 2.5},
    )

    assert timing == {"gen": 5.0, "agent_loop/generate_sequences/mean": 4.0}


def test_final_retained_batch_probes_before_sleep_and_preserves_ids():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=True)
    trainer.global_steps = 4
    trainer._rollout_policy_version = 3
    events = []
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: events.append("sleep"))

    def probe(self, batch, metrics, timing_raw):
        events.append(("probe", batch.non_tensor_batch["trajectory_id"].tolist()))
        return batch

    trainer._probe_final_retained_batch = MethodType(probe, trainer)
    batch = _batch()

    result = trainer._prepare_final_retained_batch(batch, {}, {})

    assert events == [("probe", ["keep-a", "keep-b"]), "sleep"]
    assert result.non_tensor_batch["trajectory_id"].tolist() == ["keep-a", "keep-b"]


def test_feature_disabled_skips_probe_and_matches_baseline_ids():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=False, coef=0.0)
    trainer.global_steps = 4
    trainer._rollout_policy_version = 3
    events = []
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: events.append("sleep"))
    trainer._probe_final_retained_batch = MethodType(
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Probe must not run")), trainer
    )

    result = trainer._prepare_final_retained_batch(_batch(), {}, {})

    assert events == ["sleep"]
    assert result.non_tensor_batch["trajectory_id"].tolist() == ["keep-a", "keep-b"]


def test_policy_version_mismatch_fails_before_probe_or_sleep():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=True)
    trainer.global_steps = 11
    trainer._rollout_policy_version = 10
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: (_ for _ in ()).throw(AssertionError("must fail first"))
    )

    with pytest.raises(ValueError, match="policy version"):
        trainer._prepare_final_retained_batch(_batch(versions=(10, 11)), {}, {})


def test_fresh_start_accepts_server_version_zero_at_training_step_one():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=False, coef=0.0)
    trainer.global_steps = 1
    trainer._rollout_policy_version = 0
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None)

    result = trainer._prepare_final_retained_batch(_batch(versions=(0, 0)), {}, {})

    assert result.non_tensor_batch["rollout_policy_version"].tolist() == [0, 0]


def test_resume_accepts_checkpoint_version_at_next_training_step():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=False, coef=0.0)
    trainer.global_steps = 101
    trainer._rollout_policy_version = 100
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None)

    trainer._prepare_final_retained_batch(_batch(versions=(100, 100)), {}, {})


def test_multiple_generation_batches_require_one_actual_policy_version():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer._rollout_policy_version = 10

    first = trainer._validate_rollout_policy_version(_batch(versions=(10, 10)))
    second = trainer._validate_rollout_policy_version(_batch(versions=(10, 10)))

    assert first == second == 10


def test_actual_server_global_steps_is_captured_as_rollout_policy_version():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer._rollout_policy_version = 10
    batch = _batch(versions=(0, 0))
    batch.non_tensor_batch.pop("rollout_policy_version")
    batch.non_tensor_batch["global_steps"] = np.asarray([10, 10], dtype=object)

    actual = trainer._capture_actual_rollout_policy_version(batch)

    assert actual == 10
    assert batch.non_tensor_batch["rollout_policy_version"].tolist() == [10, 10]


def test_mixed_generation_batch_versions_fail_before_probe():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer._rollout_policy_version = 10

    with pytest.raises(ValueError, match="mixed rollout policy versions"):
        trainer._validate_rollout_policy_version(_batch(versions=(10, 11)))


def test_missing_actual_rollout_version_has_no_manual_fallback():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.global_steps = 10
    trainer._rollout_policy_version = 10
    batch = _batch()
    batch.non_tensor_batch.pop("rollout_policy_version")

    with pytest.raises(ValueError, match="missing actual rollout policy version"):
        trainer._validate_rollout_policy_version(batch)


def test_policy_version_changes_only_after_successful_weight_update():
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer._rollout_policy_version = 10
    updates = []

    def fail(version):
        updates.append(version)
        raise RuntimeError("sync failed")

    trainer.checkpoint_manager = SimpleNamespace(update_weights=fail)
    with pytest.raises(RuntimeError, match="sync failed"):
        trainer._publish_rollout_policy_version(11)
    assert trainer._rollout_policy_version == 10

    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda version: updates.append(version))
    trainer._publish_rollout_policy_version(11)

    assert updates == [11, 11]
    assert trainer._rollout_policy_version == 11


def test_mock_update_event_order_places_probe_and_redistribution_before_actor(monkeypatch):
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=True)
    trainer.config.algorithm.gamma = 1.0
    trainer.config.algorithm.lam = 1.0
    trainer.config.algorithm.norm_adv_by_std_in_grpo = True
    trainer.config.algorithm.get = lambda name, default=None: getattr(trainer.config.algorithm, name, default)
    trainer.config.actor_rollout_ref.rollout.n = 4
    trainer.global_steps = 4
    trainer._rollout_policy_version = 3
    events = ["terminal_reward", "filter", "final_selection"]
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: events.append("sleep"))
    trainer._probe_final_retained_batch = MethodType(
        lambda self, batch, _metrics, _timing: events.append("probe") or batch, trainer
    )
    trainer._compute_probe_credit_advantage = MethodType(
        lambda self, batch, _metrics: events.append("redistribute") or batch, trainer
    )
    trainer._update_actor = MethodType(
        lambda self, _batch: events.append("actor") or SimpleNamespace(meta_info={"metrics": {}}), trainer
    )
    monkeypatch.setattr(
        dapo_trainer_module,
        "compute_advantage",
        lambda batch, **_kwargs: events.append("standard_grpo") or batch,
    )

    batch = trainer._prepare_final_retained_batch(_batch(), {}, {})
    trainer._compute_advantage_and_actor_update(batch, {}, {})

    assert events == [
        "terminal_reward",
        "filter",
        "final_selection",
        "probe",
        "sleep",
        "standard_grpo",
        "redistribute",
        "actor",
    ]


@pytest.mark.parametrize(
    ("values", "valid_mask", "message"),
    [
        (torch.zeros(2, 5), torch.ones(2, 4, dtype=torch.bool), "same shape"),
        (torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool), "exactly 5 positions"),
        (
            torch.zeros(2, 5),
            torch.tensor([[True] * 5, [True, True, False, True, True]]),
            "all Probe values must be valid",
        ),
        (torch.tensor([[0.0, 0.0, float("nan"), 0.0, 0.0]]), torch.ones(1, 5, dtype=torch.bool), "finite"),
        (torch.tensor([[0.0, 0.0, float("inf"), 0.0, 0.0]]), torch.ones(1, 5, dtype=torch.bool), "finite"),
        (torch.tensor([[0.0, 0.0, 1.1, 0.0, 0.0]]), torch.ones(1, 5, dtype=torch.bool), r"in \[0, 1\]"),
        (torch.tensor([[0.0, 0.0, -0.1, 0.0, 0.0]]), torch.ones(1, 5, dtype=torch.bool), r"in \[0, 1\]"),
    ],
)
def test_probe_data_is_validated_before_advantage(values, valid_mask, message):
    trainer = object.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = _config(enable=True)
    batch = DataProto.from_dict(tensors={"probe_values": values, "probe_valid_mask": valid_mask})

    with pytest.raises(ValueError, match=message):
        trainer._compute_probe_credit_advantage(batch, {})
