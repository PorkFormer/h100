import json
import sys
import types
from types import MethodType, SimpleNamespace

import pytest
import torch

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
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer  # noqa: E402
from verl.experimental.success_support_floor.cache import (  # noqa: E402
    canonical_prompt_key,
    tokenizer_fingerprints,
    write_cache,
)
from verl.experimental.success_support_floor.dapo_trainer import (  # noqa: E402
    RayDAPOSuccessSupportFloorTrainer,
    support_metrics_from_log_probs,
)
from verl.experimental.success_support_floor import dapo_trainer as bssf_trainer_module  # noqa: E402
from verl.trainer.config import (  # noqa: E402
    ProbeCreditConfig,
    ReadinessDominanceConfig,
    SuccessSupportFloorConfig,
)


class _Tokenizer:
    pad_token_id = 0
    chat_template = "template"
    special_tokens_map = {"eos_token": "</s>"}

    def get_vocab(self):
        return {"a": 1, "</s>": 2}


def _cache(path):
    tokenizer = _Tokenizer()
    tok, template = tokenizer_fingerprints(tokenizer)
    key = canonical_prompt_key(tok, template, [1])
    manifest = {
        "schema_version": 1,
        "algorithm": "budgeted_success_support_floor",
        "reference_model_id": "base",
        "reference_model_hash": "a" * 64,
        "reference_budget": 4,
        "base_rollouts_per_prompt": 2,
        "support_threshold": 2,
        "tokenizer_fingerprint": tok,
        "chat_template_fingerprint": template,
        "prompt_manifest_fingerprint": "prompts",
        "verifier_fingerprint": "verifier",
        "logprob_temperature": 1.0,
        "logprob_convention": "response-token-sum",
        "include_eos": True,
        "created_at": "now",
        "source_git_commit": "sha",
    }
    prompts = [{
        "prompt_key": key,
        "prompt_id": 0,
        "original_dataset_index": 0,
        "prompt_hash": "p",
        "prompt_token_ids": [1],
        "prompt_token_count": 1,
        "base_rollout_count": 2,
        "eligible_success_count": 2,
        "q_reference": 1.0,
    }]
    witnesses = [
        {
            "prompt_key": key,
            "witness_id": index,
            "source_rollout_index": index,
            "response_token_ids": [2],
            "response_token_count": 1,
            "reference_seq_logprob": -1.0,
            "reference_mean_logprob": -1.0,
            "finish_reason": "eos",
            "full_reward": True,
            "prefix_reward_reference_budget": True,
            "response_hash": f"r{index}",
        }
        for index in range(2)
    ]
    fingerprint = write_cache(path, manifest, prompts, witnesses)
    (path / "validation_report.json").write_text(
        json.dumps({"cache_fingerprint": fingerprint, "passed": True})
    )
    return tokenizer


def _config(cache_path, *, mode="shadow"):
    algorithm = SimpleNamespace(
        adv_estimator="grpo",
        probe_credit=ProbeCreditConfig(),
        readiness_dominance=ReadinessDominanceConfig(),
        success_support_floor=SuccessSupportFloorConfig(
            mode=mode,
            cache_path=str(cache_path) if cache_path is not None else None,
            reference_budget=4,
            support_threshold=2,
            constraint_batch_size=1,
            update_interval=2,
        ),
        use_kl_in_reward=False,
        rollout_correction=None,
        filter_groups=SimpleNamespace(enable=True, metric="acc"),
        gamma=1.0,
        lam=1.0,
    )
    algorithm.get = lambda name, default=None: getattr(algorithm, name, default)
    actor = SimpleNamespace(
        loss_agg_mode="token-mean",
        ppo_mini_batch_size=1,
        ppo_epochs=1,
        use_kl_loss=False,
    )
    rollout = SimpleNamespace(
        name="vllm",
        n=1,
        temperature=1.0,
        multi_turn=SimpleNamespace(enable=False),
    )
    return SimpleNamespace(
        algorithm=algorithm,
        actor_rollout_ref=SimpleNamespace(actor=actor, rollout=rollout),
        distillation=SimpleNamespace(enabled=False),
        global_profiler=SimpleNamespace(steps=None),
    )


def _trainer(config, tokenizer):
    trainer = object.__new__(RayDAPOSuccessSupportFloorTrainer)
    trainer.config = config
    trainer.tokenizer = tokenizer
    trainer.use_critic = False
    trainer.use_teacher_policy = False
    trainer.use_reference_policy = False
    trainer.global_steps = 2
    return trainer


def test_off_mode_does_not_touch_missing_cache():
    trainer = _trainer(_config(None, mode="off"), _Tokenizer())
    trainer._validate_probe_credit_mode()
    assert trainer._success_support_cache is None


def test_shadow_validation_loads_strict_cache(tmp_path):
    tokenizer = _cache(tmp_path)
    trainer = _trainer(_config(tmp_path), tokenizer)
    trainer._validate_probe_credit_mode()
    assert trainer._success_support_cache.manifest["witness_count"] == 2
    assert trainer._lambda == 0.0


def test_shadow_requires_passed_logprob_validation_report(tmp_path):
    tokenizer = _cache(tmp_path)
    (tmp_path / "validation_report.json").unlink()
    trainer = _trainer(_config(tmp_path), tokenizer)

    with pytest.raises(ValueError, match="validation report is missing"):
        trainer._validate_probe_credit_mode()


def test_shadow_rejects_wrong_materialized_reference_model(tmp_path):
    cache_path = tmp_path / "cache"
    tokenizer = _cache(cache_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors").write_bytes(b"different model")
    trainer = _trainer(_config(cache_path), tokenizer)
    trainer._bssf_reference_model_local_path = str(model_path)

    with pytest.raises(ValueError, match="weight hash mismatch"):
        trainer._validate_probe_credit_mode()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: setattr(c.algorithm, "probe_credit", ProbeCreditConfig(enable=True)), "ProbeCredit"),
        (
            lambda c: setattr(
                c.algorithm, "readiness_dominance", ReadinessDominanceConfig(mode="shadow")
            ),
            "ReadinessDominance",
        ),
        (lambda c: setattr(c.actor_rollout_ref.actor, "use_kl_loss", True), "KL"),
        (lambda c: setattr(c.actor_rollout_ref.rollout, "name", "sglang"), "vLLM"),
    ],
)
def test_validation_rejects_unsupported_combinations(tmp_path, mutation, message):
    tokenizer = _cache(tmp_path)
    config = _config(tmp_path)
    mutation(config)
    trainer = _trainer(config, tokenizer)
    with pytest.raises(ValueError, match=message):
        trainer._validate_probe_credit_mode()


def test_support_metrics_have_unambiguous_names():
    metrics = support_metrics_from_log_probs(
        torch.tensor([[-2.0, -1.0], [-1.0, 0.0]]),
        torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        torch.tensor([-2.0, -1.0]),
        alpha=0.5,
        delta=0.05,
        lambda_value=0.2,
    )
    assert metrics["support_floor/shortfall_mean"] > 0
    assert "support_floor/log_ratio_mean" in metrics
    assert "support_floor/constraint_residual" in metrics


def test_shadow_runs_after_standard_actor_update_without_dual_update(monkeypatch, tmp_path):
    tokenizer = _cache(tmp_path)
    trainer = _trainer(_config(tmp_path), tokenizer)
    trainer._validate_probe_credit_mode()
    events = []
    batch = DataProto.from_dict(tensors={"dummy": torch.ones(1, 1)})
    actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": {}})

    def parent_update(self, received, metrics, timing):
        events.append("actor")
        return received, actor_output

    monkeypatch.setattr(RayDAPOProbeCreditTrainer, "_compute_advantage_and_actor_update", parent_update)
    trainer._compute_shadow_metrics = MethodType(
        lambda self: events.append("shadow") or {"support_floor/update_applied": 0.0}, trainer
    )
    result, output = trainer._compute_advantage_and_actor_update(batch, {}, {})

    assert events == ["actor", "shadow"]
    assert result is batch and output is actor_output
    assert trainer._lambda == 0.0


def test_active_update_orders_advantage_support_actor_then_dual(monkeypatch, tmp_path):
    tokenizer = _cache(tmp_path)
    trainer = _trainer(_config(tmp_path, mode="dual"), tokenizer)
    trainer._validate_probe_credit_mode()
    events = []
    batch = DataProto.from_dict(tensors={"response_mask": torch.ones(1, 1)})
    support = DataProto.from_dict(tensors={"response_mask": torch.ones(1, 1)})
    augmented = DataProto.from_dict(tensors={"dummy": torch.ones(2, 1)})
    actor_output = DataProto.from_single_dict(
        data={},
        meta_info={
            "metrics": {
                "actor/support_floor_unweighted_shortfall": [1.0],
                "actor/support_floor_log_ratio_mean": [-1.0],
                "actor/support_floor_active_fraction": [1.0],
            }
        },
    )

    monkeypatch.setattr(
        bssf_trainer_module,
        "compute_advantage",
        lambda received, **_kwargs: events.append("advantage") or received,
    )
    trainer._build_support_batch = MethodType(
        lambda self: events.append("support") or support, trainer
    )
    monkeypatch.setattr(
        bssf_trainer_module,
        "build_augmented_actor_batch",
        lambda *_args, **_kwargs: events.append("augment") or augmented,
    )
    trainer._update_actor_augmented = MethodType(
        lambda self, received, *, rl_batch_size: events.append("actor") or actor_output,
        trainer,
    )
    real_update_dual = trainer._update_dual
    trainer._update_dual = MethodType(
        lambda self, output, metrics: events.append("dual") or real_update_dual(output, metrics),
        trainer,
    )
    metrics = {}
    result, output = trainer._compute_advantage_and_actor_update(batch, metrics, {})

    assert events == ["advantage", "support", "augment", "actor", "dual"]
    assert result is batch and output is actor_output
    assert trainer._violation_ema == pytest.approx(0.1)
    assert trainer._lambda == pytest.approx(0.0005)
    assert metrics["support_floor/constraint_residual"] == pytest.approx(0.95)


def test_dual_sums_optimizer_partitions_and_averages_ppo_epochs(tmp_path):
    tokenizer = _cache(tmp_path)
    trainer = _trainer(_config(tmp_path, mode="dual"), tokenizer)
    trainer._validate_probe_credit_mode()
    trainer.config.actor_rollout_ref.actor.ppo_epochs = 2
    actor_output = DataProto.from_single_dict(
        data={},
        meta_info={
            "metrics": {
                "actor/support_floor_unweighted_shortfall": [0.2, 0.3, 0.4, 0.5],
                "actor/support_floor_log_ratio_mean": [-0.1, -0.2, -0.3, -0.4],
                "actor/support_floor_active_fraction": [0.2, 0.3, 0.4, 0.5],
                "actor/support_floor_quantile_weight": [1.0, 0.0, 1.0, 0.0],
                "actor/support_floor_log_ratio_p50": [-0.5, 0.0, -0.7, 0.0],
            }
        },
    )
    metrics = {}

    trainer._update_dual(actor_output, metrics)

    assert metrics["support_floor/shortfall_mean"] == pytest.approx(0.7)
    assert metrics["support_floor/log_ratio_mean"] == pytest.approx(-0.5)
    assert metrics["support_floor/log_ratio_p50"] == pytest.approx(-0.6)


def test_explicit_resume_path_selects_colocated_bssf_state(tmp_path):
    trainer = object.__new__(RayDAPOSuccessSupportFloorTrainer)
    checkpoint = tmp_path / "external" / "global_step_7"
    trainer.global_steps = 7
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            resume_mode="resume_path",
            resume_from_path=str(checkpoint),
            default_local_dir=str(tmp_path / "default"),
        )
    )

    assert trainer._resume_state_path() == str(
        checkpoint / "success_support_floor" / "state.json"
    )
