from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from analysis.fa_cac_v2.tools.dapo_adapter import (
    EXPECTED_DAPO_FIT_SHA256,
    attest_canonical_sources,
    build_patched_dapo_fit_source,
)
from verl import DataProto
from verl.trainer.ppo.forced_answer_probe import (
    ForcedAnswerCensorEvidence,
    ForcedAnswerProbeCapture,
    ForcedAnswerProbeDiagnostics,
    ForcedAnswerProbeScoreResult,
    ForcedAnswerTrainingCreditResult,
)


class _Tracking:
    records = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def log(self, data, step):
        self.records.append((data, step))


class _Progress:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _config(*, v1=False, cac=False, apply=True):
    return OmegaConf.create(
        {
            "trainer": {
                "project_name": "cpu-harness",
                "experiment_name": "cpu-harness",
                "logger": ["console"],
                "val_before_train": False,
                "val_only": False,
                "total_epochs": 2,
                "balance_batch": False,
                "critic_warmup": 0,
                "save_freq": -1,
                "test_freq": -1,
                "esi_redundant_time": 0,
                "rollout_data_dir": None,
                "default_local_dir": "/tmp/fa-cac-v2-canonical-harness",
            },
            "actor_rollout_ref": {
                "rollout": {
                    "n": 2,
                    "temperature": 1.0,
                    "max_model_len": 16,
                    "skip_rollout": False,
                    "forced_answer_probe": {
                        "_target_": "verl.workers.config.ForcedAnswerProbeConfig",
                        "enable": bool(v1 or cac),
                        "training_credit": {
                            "_target_": "verl.workers.config.ForcedAnswerTrainingCreditConfig",
                            "enable": v1,
                        },
                    },
                }
            },
            "algorithm": {
                "adv_estimator": "grpo",
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
                "filter_groups": {"enable": False, "metric": "acc", "max_num_gen_batches": 1},
                "rollout_correction": None,
                "censor_aware_advantage": {
                    "_target_": "verl.trainer.config.CensorAwareAdvantageConfig",
                    "enable": cac,
                    "apply": apply,
                    "mode": "attenuate_negative_correctness",
                },
            },
            "global_profiler": {"steps": None, "profile_continuous_steps": False},
        }
    )


def _rollout_output():
    prompts = torch.tensor([[1], [1], [2], [2]])
    responses = torch.tensor([[11, 12, 0], [21, 0, 0], [31, 32, 33], [41, 0, 0]])
    response_mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    attention_mask = torch.cat((torch.ones_like(prompts), response_mask), dim=-1)
    rewards = torch.tensor([[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    return DataProto(
        batch=TensorDict(
            {
                "responses": responses,
                "input_ids": torch.cat((prompts, responses), dim=-1),
                "attention_mask": attention_mask,
                "position_ids": torch.clamp(attention_mask.cumsum(-1) - 1, min=0),
                "response_mask": response_mask,
                "rm_scores": rewards,
            },
            batch_size=4,
        ),
        non_tensor_batch={
            "score": np.asarray([-1.0, 1.0, -1.0, 1.0]),
            "acc": np.asarray([0.0, 1.0, 0.0, 1.0]),
        },
        meta_info={"timing": {}, "reward_extra_keys": ["score", "acc"]},
    )


def _probe_result(original_rewards, *, v1):
    effective = original_rewards.clone()
    if v1:
        effective[0].zero_()
        effective[0, 1] = 0.5
    evidence = ForcedAnswerCensorEvidence(
        current_row_to_parent=np.arange(4, dtype=np.int64),
        hit_response_cap=np.asarray([True, False, False, False]),
        probe_attempted=np.asarray([True, False, False, False]),
        context_overflow=np.zeros(4, dtype=bool),
        original_correctness_by_parent=np.asarray([0.0, 1.0, 0.0, 1.0]),
        task_score_by_parent=np.asarray([-1.0, 1.0, -1.0, 1.0]),
        pfa_by_parent={0: 0.75},
        correctness_threshold=0.5,
    )
    return ForcedAnswerProbeScoreResult(
        diagnostics=ForcedAnswerProbeDiagnostics(metrics={}, rewards_by_parent={}, successes_by_parent={}),
        training_credit=ForcedAnswerTrainingCreditResult(
            effective_reward_tensor=effective,
            corrected_parent_indices=(0,) if v1 else (),
            pfa_by_parent={0: 0.75},
            target_reward_by_parent={0: 0.5} if v1 else {},
            metrics={},
        ),
        censor_evidence=evidence,
    )


def _run_fit(fit_fn, monkeypatch, *, v1=False, cac=False, apply=True):
    from recipe.dapo import dapo_ray_trainer
    import verl.utils.tracking

    monkeypatch.setattr(verl.utils.tracking, "Tracking", _Tracking)
    monkeypatch.setattr(dapo_ray_trainer, "tqdm", _Progress)
    trainer = dapo_ray_trainer.RayDAPOTrainer.__new__(dapo_ray_trainer.RayDAPOTrainer)
    trainer.config = _config(v1=v1, cac=cac, apply=apply)
    trainer.total_training_steps = 1
    trainer.train_dataloader = [
        {
            "prompts": torch.tensor([[1], [2]]),
        }
    ]
    trainer.use_rm = False
    trainer.use_critic = False
    trainer.actor_rollout_wg = SimpleNamespace()
    trainer._load_checkpoint = lambda: None
    trainer._save_checkpoint = lambda: None
    trainer._start_profiling = lambda *args, **kwargs: None
    trainer._stop_profiling = lambda *args, **kwargs: None
    trainer._get_gen_batch = lambda batch: DataProto(
        batch=TensorDict({"prompts": batch.batch["prompts"].clone()}, batch_size=2)
    )
    output = _rollout_output()
    trainer.async_rollout_manager = SimpleNamespace(generate_sequences=lambda batch: output)
    trainer.checkpoint_manager = SimpleNamespace(
        update_weights=lambda *args, **kwargs: None,
        sleep_replicas=lambda: None,
    )
    trainer.compute_kl_related_metrics = lambda batch, metrics, timing: batch
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    capture = ForcedAnswerProbeCapture(
        hit_response_cap=np.asarray([True, False, False, False]),
        probe_attempted=np.asarray([True, False, False, False]),
        context_overflow=np.zeros(4, dtype=bool),
        generations=(),
        probe_input_tokens=0,
    )
    trainer._forced_answer_probe_enabled = lambda: bool(v1 or cac)
    trainer._generate_forced_answer_probe_with_replica_cleanup = lambda *args, **kwargs: capture
    trainer._score_forced_answer_probe = lambda **kwargs: _probe_result(
        kwargs["original_reward_tensor"], v1=v1
    )
    captured = {}

    def update_actor(batch):
        captured["batch"] = batch
        return DataProto(meta_info={"metrics": {}})

    trainer._update_actor = update_actor
    fit_fn(trainer)
    return captured["batch"]


@pytest.fixture(scope="module")
def fit_functions():
    from recipe.dapo import dapo_ray_trainer

    canonical = textwrap.dedent(inspect.getsource(dapo_ray_trainer.RayDAPOTrainer.fit))
    namespace = dapo_ray_trainer.__dict__.copy()
    exec(compile(canonical, "<canonical_fit>", "exec"), namespace)
    canonical_fit = namespace["fit"]
    patched = build_patched_dapo_fit_source(canonical)
    exec(compile(patched, "<fa_cac_fit>", "exec"), namespace)
    return canonical_fit, namespace["fit"]


def test_canonical_source_attestation_and_import_chain():
    evidence = attest_canonical_sources()
    assert evidence["canonical_commit"] == "7aed6b230776f963fa09509c10d9c3a767d1102c"
    launcher = Path("analysis/fa_cac_v2/tools/matched_dapo_main.py").read_text()
    assert "analysis.fa_cac_v2.tools.dapo_adapter" in launcher
    assert "MatchedFACACDAPOTaskRunner" in launcher


def test_source_guard_rejects_any_canonical_drift():
    with pytest.raises(RuntimeError, match="source changed"):
        build_patched_dapo_fit_source("def fit(self):\n    pass\n")
    assert len(EXPECTED_DAPO_FIT_SHA256) == 64


def test_cac_disabled_v2_matches_unpatched_canonical_tensor_for_tensor(fit_functions, monkeypatch):
    canonical_fit, patched_fit = fit_functions
    vanilla = _run_fit(canonical_fit, monkeypatch)
    disabled = _run_fit(patched_fit, monkeypatch)
    for key in ("token_level_scores", "token_level_rewards", "advantages", "returns"):
        assert torch.equal(disabled.batch[key], vanilla.batch[key])


def test_v1_only_keeps_historical_reward_before_grpo_semantics(fit_functions, monkeypatch):
    _, patched_fit = fit_functions
    vanilla = _run_fit(patched_fit, monkeypatch)
    v1 = _run_fit(patched_fit, monkeypatch, v1=True)
    assert not torch.equal(v1.batch["token_level_rewards"], vanilla.batch["token_level_rewards"])
    assert not torch.equal(v1.batch["advantages"], vanilla.batch["advantages"])
    assert v1.batch["token_level_rewards"][0].sum().item() == pytest.approx(0.5)


def test_cac_only_changes_actor_advantages_and_keeps_rewards(fit_functions, monkeypatch):
    _, patched_fit = fit_functions
    vanilla = _run_fit(patched_fit, monkeypatch)
    cac = _run_fit(patched_fit, monkeypatch, cac=True)
    assert torch.equal(cac.batch["token_level_scores"], vanilla.batch["token_level_scores"])
    assert torch.equal(cac.batch["token_level_rewards"], vanilla.batch["token_level_rewards"])
    assert not torch.equal(cac.batch["advantages"], vanilla.batch["advantages"])
    assert torch.equal(cac.batch["returns"], cac.batch["advantages"])


def test_both_interventions_fail_before_training(fit_functions, monkeypatch):
    _, patched_fit = fit_functions
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run_fit(patched_fit, monkeypatch, v1=True, cac=True)
