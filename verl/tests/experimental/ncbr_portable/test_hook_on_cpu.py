import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.natural_continuation_boundary_return.config import map_ncbr_config, resolve_boundary_config
from verl.experimental.natural_continuation_boundary_return.hook import NCBRHook
from verl.experimental.natural_continuation_boundary_return.reward_adapter import BoundaryRewardOutput
from verl.experimental.natural_continuation_boundary_return.scoring import score_long_generations
from verl.workers.config.rollout import BoundaryReturnConfig


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("mode", ["off", "shadow", "replace"])
def test_hook_four_transitions_padding_and_input_immutability(dtype, mode):
    mask = torch.ones((4, 4), dtype=torch.long)
    raw = torch.tensor([[-.125, 0, 0, value] for value in (0, 0, 1, 1)], dtype=dtype)
    batch = DataProto.from_dict(tensors={
        "prompts": torch.ones((4, 2), dtype=torch.long), "responses": torch.arange(16).reshape(4, 4)+10,
        "response_mask": mask, "loss_mask": mask.clone(), "attention_mask": torch.ones((4, 6), dtype=torch.long),
        "input_ids": torch.arange(24).reshape(4, 6), "position_ids": torch.arange(6).repeat(4, 1),
        "old_log_probs": torch.full((4, 4), -1.), "ref_log_prob": torch.full((4, 4), -2.),
        "rollout_log_probs": torch.full((4, 4), -3.), "rm_scores": raw.clone(),
    }, non_tensors={
        "uid": np.array(["a", "a", "b", "b"], dtype=object),
        "trajectory_id": np.array([f"t{i}" for i in range(4)], dtype=object),
        "rollout_policy_version": np.array([7]*4), "finish_reason": np.array(["length"]*4, dtype=object),
        "long_score": np.array([0., 1., 0., 1.]),
    })
    original, raw_original = copy.deepcopy(batch), raw.clone()
    extras = {"acc": np.array([0., 0., 1., 1.]), "score": np.array([0., 0., 1., 1.])}
    extras_original = copy.deepcopy(extras)
    requests, scored = [], []
    class Client:
        async def start_grouped(self, request_id, **kw):
            requests.append((request_id, kw))
            class Handle:
                server_id = "mock"
                backend_request_id = request_id
                async def result(self):
                    return [SimpleNamespace(token_ids=[80], stop_reason="completed",
                        extra_fields={"branch_id": 0, "global_steps": 7, "finish_reason": "stop"})]
                async def abort(self): pass
                async def drain(self): pass
                async def release(self): pass
            return Handle()
    def score(long_batch):
        scored.append(len(long_batch))
        values = long_batch.non_tensor_batch["long_score"]
        return BoundaryRewardOutput(torch.zeros((len(long_batch), 5)), {"acc": values, "score": values})
    config = BoundaryReturnConfig(mode=mode, long_response_length=8, long_reward_chunk_size=3)
    outcome = NCBRHook().apply(batch, raw, raw_verifier_extras=extras, config=config, policy_version=7,
        sampling_params={"temperature": 1., "top_p": 1., "top_k": -1}, continuation_client=Client(),
        score_long=lambda b, g, c: score_long_generations(b, g, c, score_batch=score, pad_token_id=0,
                                                        reward_worker_count=2),
        eos_token_id=99, short_response_length=4, max_model_len=10)
    for key in original.batch.keys(): assert torch.equal(batch.batch[key], original.batch[key]), key
    for key in original.non_tensor_batch: assert np.array_equal(batch.non_tensor_batch[key], original.non_tensor_batch[key])
    assert torch.equal(raw, raw_original)
    for key in extras: assert np.array_equal(extras[key], extras_original[key])
    expected = raw.clone()
    if mode == "replace": expected[:, -1] += torch.tensor([0, 1, -1, 0], dtype=dtype)
    assert torch.equal(outcome.effective_scores, expected)
    assert outcome.effective_scores.dtype == raw.dtype
    if mode == "off":
        assert not requests and not scored
        assert outcome.effective_scores is raw
    else:
        assert len(requests) == 4 and scored == [4, 2]
        assert outcome.boundary_mask.tolist() == [True]*4
        assert outcome.corrected_mask.tolist() == ([False, True, True, False] if mode == "replace" else [False]*4)
        assert "boundary_group_unlocked" not in outcome.aux["row_labels"]


@pytest.mark.parametrize("name", ["ppo_trainer", "natural_continuation_boundary_return_dapo_trainer"])
@pytest.mark.parametrize("enable,mode,expected", [(None,"replace","replace"), (False,"replace","off"),
                                               (True,"shadow","shadow"), (None,"off","off")])
def test_hydra_enable_mapping(name, enable, mode, expected):
    config_dir = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=name, overrides=[f"ncbr.enable={str(enable).lower() if enable is not None else 'null'}",
            f"actor_rollout_ref.rollout.boundary_return.mode={mode}"])
    boundary = map_ncbr_config(config)
    assert boundary.mode == expected
    assert config.actor_rollout_ref.rollout.boundary_return.mode == expected
    assert resolve_boundary_config(config).mode == expected


@pytest.mark.parametrize("enable", [True, "true", 1])
def test_enable_rejects_ambiguous_configuration(enable):
    cfg = OmegaConf.create({"ncbr": {"enable": enable}, "actor_rollout_ref": {"rollout": {"boundary_return": {"mode":"off"}}}})
    with pytest.raises(ValueError): resolve_boundary_config(cfg)


def test_default_standard_config_is_off():
    config_dir = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer")
    assert resolve_boundary_config(config).mode == "off"


@pytest.mark.parametrize("reason,tokens,expected", [("length",[1,2,3,99],True), ("stop",[1,2,3,4],False),
    (None,[1,2,3,4],True), (None,[1,2,3,99],False), ("completed",[1,2,3,4],True), ("abort",[1,2,3,4],False)])
def test_original_finish_reason_priority(reason, tokens, expected):
    from verl.trainer.ppo.forced_answer_probe import detect_hit_response_cap
    assert detect_hit_response_cap(finish_reasons=[reason], response_lengths=[4], max_response_length=4,
                                   response_token_ids=[tokens], eos_token_id=99).tolist() == [expected]


def test_explicit_false_hook_is_identity_before_any_dependency_access():
    raw = torch.ones((1, 4))
    def forbidden(*a, **kw): pytest.fail("disabled hook accessed an auxiliary service")
    result = NCBRHook(continuation_runner=forbidden, reward_adapter=forbidden).apply(
        None, raw, raw_verifier_extras={}, config=BoundaryReturnConfig(enable=False, mode="replace"),
        policy_version=None, sampling_params=None, continuation_client=None, score_long=forbidden,
        eos_token_id=None, short_response_length=None, max_model_len=None)
    assert result.effective_scores is raw
    assert result.aux is None


def test_dapo_explicit_false_matches_legacy_off():
    import source_harness
    from verl.experimental.natural_continuation_boundary_return.dapo_trainer import RayDAPOBoundaryReturnTrainer
    original_fit = RayDAPOBoundaryReturnTrainer.fit
    runs = []
    for mode, enable in (("off", None), ("replace", False)):
        with pytest.MonkeyPatch.context() as mp:
            def fit(self):
                self.config.ncbr = {"enable": enable}
                return original_fit(self)
            mp.setattr(RayDAPOBoundaryReturnTrainer, "fit", fit)
            runs.append(source_harness._run_fit_harness(mp, mode=mode, generation_batches=2))
    a, b = runs
    assert a.events == b.events
    for key in a.actor_batch.batch.keys(): assert torch.equal(a.actor_batch.batch[key], b.actor_batch.batch[key]), key
    assert "long_reward" not in b.events
