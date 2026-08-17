from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.config import CensorAwareAdvantageConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.censor_aware_advantage import apply_fa_cac_post_advantage_hook
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_grpo_outcome_advantage
from verl.trainer.ppo.forced_answer_probe import ForcedAnswerCensorEvidence, build_fa_cac_evidence
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils.config import validate_censor_aware_advantage_config
from verl.workers.config import ForcedAnswerProbeConfig, ForcedAnswerTrainingCreditConfig


def _frozen_pre_refactor_grpo(rewards, mask, uid, *, norm):
    scores = rewards.sum(-1)
    grouped = defaultdict(list)
    means = {}
    stds = {}
    with torch.no_grad():
        for row in range(len(scores)):
            grouped[uid[row]].append(scores[row])
        for key in grouped:
            if len(grouped[key]) == 1:
                means[key] = torch.tensor(0.0)
                stds[key] = torch.tensor(1.0)
            else:
                values = torch.stack(grouped[key])
                means[key] = torch.mean(values)
                stds[key] = torch.std(values)
        for row in range(len(scores)):
            if norm:
                scores[row] = (scores[row] - means[uid[row]]) / (stds[uid[row]] + 1e-6)
            else:
                scores[row] = scores[row] - means[uid[row]]
        scores = scores.unsqueeze(-1) * mask
    return scores, scores


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("norm", [False, True])
def test_randomized_grpo_refactor_golden(dtype, norm):
    generator = torch.Generator().manual_seed(20260815)
    uid = np.asarray(["a", "a", "a", "b", "c", "c", "d", "d", "d", "d"], dtype=object)
    for _ in range(20):
        rewards = torch.randn((len(uid), 9), generator=generator, dtype=torch.float32).to(dtype)
        lengths = torch.randint(1, 10, (len(uid),), generator=generator)
        mask = (torch.arange(9).unsqueeze(0) < lengths.unsqueeze(1)).to(dtype)
        expected_adv, expected_returns = _frozen_pre_refactor_grpo(rewards, mask, uid, norm=norm)
        actual_adv, actual_returns = compute_grpo_outcome_advantage(
            rewards, mask, uid, norm_adv_by_std_in_grpo=norm
        )
        assert torch.equal(actual_adv, expected_adv)
        assert torch.equal(actual_returns, expected_returns)


def _config(*, cac=False, apply=True, v1=False, probe=True, max_model_len=4096, adv="grpo", mode=None):
    cac_config = CensorAwareAdvantageConfig(
        enable=cac,
        apply=apply,
        mode=mode or "attenuate_negative_correctness",
    )
    probe_config = ForcedAnswerProbeConfig(
        enable=probe,
        training_credit=ForcedAnswerTrainingCreditConfig(enable=v1),
    )
    return SimpleNamespace(
        algorithm=SimpleNamespace(
            adv_estimator=adv,
            norm_adv_by_std_in_grpo=True,
            censor_aware_advantage=cac_config,
        ),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(forced_answer_probe=probe_config, max_model_len=max_model_len)
        ),
    )


def test_cac_config_defaults():
    config = CensorAwareAdvantageConfig()
    assert (config.enable, config.apply, config.mode) == (
        False,
        True,
        "attenuate_negative_correctness",
    )
    config.validate()


def test_frozen_v2_tensors_and_metrics_remain_unchanged():
    data = _batch()
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(), algorithm_config=_algorithm()
    )
    expected = torch.tensor(
        [
            [-0.3535532653, -0.3535532653, -0.3535532653, -0.0],
            [0.7071064711, 0.7071064711, 0.0, 0.0],
            [-0.7071062922, -0.7071062922, -0.7071062922, -0.7071062922],
            [0.7071062922, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(data.batch["advantages"], expected)
    assert torch.equal(data.batch["returns"], expected)
    frozen_metrics = {
        "fa_cac/vanilla_adv_mean": -0.7071064710617065,
        "fa_cac/task_adv_mean": -0.4714043438434601,
        "fa_cac/reg_adv_mean": -0.23570217192173004,
        "fa_cac/pre_adv_mean": -0.35355326533317566,
        "fa_cac/projected_adv_mean": -0.35355326533317566,
        "fa_cac/drift_abs_mean": 0.3535532057285309,
        "fa_cac/batch_before_adv_token_weighted_sum": -2.828425347805023,
        "fa_cac/batch_after_adv_token_weighted_sum": -1.7677657306194305,
        "fa_cac/candidate_count": 1.0,
        "fa_cac/eligible_count": 1.0,
        "fa_cac/sign_clamp_count": 0.0,
        "fa_cac/reward_drift_max": 0.0,
        "fa_cac/score_drift_max": 0.0,
    }
    assert {key: metrics[key] for key in frozen_metrics} == frozen_metrics


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cac": True, "probe": False}, "requires forced_answer_probe.enable=true"),
        ({"cac": True, "max_model_len": None}, "requires an explicit"),
        ({"cac": True, "adv": "gae"}, "supports only"),
        ({"cac": True, "mode": "replacement"}, "mode must be"),
        ({"cac": True, "v1": True}, "mutually exclusive"),
    ],
)
def test_cac_config_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_censor_aware_advantage_config(_config(**kwargs))


@pytest.mark.parametrize(
    ("v1", "cac", "valid"),
    [(False, False, True), (True, False, True), (False, True, True), (True, True, False)],
)
def test_four_mode_configuration_truth_table(v1, cac, valid):
    config = _config(v1=v1, cac=cac)
    if valid:
        validate_censor_aware_advantage_config(config)
    else:
        with pytest.raises(ValueError, match="mutually exclusive"):
            validate_censor_aware_advantage_config(config)


def _batch(total_scores=(-2.0, 1.0, -1.0, 1.0), *, norm=True):
    total = torch.tensor(total_scores, dtype=torch.float32)
    mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32
    )
    rewards = torch.zeros_like(mask)
    for row, scalar in enumerate(total):
        rewards[row, int(mask[row].sum().item()) - 1] = scalar
    data = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": rewards,
                "token_level_scores": rewards.clone(),
                "response_mask": mask,
                "untouched_tensor": torch.arange(len(total) * 4).reshape(len(total), 4),
            },
            batch_size=len(total),
        ),
        non_tensor_batch={"uid": np.asarray(["a", "a", "b", "b"], dtype=object)},
    )
    compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        norm_adv_by_std_in_grpo=norm,
        config={},
    )
    return data


def _tensor_snapshot(data):
    return {key: value.clone() for key, value in data.batch.items()}


def _assert_all_tensors_equal(data, before):
    assert set(data.batch.keys()) == set(before)
    for key, expected in before.items():
        assert torch.equal(data.batch[key], expected), key


def _assert_candidate_accounting(metrics):
    assert metrics["fa_cac/candidate_count"] == (
        metrics["fa_cac/eligible_count"]
        + metrics["fa_cac/excluded_pfa_zero_count"]
        + metrics["fa_cac/excluded_nonnegative_vanilla_adv_count"]
        + metrics["fa_cac/excluded_nonnegative_task_adv_count"]
    )


def _evidence(*, pfa=0.75, eligible_parent=0, task=(-1.0, 1.0, -1.0, 1.0)):
    hit = np.zeros(4, dtype=bool)
    attempted = np.zeros(4, dtype=bool)
    hit[eligible_parent] = True
    attempted[eligible_parent] = True
    return ForcedAnswerCensorEvidence(
        current_row_to_parent=np.arange(4, dtype=np.int64),
        hit_response_cap=hit,
        probe_attempted=attempted,
        context_overflow=np.zeros(4, dtype=bool),
        original_correctness_by_parent=np.asarray([0.0, 1.0, 0.0, 1.0]),
        task_score_by_parent=np.asarray(task, dtype=np.float64),
        pfa_by_parent={eligible_parent: pfa},
        correctness_threshold=0.5,
    )


def _algorithm(*, enable=True, apply=True, norm=True):
    return SimpleNamespace(
        adv_estimator="grpo",
        norm_adv_by_std_in_grpo=norm,
        censor_aware_advantage=CensorAwareAdvantageConfig(enable=enable, apply=apply),
    )


def test_disabled_hook_is_object_and_tensor_identity():
    data = _batch()
    advantages = data.batch["advantages"]
    returns = data.batch["returns"]
    result, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=None, algorithm_config=_algorithm(enable=False)
    )
    assert result is data
    assert result.batch["advantages"] is advantages
    assert result.batch["returns"] is returns
    assert metrics["fa_cac/applied"] == 0.0


def test_shadow_mode_computes_projection_but_preserves_actor_tensors():
    data = _batch()
    advantages = data.batch["advantages"].clone()
    returns = data.batch["returns"].clone()
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(), algorithm_config=_algorithm(apply=False)
    )
    assert torch.equal(data.batch["advantages"], advantages)
    assert torch.equal(data.batch["returns"], returns)
    assert metrics["fa_cac/shadow"] == 1.0
    assert metrics["fa_cac/projected_changed_trajectory_count"] == 1.0
    assert metrics["fa_cac/actor_visible_advantage_drift_max"] == 0.0


def test_shadow_and_apply_share_all_counterfactual_projection_metrics():
    shadow = _batch()
    applied = _batch()
    _, shadow_metrics = apply_fa_cac_post_advantage_hook(
        shadow, evidence=_evidence(), algorithm_config=_algorithm(apply=False)
    )
    _, applied_metrics = apply_fa_cac_post_advantage_hook(
        applied, evidence=_evidence(), algorithm_config=_algorithm(apply=True)
    )
    actor_only = {
        "fa_cac/applied",
        "fa_cac/shadow",
        "fa_cac/actor_visible_advantage_drift_max",
    }
    for key in shadow_metrics.keys() - actor_only:
        assert shadow_metrics[key] == applied_metrics[key], key
    assert shadow_metrics["fa_cac/actor_visible_advantage_drift_max"] == 0.0
    assert applied_metrics["fa_cac/actor_visible_advantage_drift_max"] > 0.0


def test_batch_wide_projected_diagnostics_and_safety_counts():
    data = _batch()
    before = data.batch["advantages"][:, 0].double().numpy()
    token_counts = data.batch["response_mask"].sum(-1).double().numpy()
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(), algorithm_config=_algorithm(apply=False)
    )
    assert metrics["fa_cac/batch_before_adv_mean"] == pytest.approx(before.mean())
    assert metrics["fa_cac/batch_before_adv_abs_mean"] == pytest.approx(np.abs(before).mean())
    assert metrics["fa_cac/batch_before_adv_rms"] == pytest.approx(np.sqrt(np.mean(before**2)))
    assert metrics["fa_cac/batch_before_adv_token_weighted_sum"] == pytest.approx(
        np.sum(before * token_counts)
    )
    assert metrics["fa_cac/batch_after_adv_mean"] != metrics["fa_cac/batch_before_adv_mean"]
    assert metrics["fa_cac/raw_correct_changed_count"] == 0.0
    assert metrics["fa_cac/incorrect_became_positive_count"] == 0.0


def test_real_grpo_integration_changes_only_eligible_valid_tokens_and_not_rewards():
    data = _batch()
    before = _tensor_snapshot(data)
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(), algorithm_config=_algorithm()
    )
    assert not torch.equal(data.batch["advantages"][0, :3], before["advantages"][0, :3])
    assert torch.equal(data.batch["advantages"][0, 3:], before["advantages"][0, 3:])
    assert torch.equal(data.batch["advantages"][1:], before["advantages"][1:])
    assert torch.equal(data.batch["returns"][0, 3:], before["returns"][0, 3:])
    assert torch.equal(data.batch["returns"][1:], before["returns"][1:])
    for key in before.keys() - {"advantages", "returns"}:
        assert torch.equal(data.batch[key], before[key]), key
    assert torch.equal(data.batch["returns"], data.batch["advantages"])
    assert metrics["fa_cac/reconstruction_error_max"] <= 1e-6
    assert metrics["fa_cac/non_target_advantage_drift_max"] == 0.0
    assert metrics["fa_cac/padding_advantage_drift_max"] == 0.0
    assert metrics["fa_cac/reward_drift_max"] == 0.0


def test_pre_clamp_keeps_full_residual_and_uses_total_denominator():
    data = _batch()
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(pfa=0.75), algorithm_config=_algorithm()
    )
    assert metrics["fa_cac/pre_adv_mean"] == pytest.approx(
        metrics["fa_cac/reg_adv_mean"] + 0.25 * metrics["fa_cac/task_adv_mean"], abs=1e-6
    )


@pytest.mark.parametrize(
    ("response_length", "expected_residual"),
    [(1200, 0.0), (1638, 0.0), (1639, -1.0 / 410.0), (2048, -1.0)],
)
def test_formal_dapo_overlong_residual_definition(response_length, expected_residual):
    task_score = -1.0
    total_reward = task_score + min(-(response_length - 1638) / 410.0, 0.0)
    assert total_reward - task_score == pytest.approx(expected_residual)


def test_positive_pre_advantage_is_conservatively_sign_projected():
    data = _batch(total_scores=(-0.1, 0.1, -1.0, 1.0))
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(pfa=0.75), algorithm_config=_algorithm()
    )
    assert metrics["fa_cac/vanilla_adv_mean"] < 0.0
    assert metrics["fa_cac/task_adv_mean"] < 0.0
    assert metrics["fa_cac/pre_adv_mean"] > 0.0
    assert torch.all(data.batch["advantages"][0, :3] == 0)
    assert metrics["fa_cac/sign_clamp_count"] == 1.0
    assert metrics["fa_cac/sign_clamp_rate"] == 1.0
    assert metrics["fa_cac/sign_clamp_magnitude_mean"] > 0.0


def test_no_sign_projection_reports_zero_magnitude():
    data = _batch()
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(), algorithm_config=_algorithm()
    )
    assert metrics["fa_cac/sign_clamp_count"] == 0.0
    assert metrics["fa_cac/sign_clamp_magnitude_mean"] == 0.0


@pytest.mark.parametrize("pfa", [0.25, 0.5, 0.75, 1.0])
def test_continuous_pfa_mechanism_and_strata(pfa):
    data = _batch()
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(pfa=pfa), algorithm_config=_algorithm()
    )
    assert metrics["fa_cac/pfa_mean"] == pfa
    stratum_count = sum(
        value for key, value in metrics.items() if key.startswith("fa_cac/pfa_") and key.endswith("_count")
    )
    assert stratum_count == 1.0


def test_pfa_zero_is_valid_evidence_but_exact_noop():
    data = _batch()
    before = _tensor_snapshot(data)
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(pfa=0.0), algorithm_config=_algorithm()
    )
    _assert_all_tensors_equal(data, before)
    assert metrics["fa_cac/candidate_count"] == 1.0
    assert metrics["fa_cac/eligible_count"] == 0.0
    assert metrics["fa_cac/excluded_pfa_zero_count"] == 1.0
    _assert_candidate_accounting(metrics)


def test_nonnegative_vanilla_advantage_is_exact_noop():
    data = _batch(total_scores=(0.0, -1.0, -1.0, 1.0))
    before = _tensor_snapshot(data)
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(pfa=0.75), algorithm_config=_algorithm()
    )
    _assert_all_tensors_equal(data, before)
    assert metrics["fa_cac/excluded_nonnegative_vanilla_adv_count"] == 1.0
    assert metrics["fa_cac/eligible_count"] == 0.0
    _assert_candidate_accounting(metrics)


def test_nonnegative_task_advantage_is_exact_noop():
    data = _batch()
    before = _tensor_snapshot(data)
    evidence = _evidence(pfa=0.75, task=(1.0, -1.0, -1.0, 1.0))
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=evidence, algorithm_config=_algorithm()
    )
    _assert_all_tensors_equal(data, before)
    assert metrics["fa_cac/excluded_nonnegative_task_adv_count"] == 1.0
    assert metrics["fa_cac/eligible_count"] == 0.0
    _assert_candidate_accounting(metrics)


def test_exclusion_counts_use_stable_first_failure_attribution():
    data = _batch(total_scores=(0.0, -1.0, -1.0, 1.0))
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(pfa=0.0), algorithm_config=_algorithm()
    )
    assert metrics["fa_cac/excluded_pfa_zero_count"] == 1.0
    assert metrics["fa_cac/excluded_nonnegative_vanilla_adv_count"] == 0.0
    assert metrics["fa_cac/excluded_nonnegative_task_adv_count"] == 0.0
    _assert_candidate_accounting(metrics)


@pytest.mark.parametrize("pfa", [np.nan, -0.1, 1.1])
def test_candidate_invalid_pfa_fails_closed(pfa):
    with pytest.raises(RuntimeError, match="pFA must be in"):
        apply_fa_cac_post_advantage_hook(
            _batch(), evidence=_evidence(pfa=pfa), algorithm_config=_algorithm()
        )


def test_candidate_missing_pfa_fails_closed():
    evidence = _evidence()
    evidence.pfa_by_parent.clear()
    with pytest.raises(RuntimeError, match="candidate.*has no pFA"):
        apply_fa_cac_post_advantage_hook(
            _batch(), evidence=evidence, algorithm_config=_algorithm()
        )


def test_zero_valid_response_tokens_fail_closed():
    data = _batch()
    data.batch["response_mask"][0].zero_()
    with pytest.raises(RuntimeError, match="zero-token rows"):
        apply_fa_cac_post_advantage_hook(
            data, evidence=_evidence(), algorithm_config=_algorithm()
        )


def test_runtime_reward_invariant_detects_actual_mutation(monkeypatch):
    data = _batch()
    original = core_algos.compute_grpo_group_statistics
    mutated = False

    def mutate_reward_once(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            data.batch["token_level_rewards"][0, 0] += 1.0
            mutated = True
        return original(*args, **kwargs)

    monkeypatch.setattr(core_algos, "compute_grpo_group_statistics", mutate_reward_once)
    with pytest.raises(RuntimeError, match="changed token_level_rewards"):
        apply_fa_cac_post_advantage_hook(
            data, evidence=_evidence(), algorithm_config=_algorithm()
        )


def test_dr_grpo_decomposition_and_singleton_semantics():
    data = _batch(norm=False)
    _, metrics = apply_fa_cac_post_advantage_hook(
        data, evidence=_evidence(), algorithm_config=_algorithm(norm=False)
    )
    assert metrics["fa_cac/reconstruction_error_max"] == 0.0

    singleton_rewards = torch.tensor([[0.0, -2.0]])
    singleton_mask = torch.ones_like(singleton_rewards)
    adv, _ = compute_grpo_outcome_advantage(
        singleton_rewards, singleton_mask, np.asarray(["only"], dtype=object)
    )
    expected, _ = _frozen_pre_refactor_grpo(
        singleton_rewards, singleton_mask, np.asarray(["only"], dtype=object), norm=True
    )
    assert torch.equal(adv, expected)


def test_missing_evidence_fails_closed():
    with pytest.raises(RuntimeError, match="evidence is absent"):
        apply_fa_cac_post_advantage_hook(
            _batch(), evidence=None, algorithm_config=_algorithm()
        )


@pytest.mark.parametrize("task", [[-1.0, np.nan], [-1.0]])
def test_task_score_absent_nonfinite_or_misaligned_fails_closed(task):
    with pytest.raises(RuntimeError, match="score|align"):
        build_fa_cac_evidence(
            current_row_to_parent=[0, 1],
            hit_response_cap=[True, False],
            probe_attempted=[True, False],
            context_overflow=[False, False],
            original_correctness_by_parent=[0.0, 1.0],
            task_score_in_current_row_order=task,
            successes_by_parent={0: [True]},
            correctness_threshold=0.5,
        )


def test_duplicate_or_invalid_parent_identity_fails_closed():
    with pytest.raises(RuntimeError, match="unique parent"):
        build_fa_cac_evidence(
            current_row_to_parent=[0, 0],
            hit_response_cap=[True, False],
            probe_attempted=[True, False],
            context_overflow=[False, False],
            original_correctness_by_parent=[0.0, 1.0],
            task_score_in_current_row_order=[-1.0, 1.0],
            successes_by_parent={0: [True]},
            correctness_threshold=0.5,
        )
