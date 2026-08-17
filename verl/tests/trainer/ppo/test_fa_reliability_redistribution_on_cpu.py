from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.config import CensorAwareAdvantageConfig
from verl.trainer.ppo.censor_aware_advantage import (
    apply_fa_cac_post_advantage_hook,
    compute_fa_reliability_redistributed_advantage,
)
from verl.trainer.ppo.forced_answer_probe import (
    ForcedAnswerReliabilityEvidence,
    build_fa_reliability_evidence,
)


def _pure(
    advantages,
    p_fa,
    lengths,
    groups,
    eligible,
    *,
    dtype=torch.float32,
):
    max_length = max(lengths)
    mask = torch.arange(max_length).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    return compute_fa_reliability_redistributed_advantage(
        torch.tensor(advantages, dtype=dtype),
        torch.tensor(p_fa, dtype=dtype),
        mask,
        np.asarray(groups, dtype=object),
        torch.tensor(eligible, dtype=torch.bool),
    )


def test_no_fa_signal_is_bitwise_noop():
    vanilla = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0]])
    projected, metrics = compute_fa_reliability_redistributed_advantage(
        vanilla,
        torch.zeros_like(vanilla),
        mask,
        np.asarray(["q", "q", "q", "q"], dtype=object),
        torch.zeros(4, dtype=torch.bool),
    )
    assert torch.equal(projected, vanilla)
    assert metrics["fa_rar/negative_net_correction_token_weighted_sum"] == 0.0
    assert metrics["fa_rar/baseline_mean"] == 0.0


def test_equal_length_equal_advantage_redistributes_and_conserves():
    projected, metrics = _pure(
        [-0.8, -0.8, -0.8, -0.8],
        [0.0, 0.0, 0.0, 0.5],
        [100, 100, 100, 100],
        ["q", "q", "q", "q"],
        [False, False, False, True],
    )
    assert torch.allclose(projected, torch.tensor([-0.9, -0.9, -0.9, -0.5]))
    assert metrics["fa_rar/centering_baseline_mean"] == pytest.approx(0.1)
    assert metrics["fa_rar/token_weighted_advantage_before"] == pytest.approx(-320.0)
    assert metrics["fa_rar/token_weighted_advantage_after"] == pytest.approx(-320.0)
    required_metrics = {
        "fa_rar/eligible_traj_count",
        "fa_rar/eligible_rate",
        "fa_rar/negative_traj_count",
        "fa_rar/negative_token_count",
        "fa_rar/pfa_mean",
        "fa_rar/pfa_max",
        "fa_rar/raw_correction_mean",
        "fa_rar/raw_correction_max",
        "fa_rar/raw_correction_token_mass",
        "fa_rar/centering_baseline_mean",
        "fa_rar/centering_baseline_max",
        "fa_rar/eligible_advantage_before_mean",
        "fa_rar/eligible_advantage_after_mean",
        "fa_rar/noneligible_negative_before_mean",
        "fa_rar/noneligible_negative_after_mean",
        "fa_rar/token_weighted_advantage_before",
        "fa_rar/token_weighted_advantage_after",
        "fa_rar/net_token_weighted_correction",
        "fa_rar/conservation_error_max",
        "fa_rar/sign_flip_count",
    }
    assert required_metrics <= metrics.keys()


def test_different_lengths_and_advantage_magnitudes_use_length_weighted_baseline():
    projected, metrics = _pure(
        [-2.0, -1.0, 1.0],
        [0.5, 0.0, 0.0],
        [4, 2, 3],
        ["q", "q", "q"],
        [True, False, False],
        dtype=torch.float64,
    )
    assert torch.allclose(projected, torch.tensor([-5 / 3, -5 / 3, 1.0], dtype=torch.float64))
    assert metrics["fa_rar/baseline_mean"] == pytest.approx(2 / 3)
    assert metrics["fa_rar/nonnegative_drift_max"] == 0.0
    assert metrics["fa_rar/conservation_error_group_max"] <= metrics[
        "fa_rar/conservation_rounding_bound_group_max"
    ]


def test_multiple_eligible_rows_multiple_uid_groups_and_nonnegative_invariance():
    vanilla = [-2.0, -1.0, 0.0, 3.0, -4.0, -0.5, 2.0]
    projected, metrics = _pure(
        vanilla,
        [0.25, 1.0, 0.0, 0.0, 0.75, 0.0, 0.0],
        [2, 4, 1, 3, 5, 2, 1],
        ["a", "a", "a", "a", "b", "b", "b"],
        [True, True, False, False, True, False, False],
        dtype=torch.float64,
    )
    projected = projected.numpy()
    vanilla_array = np.asarray(vanilla)
    assert np.array_equal(projected[vanilla_array >= 0], vanilla_array[vanilla_array >= 0])
    assert np.all(projected[vanilla_array < 0] <= 0)
    assert metrics["fa_rar/group_with_negative_count"] == 2.0
    assert metrics["fa_rar/eligible_trajectory_count"] == 3.0
    assert metrics["fa_rar/conservation_error_group_max"] <= metrics[
        "fa_rar/conservation_rounding_bound_group_max"
    ]


def test_group_without_negative_has_zero_baseline_and_exact_noop():
    projected, metrics = _pure(
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0],
        [1, 2, 3],
        ["q", "q", "q"],
        [False, False, False],
    )
    assert torch.equal(projected, torch.tensor([0.0, 1.0, 2.0]))
    assert metrics["fa_rar/group_with_negative_count"] == 0.0
    assert metrics["fa_rar/group_without_negative_count"] == 1.0
    assert metrics["fa_rar/baseline_group_count"] == 0.0


def test_2048_token_candidate_against_multiple_500_token_negatives():
    projected, metrics = _pure(
        [-1.0] * 5,
        [0.75, 0.0, 0.0, 0.0, 0.0],
        [2048, 500, 500, 500, 500],
        ["q"] * 5,
        [True, False, False, False, False],
        dtype=torch.float64,
    )
    baseline = (2048 * 0.75) / (2048 + 4 * 500)
    assert projected[0].item() == pytest.approx(-1 + 0.75 - baseline)
    assert torch.allclose(projected[1:], torch.full((4,), -1 - baseline, dtype=torch.float64))
    assert metrics["fa_rar/negative_net_correction_token_weighted_sum"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_final_dtype_conservation_gate_and_sign_theorem(dtype):
    projected, metrics = _pure(
        [-2.5, -0.75, -1.25, 0.0, 1.0],
        [0.875, 0.5, 0.0, 0.0, 0.0],
        [7, 3, 11, 2, 5],
        [0, 0, 0, 0, 0],
        [True, True, False, False, False],
        dtype=dtype,
    )
    assert torch.all(projected[:3] <= 0)
    assert torch.equal(projected[3:], torch.tensor([0.0, 1.0], dtype=dtype))
    assert metrics["fa_rar/conservation_error_group_max"] <= metrics[
        "fa_rar/conservation_rounding_bound_group_max"
    ]


@pytest.mark.parametrize(
    ("advantages", "pfa", "eligible", "message"),
    [
        ([-1.0, float("nan")], [0.0, 0.0], [False, False], "must be finite"),
        ([-1.0, -2.0], [1.1, 0.0], [True, False], "must be in"),
        ([-1.0, -2.0], [0.5, 0.25], [True, False], "non-eligible"),
        ([1.0, -2.0], [0.5, 0.0], [True, False], "only for negative"),
    ],
)
def test_pure_function_invalid_inputs_fail_closed(advantages, pfa, eligible, message):
    with pytest.raises(RuntimeError, match=message):
        _pure(advantages, pfa, [2, 2], ["q", "q"], eligible)


def test_zero_valid_tokens_and_group_alignment_fail_closed():
    with pytest.raises(RuntimeError, match="zero-token"):
        compute_fa_reliability_redistributed_advantage(
            torch.tensor([-1.0]),
            torch.tensor([0.0]),
            torch.zeros((1, 2)),
            ["q"],
            torch.tensor([False]),
        )
    with pytest.raises(RuntimeError, match="group identity"):
        _pure([-1.0, -2.0], [0.0, 0.0], [1, 1], ["q"], [False, False])


def _hook_batch():
    scalars = torch.tensor([-2.0, 1.0, -1.0, 0.0, -0.5, -1.5])
    lengths = torch.tensor([4, 2, 3, 1, 5, 2])
    mask = (torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)).float()
    advantages = scalars.unsqueeze(-1) * mask
    rewards = torch.zeros_like(advantages)
    rewards[:, 0] = scalars
    return DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": rewards,
                "token_level_scores": rewards.clone(),
                "response_mask": mask,
                "advantages": advantages,
                "returns": advantages.clone(),
                "untouched_tensor": torch.arange(30).reshape(6, 5),
            },
            batch_size=6,
        ),
        non_tensor_batch={"uid": np.asarray(["a", "a", "a", "a", "b", "b"], dtype=object)},
    )


def _hook_evidence(*, pfa0=0.5, pfa4=1.0):
    return ForcedAnswerReliabilityEvidence(
        current_row_to_parent=np.arange(6, dtype=np.int64),
        hit_response_cap=np.asarray([True, False, False, False, True, False]),
        probe_attempted=np.asarray([True, False, False, False, True, False]),
        context_overflow=np.zeros(6, dtype=bool),
        original_correctness_by_parent=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
        pfa_by_parent={0: pfa0, 4: pfa4},
        correctness_threshold=0.5,
    )


def _algorithm(*, apply=True):
    return SimpleNamespace(
        adv_estimator="grpo",
        censor_aware_advantage=CensorAwareAdvantageConfig(
            enable=True,
            apply=apply,
            mode="reliability_redistribution",
        ),
    )


def test_hook_updates_only_valid_advantages_and_returns_and_guards_every_other_tensor():
    data = _hook_batch()
    before = {key: value.clone() for key, value in data.batch.items()}
    _, metrics = apply_fa_cac_post_advantage_hook(
        data,
        evidence=_hook_evidence(),
        algorithm_config=_algorithm(),
    )
    assert torch.equal(data.batch["returns"], data.batch["advantages"])
    assert not torch.equal(data.batch["advantages"], before["advantages"])
    for key in before.keys() - {"advantages", "returns"}:
        assert torch.equal(data.batch[key], before[key]), key
    padding = ~data.batch["response_mask"].bool()
    assert torch.equal(data.batch["advantages"][padding], before["advantages"][padding])
    assert torch.equal(data.batch["advantages"][1], before["advantages"][1])
    assert torch.equal(data.batch["advantages"][3], before["advantages"][3])
    assert metrics["fa_rar/reward_drift_max"] == 0.0
    assert metrics["fa_rar/score_drift_max"] == 0.0
    assert metrics["fa_rar/positive_advantage_drift_max"] == 0.0
    assert metrics["fa_rar/padding_advantage_drift_max"] == 0.0
    assert metrics["fa_rar/sign_flip_count"] == 0.0


def test_shadow_mode_computes_identical_counterfactual_metrics_without_actor_drift():
    shadow = _hook_batch()
    applied = _hook_batch()
    shadow_before = {key: value.clone() for key, value in shadow.batch.items()}
    _, shadow_metrics = apply_fa_cac_post_advantage_hook(
        shadow,
        evidence=_hook_evidence(),
        algorithm_config=_algorithm(apply=False),
    )
    _, applied_metrics = apply_fa_cac_post_advantage_hook(
        applied,
        evidence=_hook_evidence(),
        algorithm_config=_algorithm(apply=True),
    )
    for key, expected in shadow_before.items():
        assert torch.equal(shadow.batch[key], expected), key
    actor_only = {"fa_rar/applied", "fa_rar/shadow", "fa_rar/actor_visible_advantage_drift_max"}
    for key in shadow_metrics.keys() - actor_only:
        assert shadow_metrics[key] == applied_metrics[key], key
    assert shadow_metrics["fa_rar/actor_visible_advantage_drift_max"] == 0.0
    assert applied_metrics["fa_rar/actor_visible_advantage_drift_max"] > 0.0


def test_hook_missing_candidate_pfa_invalid_identity_and_nonconstant_advantage_fail_closed():
    evidence = _hook_evidence()
    evidence.pfa_by_parent.pop(0)
    with pytest.raises(RuntimeError, match="candidate.*has no pFA"):
        apply_fa_cac_post_advantage_hook(
            _hook_batch(), evidence=evidence, algorithm_config=_algorithm()
        )
    evidence = _hook_evidence()
    evidence.current_row_to_parent[1] = 0
    with pytest.raises(RuntimeError, match="unique parent"):
        apply_fa_cac_post_advantage_hook(
            _hook_batch(), evidence=evidence, algorithm_config=_algorithm()
        )
    data = _hook_batch()
    data.batch["advantages"][0, 1] += 0.25
    data.batch["returns"].copy_(data.batch["advantages"])
    with pytest.raises(RuntimeError, match="non-constant"):
        apply_fa_cac_post_advantage_hook(
            data, evidence=_hook_evidence(), algorithm_config=_algorithm()
        )


def test_reliability_evidence_builder_is_task_score_free_and_fail_closed():
    evidence = build_fa_reliability_evidence(
        current_row_to_parent=[1, 0],
        hit_response_cap=[True, True],
        probe_attempted=[True, True],
        context_overflow=[False, False],
        original_correctness_by_parent=[0.0, 0.0],
        successes_by_parent={0: [True, False], 1: [True, True]},
        correctness_threshold=0.5,
    )
    assert not hasattr(evidence, "task_score_by_parent")
    assert evidence.pfa_by_parent == {0: 0.5, 1: 1.0}
    with pytest.raises(RuntimeError, match="has no pFA"):
        build_fa_reliability_evidence(
            current_row_to_parent=[0],
            hit_response_cap=[True],
            probe_attempted=[True],
            context_overflow=[False],
            original_correctness_by_parent=[0.0],
            successes_by_parent={},
            correctness_threshold=0.5,
        )


def test_rar_mode_is_supported_without_changing_v2_default():
    default = CensorAwareAdvantageConfig()
    assert default.mode == "attenuate_negative_correctness"
    default.validate()
    CensorAwareAdvantageConfig(mode="reliability_redistribution").validate()


def test_randomized_10000_groups_conserve_mass_preserve_sign_and_remain_finite():
    group_count = 10_000
    rollout_n = 8
    row_count = group_count * rollout_n
    generator = torch.Generator().manual_seed(20260817)
    negative_magnitude = 0.01 + 3.0 * torch.rand(row_count, generator=generator)
    vanilla = -negative_magnitude
    positions = torch.arange(row_count) % rollout_n
    vanilla[positions == 6] = 0.0
    vanilla[positions == 7] = 0.25 + torch.rand(group_count, generator=generator)
    eligible = (positions < 6) & (torch.rand(row_count, generator=generator) < 0.45)
    p_fa = torch.zeros(row_count)
    p_fa[eligible] = 0.01 + 0.99 * torch.rand(int(eligible.sum().item()), generator=generator)
    lengths = torch.randint(1, 33, (row_count,), generator=generator)
    response_mask = torch.arange(32).unsqueeze(0) < lengths.unsqueeze(1)
    groups = np.repeat(np.arange(group_count), rollout_n)

    projected, metrics = compute_fa_reliability_redistributed_advantage(
        vanilla,
        p_fa,
        response_mask,
        groups,
        eligible,
    )
    negative = vanilla < 0
    nonnegative = ~negative
    before_mass = (vanilla.double() * lengths.double()).reshape(group_count, rollout_n).sum(-1)
    after_mass = (projected.double() * lengths.double()).reshape(group_count, rollout_n).sum(-1)
    max_error = torch.max(torch.abs(after_mass - before_mass)).item()
    print(f"FA-RAR randomized max conservation error: {max_error:.12g}")
    assert max_error <= metrics["fa_rar/conservation_rounding_bound_group_max"]
    assert torch.all(projected[negative] <= 0)
    assert torch.equal(projected[nonnegative], vanilla[nonnegative])
    assert torch.all(torch.isfinite(projected))
    assert metrics["fa_rar/group_count"] == float(group_count)
    assert metrics["fa_rar/group_with_negative_count"] == float(group_count)
