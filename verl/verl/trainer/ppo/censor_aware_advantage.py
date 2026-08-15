"""FA-CAC v2 post-GRPO advantage projection.

The hook never changes rewards. It decomposes canonical Vanilla GRPO using
the exact task score emitted by the reward manager, applies the censor-aware
mechanism only to eligible capped failures, and optionally exposes the
projected tensors to the actor. The final ``min(0, A_pre)`` is a conservative
sign projection; when it activates, exact residual preservation is not
claimed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.forced_answer_probe import ForcedAnswerCensorEvidence


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _quantile_metrics(prefix: str, values: np.ndarray) -> dict[str, float]:
    quantiles = (("q00", 0.0), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q100", 1.0))
    if values.size == 0:
        return {f"{prefix}_{name}": 0.0 for name, _ in quantiles}
    return {f"{prefix}_{name}": float(np.quantile(values, q)) for name, q in quantiles}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or float(weights.sum()) == 0.0:
        return 0.0
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = q * float(sorted_weights.sum())
    position = min(int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")), len(values) - 1)
    return float(sorted_values[position])


def _token_weighted_quantile_metrics(
    prefix: str, values: np.ndarray, token_counts: np.ndarray
) -> dict[str, float]:
    return {
        f"{prefix}_q00": _weighted_quantile(values, token_counts, 0.0),
        f"{prefix}_q25": _weighted_quantile(values, token_counts, 0.25),
        f"{prefix}_q50": _weighted_quantile(values, token_counts, 0.5),
        f"{prefix}_q75": _weighted_quantile(values, token_counts, 0.75),
        f"{prefix}_q100": _weighted_quantile(values, token_counts, 1.0),
    }


def apply_fa_cac_post_advantage_hook(
    data: DataProto,
    *,
    evidence: ForcedAnswerCensorEvidence | None,
    algorithm_config: Any,
) -> tuple[DataProto, dict[str, float]]:
    """Apply or shadow FA-CAC after canonical Vanilla ``compute_advantage``."""
    cac_config = _get(algorithm_config, "censor_aware_advantage", None)
    enabled = bool(_get(cac_config, "enable", False))
    apply = bool(_get(cac_config, "apply", True))
    metrics: dict[str, float] = {
        "fa_cac/enabled": float(enabled),
        "fa_cac/applied": float(enabled and apply),
        "fa_cac/shadow": float(enabled and not apply),
    }
    if not enabled:
        return data, metrics
    if _get(cac_config, "mode", None) != "attenuate_negative_correctness":
        raise RuntimeError("unsupported FA-CAC mode reached post-advantage hook")
    raw_adv_estimator = _get(algorithm_config, "adv_estimator", None)
    adv_estimator = getattr(raw_adv_estimator, "value", raw_adv_estimator)
    if adv_estimator != "grpo":
        raise RuntimeError("FA-CAC post-advantage hook requires canonical Vanilla GRPO")
    if evidence is None:
        raise RuntimeError("FA-CAC enabled but parent-keyed forced-answer evidence is absent")

    required = {"token_level_rewards", "response_mask", "advantages", "returns"}
    missing = required - set(data.batch.keys())
    if missing:
        raise RuntimeError(f"FA-CAC batch is missing required tensors: {sorted(missing)}")
    rewards = data.batch["token_level_rewards"]
    rewards_before = rewards.detach().clone()
    scores_before = (
        data.batch["token_level_scores"].detach().clone()
        if "token_level_scores" in data.batch
        else None
    )
    response_mask = data.batch["response_mask"]
    vanilla_advantages = data.batch["advantages"]
    vanilla_returns = data.batch["returns"]
    if rewards.ndim != 2 or response_mask.shape != rewards.shape:
        raise RuntimeError("FA-CAC reward and response mask tensors must be aligned rank-2 tensors")
    if vanilla_advantages.shape != rewards.shape or vanilla_returns.shape != rewards.shape:
        raise RuntimeError("FA-CAC Vanilla actor tensors must align with rewards")
    response_token_counts = response_mask.sum(dim=-1)
    if torch.any(response_token_counts <= 0):
        bad_rows = torch.nonzero(response_token_counts <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "FA-CAC retained PPO rows must each contain at least one valid response token; "
            f"zero-token rows={bad_rows}"
        )
    parents = np.asarray(evidence.current_row_to_parent, dtype=np.int64)
    uids = data.non_tensor_batch.get("uid")
    if uids is None or len(uids) != len(rewards) or len(parents) != len(rewards):
        raise RuntimeError("FA-CAC UID and row identity must align with the retained PPO batch")

    task_np = evidence.task_score_by_parent[parents]
    if not np.all(np.isfinite(task_np)):
        raise RuntimeError("FA-CAC exact task score is absent or non-finite for a retained row")
    task_scores = torch.as_tensor(task_np, dtype=rewards.dtype, device=rewards.device)
    total_scores = rewards.sum(dim=-1)
    residual_scores = total_scores - task_scores
    norm_by_std = bool(_get(algorithm_config, "norm_adv_by_std_in_grpo", True))

    with torch.no_grad():
        total_stats = core_algos.compute_grpo_group_statistics(total_scores, uids)
        task_stats = core_algos.compute_grpo_group_statistics(task_scores, uids)
        residual_stats = core_algos.compute_grpo_group_statistics(residual_scores, uids)
        vanilla_scalar = core_algos.normalize_grpo_scores(
            total_scores,
            uids,
            numerator_statistics=total_stats,
            norm_adv_by_std_in_grpo=norm_by_std,
        )
        task_advantage = core_algos.normalize_grpo_scores(
            task_scores,
            uids,
            numerator_statistics=task_stats,
            denominator_statistics=total_stats,
            norm_adv_by_std_in_grpo=norm_by_std,
        )
        residual_advantage = core_algos.normalize_grpo_scores(
            residual_scores,
            uids,
            numerator_statistics=residual_stats,
            denominator_statistics=total_stats,
            norm_adv_by_std_in_grpo=norm_by_std,
        )
        reconstruction_error = torch.max(torch.abs(vanilla_scalar - task_advantage - residual_advantage)).item()
        if reconstruction_error > 1e-6:
            raise RuntimeError(
                "FA-CAC GRPO decomposition exceeded tolerance: "
                f"max|A_vanilla-A_task-A_reg|={reconstruction_error}"
            )

        valid_mask = response_mask.to(dtype=torch.bool)
        expanded_vanilla = vanilla_scalar.unsqueeze(-1) * response_mask
        vanilla_consistency_error = torch.max(torch.abs(expanded_vanilla - vanilla_advantages)).item()
        if vanilla_consistency_error > 1e-6:
            raise RuntimeError(
                "FA-CAC was not invoked immediately after canonical Vanilla GRPO: "
                f"max drift={vanilla_consistency_error}"
            )
        returns_consistency_error = torch.max(torch.abs(vanilla_returns - vanilla_advantages)).item()
        if returns_consistency_error > 1e-6:
            raise RuntimeError(
                "FA-CAC canonical GRPO returns do not match advantages: "
                f"max drift={returns_consistency_error}"
            )

        candidate_rows: list[int] = []
        eligible_rows: list[int] = []
        pfas: list[float] = []
        excluded_pfa_zero_count = 0
        excluded_nonnegative_vanilla_adv_count = 0
        excluded_nonnegative_task_adv_count = 0
        projected = vanilla_advantages.clone()
        projected_returns = vanilla_returns.clone()
        pre_by_row = vanilla_scalar.clone()
        cac_by_row = vanilla_scalar.clone()
        for row, parent in enumerate(parents.tolist()):
            censor_candidate = bool(
                evidence.hit_response_cap[parent]
                and evidence.probe_attempted[parent]
                and not evidence.context_overflow[parent]
                and evidence.original_correctness_by_parent[parent] < evidence.correctness_threshold
            )
            if not censor_candidate:
                continue
            candidate_rows.append(row)
            if parent not in evidence.pfa_by_parent:
                raise RuntimeError(f"FA-CAC censor candidate trajectory {parent} has no pFA")
            pfa = float(evidence.pfa_by_parent[parent])
            if not np.isfinite(pfa) or not 0.0 <= pfa <= 1.0:
                raise RuntimeError(f"FA-CAC trajectory {parent} pFA must be in [0, 1]")
            # Stable first-failure attribution. These exclusions are mutually
            # exclusive so the candidate accounting identity is exact.
            if pfa <= 0.0:
                excluded_pfa_zero_count += 1
                continue
            if vanilla_scalar[row] >= 0.0:
                excluded_nonnegative_vanilla_adv_count += 1
                continue
            if task_advantage[row] >= 0.0:
                excluded_nonnegative_task_adv_count += 1
                continue
            pre = residual_advantage[row] + (1.0 - pfa) * task_advantage[row]
            cac = torch.minimum(torch.zeros_like(pre), pre)
            pre_by_row[row] = pre
            cac_by_row[row] = cac
            projected[row, valid_mask[row]] = cac
            projected_returns[row, valid_mask[row]] = cac
            eligible_rows.append(row)
            pfas.append(pfa)

        if apply:
            data.batch["advantages"] = projected
            data.batch["returns"] = projected_returns

    eligible = np.asarray(eligible_rows, dtype=np.int64)
    pfa_array = np.asarray(pfas, dtype=np.float64)
    vanilla_np = vanilla_scalar.detach().float().cpu().numpy()[eligible]
    task_np_adv = task_advantage.detach().float().cpu().numpy()[eligible]
    residual_np_adv = residual_advantage.detach().float().cpu().numpy()[eligible]
    pre_np = pre_by_row.detach().float().cpu().numpy()[eligible]
    cac_np = cac_by_row.detach().float().cpu().numpy()[eligible]
    drift = cac_np - vanilla_np
    attenuation = pre_np - vanilla_np
    clamp = cac_np - pre_np
    all_token_counts = response_token_counts.detach().double().cpu().numpy()
    token_counts = all_token_counts[eligible]
    clamped = pre_np > 0.0
    candidate_count = len(candidate_rows)
    eligible_count = len(eligible_rows)
    total_rows = len(rewards)
    if candidate_count != (
        eligible_count
        + excluded_pfa_zero_count
        + excluded_nonnegative_vanilla_adv_count
        + excluded_nonnegative_task_adv_count
    ):
        raise RuntimeError("FA-CAC candidate exclusion accounting invariant failed")
    changed = np.not_equal(drift, 0.0)
    projected_padding_drift = (
        torch.max(torch.abs((projected - vanilla_advantages)[~valid_mask])).item()
        if (~valid_mask).any()
        else 0.0
    )
    non_target = np.ones(total_rows, dtype=bool)
    non_target[eligible] = False
    non_target_drift = (
        torch.max(
            torch.abs((projected - vanilla_advantages)[torch.as_tensor(non_target, device=rewards.device)])
        ).item()
        if np.any(non_target)
        else 0.0
    )
    batch_before = vanilla_scalar.detach().double().cpu().numpy()
    batch_after = cac_by_row.detach().double().cpu().numpy()
    raw_correct = evidence.original_correctness_by_parent[parents] >= evidence.correctness_threshold
    raw_incorrect = ~raw_correct
    batch_changed = np.not_equal(batch_after, batch_before)
    raw_correct_changed_count = int(np.count_nonzero(raw_correct & batch_changed))
    incorrect_became_positive_count = int(
        np.count_nonzero(raw_incorrect & (batch_before <= 0.0) & (batch_after > 0.0))
    )
    if raw_correct_changed_count or incorrect_became_positive_count:
        raise RuntimeError(
            "FA-CAC safety invariant failed: raw-correct rows changed or raw-incorrect rows became positive"
        )

    rewards_after = data.batch["token_level_rewards"]
    reward_drift_max = (
        torch.max(torch.abs(rewards_after - rewards_before)).item() if rewards_before.numel() else 0.0
    )
    if not torch.equal(rewards_after, rewards_before):
        raise RuntimeError(f"FA-CAC changed token_level_rewards: max drift={reward_drift_max}")
    score_drift_max = 0.0
    if scores_before is not None:
        scores_after = data.batch.get("token_level_scores")
        if scores_after is None:
            raise RuntimeError("FA-CAC removed token_level_scores")
        score_drift_max = (
            torch.max(torch.abs(scores_after - scores_before)).item() if scores_before.numel() else 0.0
        )
        if not torch.equal(scores_after, scores_before):
            raise RuntimeError(f"FA-CAC changed token_level_scores: max drift={score_drift_max}")

    def _batch_advantage_metrics(stage: str, values: np.ndarray) -> dict[str, float]:
        return {
            f"fa_cac/batch_{stage}_adv_mean": float(values.mean()),
            f"fa_cac/batch_{stage}_adv_abs_mean": float(np.abs(values).mean()),
            f"fa_cac/batch_{stage}_adv_rms": float(np.sqrt(np.mean(np.square(values)))),
            f"fa_cac/batch_{stage}_adv_token_weighted_sum": float(np.sum(values * all_token_counts)),
        }

    metrics.update(
        {
            "fa_cac/ppo_batch_trajectory_count": float(total_rows),
            "fa_cac/candidate_count": float(candidate_count),
            "fa_cac/eligible_count": float(eligible_count),
            "fa_cac/excluded_pfa_zero_count": float(excluded_pfa_zero_count),
            "fa_cac/excluded_nonnegative_vanilla_adv_count": float(
                excluded_nonnegative_vanilla_adv_count
            ),
            "fa_cac/excluded_nonnegative_task_adv_count": float(excluded_nonnegative_task_adv_count),
            "fa_cac/eligible_trajectory_count": float(eligible_count),
            "fa_cac/eligible_trajectory_rate": float(eligible_count / total_rows) if total_rows else 0.0,
            "fa_cac/pfa_mean": float(pfa_array.mean()) if eligible_count else 0.0,
            "fa_cac/vanilla_adv_mean": float(vanilla_np.mean()) if eligible_count else 0.0,
            "fa_cac/task_adv_mean": float(task_np_adv.mean()) if eligible_count else 0.0,
            "fa_cac/reg_adv_mean": float(residual_np_adv.mean()) if eligible_count else 0.0,
            "fa_cac/pre_adv_mean": float(pre_np.mean()) if eligible_count else 0.0,
            "fa_cac/projected_adv_mean": float(cac_np.mean()) if eligible_count else 0.0,
            "fa_cac/sign_clamp_count": float(clamped.sum()),
            "fa_cac/sign_clamp_rate": float(clamped.mean()) if eligible_count else 0.0,
            "fa_cac/sign_clamp_magnitude_mean": float(pre_np[clamped].mean()) if clamped.any() else 0.0,
            "fa_cac/projected_changed_trajectory_count": float(changed.sum()),
            "fa_cac/projected_changed_token_count": float(token_counts[changed].sum()) if eligible_count else 0.0,
            "fa_cac/drift_abs_mean": float(np.abs(drift).mean()) if eligible_count else 0.0,
            "fa_cac/drift_abs_max": float(np.abs(drift).max()) if eligible_count else 0.0,
            "fa_cac/drift_token_weighted_abs_mean": (
                float(np.average(np.abs(drift), weights=token_counts)) if token_counts.sum() else 0.0
            ),
            "fa_cac/mechanism_attenuation_abs_mean": float(np.abs(attenuation).mean()) if eligible_count else 0.0,
            "fa_cac/mechanism_clamp_abs_mean": float(np.abs(clamp).mean()) if eligible_count else 0.0,
            "fa_cac/reconstruction_error_max": float(reconstruction_error),
            "fa_cac/vanilla_consistency_error_max": float(vanilla_consistency_error),
            "fa_cac/returns_consistency_error_max": float(returns_consistency_error),
            "fa_cac/reward_drift_max": float(reward_drift_max),
            "fa_cac/score_drift_max": float(score_drift_max),
            "fa_cac/non_target_advantage_drift_max": float(non_target_drift),
            "fa_cac/padding_advantage_drift_max": float(projected_padding_drift),
            "fa_cac/raw_correct_changed_count": float(raw_correct_changed_count),
            "fa_cac/incorrect_became_positive_count": float(incorrect_became_positive_count),
            "fa_cac/actor_visible_advantage_drift_max": (
                float(np.abs(drift).max()) if enabled and apply and eligible_count else 0.0
            ),
        }
    )
    metrics.update(_batch_advantage_metrics("before", batch_before))
    metrics.update(_batch_advantage_metrics("after", batch_after))
    metrics.update(_quantile_metrics("fa_cac/drift", drift))
    metrics.update(_quantile_metrics("fa_cac/mechanism_attenuation", attenuation))
    metrics.update(_quantile_metrics("fa_cac/mechanism_clamp", clamp))
    metrics.update(_token_weighted_quantile_metrics("fa_cac/drift_token_weighted", drift, token_counts))
    metrics.update(
        _token_weighted_quantile_metrics(
            "fa_cac/mechanism_attenuation_token_weighted", attenuation, token_counts
        )
    )
    metrics.update(
        _token_weighted_quantile_metrics("fa_cac/mechanism_clamp_token_weighted", clamp, token_counts)
    )

    strata = {
        "eq_0": pfa_array == 0.0,
        "0_025": (pfa_array > 0.0) & (pfa_array < 0.25),
        "025_050": (pfa_array >= 0.25) & (pfa_array < 0.50),
        "050_075": (pfa_array >= 0.50) & (pfa_array < 0.75),
        "075_1": (pfa_array >= 0.75) & (pfa_array < 1.0),
        "eq_1": pfa_array == 1.0,
    }
    for name, selector in strata.items():
        metrics[f"fa_cac/pfa_{name}_count"] = float(selector.sum())
        metrics.update(_quantile_metrics(f"fa_cac/pfa_{name}_drift", drift[selector]))
        metrics.update(_quantile_metrics(f"fa_cac/pfa_{name}_attenuation", attenuation[selector]))
        metrics.update(_quantile_metrics(f"fa_cac/pfa_{name}_clamp", clamp[selector]))
    return data, metrics
