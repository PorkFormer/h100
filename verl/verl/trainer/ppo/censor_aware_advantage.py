"""Forced-answer post-GRPO advantage interventions.

The compatibility path retains FA-CAC v2's task/residual decomposition. The
FA-RAR path instead redistributes reliability credit among negative responses
while conserving their token-weighted advantage mass. Neither path changes
rewards, and both can run in shadow mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.forced_answer_probe import (
    ForcedAnswerCensorEvidence,
    ForcedAnswerReliabilityEvidence,
)


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


def _as_group_id_list(group_ids: Any, row_count: int) -> list[Any]:
    if isinstance(group_ids, torch.Tensor):
        ids = group_ids.detach().cpu().tolist()
    elif isinstance(group_ids, np.ndarray):
        ids = group_ids.tolist()
    else:
        ids = list(group_ids)
    if len(ids) != row_count:
        raise RuntimeError("FA-RAR group identity must align with response rows")
    for group_id in ids:
        if group_id is None:
            raise RuntimeError("FA-RAR group identities must be non-null")
        try:
            hash(group_id)
        except TypeError as exc:
            raise RuntimeError("FA-RAR group identities must be hashable") from exc
        if isinstance(group_id, (float, np.floating)) and not np.isfinite(group_id):
            raise RuntimeError("FA-RAR group identities must be finite")
    return ids


def _response_mean(values: torch.Tensor, selector: torch.Tensor) -> float:
    return float(values[selector].double().mean().item()) if torch.any(selector) else 0.0


def compute_fa_reliability_redistributed_advantage(
    vanilla_advantage: torch.Tensor,
    p_fa: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: Any,
    eligible_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Redistribute FA reliability credit within each UID group's negative subset.

    Inputs and output are response-level scalars. ``response_mask`` supplies the
    response lengths used by the conservation law. Non-eligible rows must carry
    zero pFA, making the pure-function contract explicit and fail closed.
    """
    if vanilla_advantage.ndim != 1 or not vanilla_advantage.is_floating_point():
        raise RuntimeError("FA-RAR vanilla advantage must be a rank-1 floating tensor")
    row_count = vanilla_advantage.numel()
    if p_fa.ndim != 1 or p_fa.shape != vanilla_advantage.shape:
        raise RuntimeError("FA-RAR pFA must align with response-level advantages")
    if eligible_mask.ndim != 1 or eligible_mask.shape != vanilla_advantage.shape:
        raise RuntimeError("FA-RAR eligible mask must align with response-level advantages")
    if eligible_mask.dtype != torch.bool:
        raise RuntimeError("FA-RAR eligible mask must be boolean")
    if response_mask.ndim != 2 or response_mask.shape[0] != row_count:
        raise RuntimeError("FA-RAR response mask must be rank-2 and align with response rows")
    if p_fa.device != vanilla_advantage.device or response_mask.device != vanilla_advantage.device:
        raise RuntimeError("FA-RAR tensors must be on the same device")
    if eligible_mask.device != vanilla_advantage.device:
        raise RuntimeError("FA-RAR eligible mask must be on the advantage device")
    if not p_fa.is_floating_point():
        raise RuntimeError("FA-RAR pFA must be floating point")
    if not torch.all(torch.isfinite(vanilla_advantage)):
        raise RuntimeError("FA-RAR vanilla advantage must be finite")
    if not torch.all(torch.isfinite(p_fa)):
        raise RuntimeError("FA-RAR pFA must be finite")
    if torch.any((p_fa < 0) | (p_fa > 1)):
        raise RuntimeError("FA-RAR pFA must be in [0, 1]")
    if response_mask.is_floating_point() and not torch.all(torch.isfinite(response_mask)):
        raise RuntimeError("FA-RAR response mask must be finite")
    mask_bool = response_mask.to(dtype=torch.bool)
    if not torch.equal(response_mask, mask_bool.to(dtype=response_mask.dtype)):
        raise RuntimeError("FA-RAR response mask must contain only zeros and ones")
    lengths = mask_bool.sum(dim=-1)
    if torch.any(lengths <= 0):
        bad_rows = torch.nonzero(lengths <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"FA-RAR response rows must contain valid tokens; zero-token rows={bad_rows}")

    eligible = eligible_mask.to(dtype=torch.bool)
    negative = vanilla_advantage < 0
    if torch.any(eligible & ~negative):
        raise RuntimeError("FA-RAR active eligibility is valid only for negative Vanilla advantages")
    if torch.any((~eligible) & (p_fa != 0)):
        raise RuntimeError("FA-RAR non-eligible rows must carry pFA=0")
    if torch.any(eligible & (p_fa <= 0)):
        raise RuntimeError("FA-RAR active eligible rows must carry pFA>0")

    ids = _as_group_id_list(group_ids, row_count)
    grouped_rows: dict[Any, list[int]] = {}
    for row, group_id in enumerate(ids):
        grouped_rows.setdefault(group_id, []).append(row)

    work = vanilla_advantage.detach().double()
    pfa_work = p_fa.detach().double()
    length_work = lengths.double()
    projected_work = work.clone()
    distortion = torch.zeros_like(work)
    baselines: list[float] = []
    conservation_errors: list[float] = []
    conservation_bounds: list[float] = []
    dtype_info = torch.finfo(vanilla_advantage.dtype)

    for rows in grouped_rows.values():
        row_tensor = torch.as_tensor(rows, dtype=torch.long, device=work.device)
        negative_rows = row_tensor[negative[row_tensor]]
        if negative_rows.numel() == 0:
            continue
        group_distortion = pfa_work[negative_rows] * torch.abs(work[negative_rows])
        distortion[negative_rows] = group_distortion
        denominator = length_work[negative_rows].sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("FA-RAR negative-subset denominator must be finite and positive")
        # Project raw FA reliability corrections onto the
        # token-mass-conserving subspace.
        baseline = torch.sum(length_work[negative_rows] * group_distortion) / denominator
        if not torch.isfinite(baseline):
            raise RuntimeError("FA-RAR group baseline must be finite")
        projected_work[negative_rows] = work[negative_rows] + group_distortion - baseline
        baselines.append(float(baseline.item()))

    projected = projected_work.to(dtype=vanilla_advantage.dtype)
    if not torch.all(torch.isfinite(projected)):
        raise RuntimeError("FA-RAR produced a non-finite advantage")
    if torch.any(projected[negative] > 0):
        raise RuntimeError("FA-RAR negative-sign theorem failed")
    if not torch.equal(projected[~negative], vanilla_advantage[~negative]):
        raise RuntimeError("FA-RAR changed a positive or zero Vanilla advantage")

    for rows in grouped_rows.values():
        row_tensor = torch.as_tensor(rows, dtype=torch.long, device=work.device)
        negative_rows = row_tensor[negative[row_tensor]]
        if negative_rows.numel() == 0:
            continue
        before_mass = torch.sum(length_work[negative_rows] * work[negative_rows])
        after_mass = torch.sum(length_work[negative_rows] * projected[negative_rows].double())
        error = torch.abs(after_mass - before_mass)
        scale = torch.sum(
            length_work[negative_rows]
            * (
                torch.abs(work[negative_rows])
                + torch.abs(projected_work[negative_rows])
                + torch.abs(distortion[negative_rows])
            )
        )
        # Bound final-dtype casts plus accumulation/formula roundoff. This is
        # intentionally derived from the actual group magnitudes, not a fixed
        # tolerance that would silently loosen for low-precision tensors.
        bound = 8.0 * dtype_info.eps * torch.maximum(scale, torch.ones_like(scale))
        bound = bound + 8.0 * dtype_info.tiny * length_work[negative_rows].sum()
        conservation_errors.append(float(error.item()))
        conservation_bounds.append(float(bound.item()))
        if error > bound:
            raise RuntimeError(
                "FA-RAR group conservation exceeded final-dtype rounding bound: "
                f"error={error.item()} bound={bound.item()}"
            )

    noneligible_negative = negative & ~eligible
    lengths_double = lengths.double()
    before_negative_mass = torch.sum(lengths_double[negative] * work[negative])
    after_negative_mass = torch.sum(lengths_double[negative] * projected[negative].double())
    baseline_array = np.asarray(baselines, dtype=np.float64)
    group_count = len(grouped_rows)
    negative_count = int(negative.sum().item())
    eligible_count = int(eligible.sum().item())
    eligible_pfa = p_fa.double()[eligible]
    eligible_distortion = distortion[eligible]
    negative_token_count = lengths_double[negative].sum()
    raw_correction_token_mass = torch.sum(lengths_double[negative] * distortion[negative])
    metrics = {
        "fa_rar/trajectory_count": float(row_count),
        "fa_rar/group_count": float(group_count),
        "fa_rar/group_with_negative_count": float(len(baselines)),
        "fa_rar/group_without_negative_count": float(group_count - len(baselines)),
        "fa_rar/baseline_group_count": float(len(baselines)),
        "fa_rar/baseline_mean": float(baseline_array.mean()) if baselines else 0.0,
        "fa_rar/baseline_min": float(baseline_array.min()) if baselines else 0.0,
        "fa_rar/baseline_max": float(baseline_array.max()) if baselines else 0.0,
        "fa_rar/baseline_rms": (
            float(np.sqrt(np.mean(np.square(baseline_array)))) if baselines else 0.0
        ),
        "fa_rar/negative_trajectory_count": float(negative_count),
        "fa_rar/eligible_trajectory_count": float(eligible_count),
        "fa_rar/eligible_rate_given_negative": float(eligible_count / negative_count) if negative_count else 0.0,
        "fa_rar/noneligible_negative_trajectory_count": float(noneligible_negative.sum().item()),
        "fa_rar/pfa_mean_eligible": _response_mean(p_fa.double(), eligible),
        "fa_rar/negative_before_adv_mean": _response_mean(work, negative),
        "fa_rar/negative_after_adv_mean": _response_mean(projected.double(), negative),
        "fa_rar/eligible_before_adv_mean": _response_mean(work, eligible),
        "fa_rar/eligible_after_adv_mean": _response_mean(projected.double(), eligible),
        "fa_rar/noneligible_negative_before_adv_mean": _response_mean(work, noneligible_negative),
        "fa_rar/noneligible_negative_after_adv_mean": _response_mean(
            projected.double(), noneligible_negative
        ),
        "fa_rar/negative_before_adv_token_weighted_sum": float(before_negative_mass.item()),
        "fa_rar/negative_after_adv_token_weighted_sum": float(after_negative_mass.item()),
        "fa_rar/negative_net_correction_token_weighted_sum": float(
            (after_negative_mass - before_negative_mass).item()
        ),
        "fa_rar/conservation_error_abs": float(
            torch.abs(after_negative_mass - before_negative_mass).item()
        ),
        "fa_rar/conservation_error_group_max": max(conservation_errors, default=0.0),
        "fa_rar/conservation_rounding_bound_group_max": max(conservation_bounds, default=0.0),
        "fa_rar/sign_flip_count": 0.0,
        "fa_rar/nonnegative_drift_max": 0.0,
        # Stable public telemetry names requested by the FA-RAR protocol.
        "fa_rar/eligible_traj_count": float(eligible_count),
        "fa_rar/eligible_rate": float(eligible_count / row_count) if row_count else 0.0,
        "fa_rar/negative_traj_count": float(negative_count),
        "fa_rar/negative_token_count": float(negative_token_count.item()),
        "fa_rar/pfa_mean": (
            float(eligible_pfa.mean().item()) if eligible_pfa.numel() else 0.0
        ),
        "fa_rar/pfa_max": float(eligible_pfa.max().item()) if eligible_pfa.numel() else 0.0,
        "fa_rar/raw_correction_mean": (
            float(eligible_distortion.mean().item()) if eligible_distortion.numel() else 0.0
        ),
        "fa_rar/raw_correction_max": (
            float(eligible_distortion.max().item()) if eligible_distortion.numel() else 0.0
        ),
        "fa_rar/raw_correction_token_mass": float(raw_correction_token_mass.item()),
        "fa_rar/centering_baseline_mean": (
            float(baseline_array.mean()) if baselines else 0.0
        ),
        "fa_rar/centering_baseline_max": (
            float(baseline_array.max()) if baselines else 0.0
        ),
        "fa_rar/eligible_advantage_before_mean": _response_mean(work, eligible),
        "fa_rar/eligible_advantage_after_mean": _response_mean(projected.double(), eligible),
        "fa_rar/noneligible_negative_before_mean": _response_mean(work, noneligible_negative),
        "fa_rar/noneligible_negative_after_mean": _response_mean(
            projected.double(), noneligible_negative
        ),
        "fa_rar/token_weighted_advantage_before": float(before_negative_mass.item()),
        "fa_rar/token_weighted_advantage_after": float(after_negative_mass.item()),
        "fa_rar/net_token_weighted_correction": float(
            (after_negative_mass - before_negative_mass).item()
        ),
        "fa_rar/conservation_error_max": max(conservation_errors, default=0.0),
    }
    return projected, metrics


def apply_fa_cac_post_advantage_hook(
    data: DataProto,
    *,
    evidence: ForcedAnswerCensorEvidence | ForcedAnswerReliabilityEvidence | None,
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
    mode = _get(cac_config, "mode", None)
    if mode == "reliability_redistribution":
        if not isinstance(evidence, ForcedAnswerReliabilityEvidence):
            raise RuntimeError("FA-RAR enabled but task-score-free reliability evidence is absent")
        return _apply_fa_rar_post_advantage_hook(
            data,
            evidence=evidence,
            algorithm_config=algorithm_config,
            apply=apply,
        )
    if mode != "attenuate_negative_correctness":
        raise RuntimeError("unsupported FA-CAC mode reached post-advantage hook")
    if not isinstance(evidence, ForcedAnswerCensorEvidence):
        raise RuntimeError("FA-CAC enabled but parent-keyed forced-answer evidence is absent")
    raw_adv_estimator = _get(algorithm_config, "adv_estimator", None)
    adv_estimator = getattr(raw_adv_estimator, "value", raw_adv_estimator)
    if adv_estimator != "grpo":
        raise RuntimeError("FA-CAC post-advantage hook requires canonical Vanilla GRPO")
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


def _apply_fa_rar_post_advantage_hook(
    data: DataProto,
    *,
    evidence: ForcedAnswerReliabilityEvidence,
    algorithm_config: Any,
    apply: bool,
) -> tuple[DataProto, dict[str, float]]:
    """Apply or shadow task-score-free FA reliability redistribution."""
    raw_adv_estimator = _get(algorithm_config, "adv_estimator", None)
    adv_estimator = getattr(raw_adv_estimator, "value", raw_adv_estimator)
    if adv_estimator != "grpo":
        raise RuntimeError("FA-RAR post-advantage hook requires canonical Vanilla GRPO")

    required = {"token_level_rewards", "response_mask", "advantages", "returns"}
    missing = required - set(data.batch.keys())
    if missing:
        raise RuntimeError(f"FA-RAR batch is missing required tensors: {sorted(missing)}")
    tensor_before = {key: value.detach().clone() for key, value in data.batch.items()}
    vanilla_advantages = tensor_before["advantages"]
    vanilla_returns = tensor_before["returns"]
    rewards_before = tensor_before["token_level_rewards"]
    response_mask = tensor_before["response_mask"]
    if vanilla_advantages.ndim != 2 or response_mask.shape != vanilla_advantages.shape:
        raise RuntimeError("FA-RAR Vanilla advantages and response mask must be aligned rank-2 tensors")
    if vanilla_returns.shape != vanilla_advantages.shape or rewards_before.shape != vanilla_advantages.shape:
        raise RuntimeError("FA-RAR Vanilla returns and rewards must align with advantages")
    if not vanilla_advantages.is_floating_point() or not torch.all(torch.isfinite(vanilla_advantages)):
        raise RuntimeError("FA-RAR Vanilla token advantages must be finite floating-point values")
    if not torch.equal(vanilla_returns, vanilla_advantages):
        raise RuntimeError("FA-RAR canonical GRPO returns must bitwise match Vanilla advantages")

    valid_mask = response_mask.to(dtype=torch.bool)
    if response_mask.is_floating_point() and not torch.all(torch.isfinite(response_mask)):
        raise RuntimeError("FA-RAR response mask must be finite")
    if not torch.equal(response_mask, valid_mask.to(dtype=response_mask.dtype)):
        raise RuntimeError("FA-RAR response mask must contain only zeros and ones")
    response_token_counts = valid_mask.sum(dim=-1)
    if torch.any(response_token_counts <= 0):
        bad_rows = torch.nonzero(response_token_counts <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "FA-RAR retained PPO rows must each contain at least one valid response token; "
            f"zero-token rows={bad_rows}"
        )
    row_count = len(vanilla_advantages)
    vanilla_scalar = torch.empty(
        row_count, dtype=vanilla_advantages.dtype, device=vanilla_advantages.device
    )
    for row in range(row_count):
        valid_values = vanilla_advantages[row, valid_mask[row]]
        scalar = valid_values[0]
        if not torch.equal(valid_values, scalar.expand_as(valid_values)):
            raise RuntimeError(
                "FA-RAR must run immediately after response-level canonical GRPO; "
                f"row {row} has non-constant valid-token advantages"
            )
        vanilla_scalar[row] = scalar

    parents = np.asarray(evidence.current_row_to_parent, dtype=np.int64)
    uids = data.non_tensor_batch.get("uid")
    parent_count = len(evidence.hit_response_cap)
    parent_arrays = (
        evidence.probe_attempted,
        evidence.context_overflow,
        evidence.original_correctness_by_parent,
    )
    if any(len(array) != parent_count for array in parent_arrays):
        raise RuntimeError("FA-RAR parent evidence arrays must remain aligned")
    if len(parents) != row_count or uids is None or len(uids) != row_count:
        raise RuntimeError("FA-RAR UID and row identity must align with the retained PPO batch")
    if len(set(parents.tolist())) != len(parents):
        raise RuntimeError("FA-RAR retained PPO rows must have unique parent identity")
    if np.any(parents < 0) or np.any(parents >= parent_count):
        raise RuntimeError("FA-RAR retained PPO row has an invalid parent identity")
    if not np.all(np.isfinite(evidence.original_correctness_by_parent)):
        raise RuntimeError("FA-RAR raw correctness must be finite")
    if not np.isfinite(evidence.correctness_threshold):
        raise RuntimeError("FA-RAR correctness threshold must be finite")
    for parent, raw_pfa in evidence.pfa_by_parent.items():
        if parent < 0 or parent >= parent_count:
            raise RuntimeError(f"FA-RAR pFA has invalid parent identity {parent}")
        pfa = float(raw_pfa)
        if not np.isfinite(pfa) or not 0.0 <= pfa <= 1.0:
            raise RuntimeError(f"FA-RAR trajectory {parent} pFA must be in [0, 1]")

    p_fa = torch.zeros_like(vanilla_scalar)
    eligible = torch.zeros(row_count, dtype=torch.bool, device=vanilla_scalar.device)
    candidate_count = 0
    excluded_pfa_zero_count = 0
    excluded_nonnegative_vanilla_adv_count = 0
    for row, parent in enumerate(parents.tolist()):
        candidate = bool(
            evidence.hit_response_cap[parent]
            and evidence.probe_attempted[parent]
            and not evidence.context_overflow[parent]
            and evidence.original_correctness_by_parent[parent] < evidence.correctness_threshold
        )
        if not candidate:
            continue
        candidate_count += 1
        if parent not in evidence.pfa_by_parent:
            raise RuntimeError(f"FA-RAR censor candidate trajectory {parent} has no pFA")
        pfa = float(evidence.pfa_by_parent[parent])
        if vanilla_scalar[row] >= 0:
            excluded_nonnegative_vanilla_adv_count += 1
            continue
        if pfa <= 0:
            excluded_pfa_zero_count += 1
            continue
        eligible[row] = True
        p_fa[row] = pfa

    eligible_count = int(eligible.sum().item())
    if candidate_count != (
        eligible_count + excluded_pfa_zero_count + excluded_nonnegative_vanilla_adv_count
    ):
        raise RuntimeError("FA-RAR candidate exclusion accounting invariant failed")
    projected_scalar, metrics = compute_fa_reliability_redistributed_advantage(
        vanilla_scalar,
        p_fa,
        response_mask,
        uids,
        eligible,
    )

    projected_advantages = vanilla_advantages.clone()
    projected_returns = vanilla_returns.clone()
    for row in range(row_count):
        projected_advantages[row, valid_mask[row]] = projected_scalar[row]
        projected_returns[row, valid_mask[row]] = projected_scalar[row]

    scalar_drift = projected_scalar - vanilla_scalar
    negative = vanilla_scalar < 0
    positive = vanilla_scalar > 0
    zero = vanilla_scalar == 0
    sign_flip_count = int(torch.sum(negative & (projected_scalar > 0)).item())
    positive_drift_max = (
        float(torch.max(torch.abs(scalar_drift[positive])).item()) if torch.any(positive) else 0.0
    )
    zero_drift_max = float(torch.max(torch.abs(scalar_drift[zero])).item()) if torch.any(zero) else 0.0
    padding_drift_max = (
        float(torch.max(torch.abs((projected_advantages - vanilla_advantages)[~valid_mask])).item())
        if torch.any(~valid_mask)
        else 0.0
    )
    if sign_flip_count:
        raise RuntimeError("FA-RAR negative Vanilla advantage became positive")
    if positive_drift_max != 0.0 or zero_drift_max != 0.0:
        raise RuntimeError("FA-RAR changed a positive or zero Vanilla advantage")
    if padding_drift_max != 0.0:
        raise RuntimeError("FA-RAR changed response padding")
    if not torch.equal(projected_returns, projected_advantages):
        raise RuntimeError("FA-RAR projected GRPO returns must bitwise match projected advantages")

    for key, before in tensor_before.items():
        if key in {"advantages", "returns"}:
            continue
        after = data.batch.get(key)
        if after is None or not torch.equal(after, before):
            raise RuntimeError(f"FA-RAR changed guarded batch tensor {key!r}")
    if apply:
        data.batch["advantages"] = projected_advantages
        data.batch["returns"] = projected_returns

    drift_abs_max = (
        float(torch.max(torch.abs(scalar_drift)).item()) if scalar_drift.numel() else 0.0
    )
    metrics.update(
        {
            "fa_rar/enabled": 1.0,
            "fa_rar/applied": float(apply),
            "fa_rar/shadow": float(not apply),
            "fa_rar/candidate_count": float(candidate_count),
            "fa_rar/eligible_count": float(eligible_count),
            "fa_rar/excluded_pfa_zero_count": float(excluded_pfa_zero_count),
            "fa_rar/excluded_nonnegative_vanilla_adv_count": float(
                excluded_nonnegative_vanilla_adv_count
            ),
            "fa_rar/pfa_mean": _response_mean(p_fa.double(), eligible),
            "fa_rar/sign_flip_count": float(sign_flip_count),
            "fa_rar/positive_advantage_drift_max": float(positive_drift_max),
            "fa_rar/zero_advantage_drift_max": float(zero_drift_max),
            "fa_rar/padding_advantage_drift_max": float(padding_drift_max),
            "fa_rar/returns_consistency_error_max": 0.0,
            "fa_rar/reward_drift_max": 0.0,
            "fa_rar/score_drift_max": 0.0,
            "fa_rar/other_tensor_changed_count": 0.0,
            "fa_rar/projected_changed_trajectory_count": float(torch.sum(scalar_drift != 0).item()),
            "fa_rar/projected_changed_token_count": float(
                response_token_counts[scalar_drift != 0].sum().item()
            ),
            "fa_rar/projected_drift_abs_max": drift_abs_max,
            "fa_rar/actor_visible_advantage_drift_max": drift_abs_max if apply else 0.0,
        }
    )
    return data, metrics
