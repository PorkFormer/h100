"""Equation-literal PPPO objective. No prefix-token normalization."""
import torch


def group_statistics(reward, lengths, group_ids, n=8, correction=1, epsilon=1e-6):
    if reward.ndim != 1 or reward.shape != lengths.shape or len(group_ids) != len(reward):
        raise ValueError("PPPO row shapes disagree")
    if not torch.isfinite(reward).all() or (lengths <= 0).any():
        raise ValueError("nonfinite reward or empty original response")
    advantage = torch.zeros_like(reward)
    total = torch.zeros_like(lengths)
    groups = {}
    for i, uid in enumerate(group_ids):
        groups.setdefault(str(uid), []).append(i)
    for rows in groups.values():
        if len(rows) != n:
            raise ValueError("PPPO requires complete original groups")
        values = reward[rows]
        advantage[rows] = (values - values.mean()) / (values.std(correction=correction) + epsilon)
        total[rows] = lengths[rows].sum()
    return advantage, total, n / total.to(reward.dtype)


def prefix_objective(log_prob, old_log_prob, advantage, mask, row_weight,
                     global_rows, dp_size=1, clip_low=0.2, clip_high=0.28):
    if log_prob.shape != old_log_prob.shape or log_prob.shape != mask.shape:
        raise ValueError("PPPO token shape mismatch")
    if advantage.shape != row_weight.shape or advantage.shape != log_prob.shape[:1]:
        raise ValueError("PPPO scalar row shape mismatch")
    if global_rows <= 0 or dp_size <= 0 or (row_weight <= 0).any():
        raise ValueError("invalid PPPO normalization")
    # Mask before exponentiation: unoptimized suffixes must not overflow or carry gradients.
    delta = torch.where(mask.bool(), log_prob - old_log_prob, 0)
    ratio = delta.exp()
    a = advantage[:, None]
    terms = torch.maximum(-ratio * a, -ratio.clamp(1 - clip_low, 1 + clip_high) * a)
    loss = (torch.where(mask.bool(), terms, 0).sum(-1) * row_weight).sum() * (dp_size / global_rows)
    if not torch.isfinite(loss):
        raise ValueError("nonfinite PPPO objective")
    return loss


def pppo_loss(config, model_output, data):
    from verl.utils.metric import Metric, AggregationType
    from verl.workers.utils.padding import no_padding_2_padding
    required = {"pppo_loss_mask", "pppo_row_weight", "pppo_advantage", "old_log_probs", "response_mask"}
    if not required.issubset(data.keys()):
        raise ValueError(f"missing PPPO fields: {required - set(data.keys())}")
    if any(k.startswith("support_") or k == "ppo_response_mask" for k in data.keys()):
        raise ValueError("PPPO cannot use support-floor fields")
    if config.use_kl_loss or config.entropy_coeff or config.clip_ratio_c is not None or "rollout_is_weights" in data:
        raise ValueError("PPPO auxiliary objectives are unsupported")
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    selected = data.select(*sorted(required)).to_padded_tensor()
    mask = selected["pppo_loss_mask"].bool()
    if (mask & ~selected["response_mask"].bool()).any():
        raise ValueError("PPPO prefix mask outside response")
    loss = prefix_objective(log_prob.float(), selected["old_log_probs"].float(),
                            selected["pppo_advantage"].float(), mask,
                            selected["pppo_row_weight"].float(),
                            int(data["global_batch_size"]), int(data["dp_size"]),
                            config.clip_ratio_low, config.clip_ratio_high)
    ratio = torch.where(mask, log_prob.float() - selected["old_log_probs"].float(), 0).exp()
    clipped = ((ratio < 1 - config.clip_ratio_low) | (ratio > 1 + config.clip_ratio_high)) & mask
    return loss, {
        "actor/pg_loss": Metric(value=loss.detach(), aggregation=AggregationType.SUM),
        "actor/pppo_prefix_clip_count": Metric(value=clipped.sum(), aggregation=AggregationType.SUM),
        "actor/pppo_prefix_token_count": Metric(value=mask.sum(), aggregation=AggregationType.SUM),
    }
