"""Original trajectory and prefix actor tensor contracts."""
import torch


def prefix_lengths(response_mask, eta_percent):
    if not 0 < eta_percent <= 100:
        raise ValueError("invalid integer eta")
    lengths = response_mask.long().sum(-1)
    expected = torch.arange(response_mask.shape[-1], device=response_mask.device)[None, :] < lengths[:, None]
    if not torch.equal(expected, response_mask.bool()) or (lengths <= 0).any():
        raise ValueError("original responses must be nonempty and right padded")
    return lengths, (lengths * eta_percent) // 100


def actor_batch(batch, view, pad_token_id):
    """Call only on final retained rows, after immutable length/reward statistics exist."""
    h = batch.batch["pppo_prefix_length"]
    old_width = batch.batch["responses"].shape[-1]
    mask = torch.arange(old_width, device=h.device)[None, :] < h[:, None]
    batch.batch["pppo_loss_mask"] = mask
    if view not in ("cropped", "full_mask_reference"):
        raise ValueError("unknown actor view")
    zero_rows = int((h == 0).sum())
    if view == "cropped":
        # Keep one original response token only for a zero-prefix row so the
        # generic ragged log-prob/reward adapters have a nonempty response.
        # Its PPO mask remains empty: this guard token receives no loss/gradient.
        # Other rows must still crop; a single zero must not uncrop the batch.
        visible_lengths = h.clamp(min=1)
        width = int(visible_lengths.max())
        for key in list(batch.batch.keys()):
            value = batch.batch[key]
            if value.ndim == 2 and value.shape[-1] == old_width and key not in (
                "prompts", "input_ids", "attention_mask", "position_ids"
            ):
                batch.batch[key] = value[:, :width].clone()
        mask = mask[:, :width]
        visible_mask = torch.arange(width, device=h.device)[None, :] < visible_lengths[:, None]
        batch.batch["responses"] = torch.where(visible_mask, batch.batch["responses"], pad_token_id)
        prompt_width = batch.batch["prompts"].shape[-1]
        prompt_mask = batch.batch["attention_mask"][:, :prompt_width]
        attention = torch.cat([prompt_mask, visible_mask.to(prompt_mask.dtype)], -1)
        batch.batch["input_ids"] = torch.cat([batch.batch["prompts"], batch.batch["responses"]], -1)
        batch.batch["attention_mask"] = attention
        batch.batch["position_ids"] = (attention.cumsum(-1) - 1).clamp(min=0)
        batch.batch["response_mask"] = visible_mask.to(prompt_mask.dtype)
    for key in ("old_log_probs", "ref_log_prob", "rollout_log_probs"):
        batch.batch.pop(key, None)
    # Token rewards are for generic diagnostics only; never reconstruct A from these.
    scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    last = batch.batch["response_mask"].sum(-1).long() - 1
    scores[torch.arange(len(batch), device=last.device), last] = batch.batch["pppo_reward"]
    batch.batch["token_level_scores"] = scores
    batch.batch["token_level_rewards"] = scores.clone()
    advantages = batch.batch["pppo_advantage"][:, None] * batch.batch["pppo_loss_mask"]
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = advantages.clone()
    batch.meta_info["pppo_zero_prefix_fallback"] = False
    batch.meta_info["pppo_zero_prefix_rows"] = zero_rows
    return batch
