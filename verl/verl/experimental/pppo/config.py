"""Fail closed on inherited objectives and ambiguous PPPO configurations."""
from collections.abc import Mapping


class _View:
    """Lazy access for either resolved plain mappings or Hydra DictConfig."""
    def __init__(self, node):
        self.node = node

    def __getattr__(self, key):
        value = self.node[key]
        return _View(value) if isinstance(value, Mapping) else value


def _select(node, path, default=None):
    for part in path.split("."):
        if isinstance(node, _View):
            node = node.node
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def validate_config(c):
    c = _View(c) if isinstance(c, Mapping) else c
    p = c.algorithm.pppo
    a, r = c.actor_rollout_ref.actor, c.actor_rollout_ref.rollout
    if not p.enable or p.profile not in ("paper_equation", "ncbr_matched"):
        raise ValueError("PPPO must select an explicit profile")
    if r.n != p.originals_per_prompt or r.n != 8 or p.extra_continuations_per_prefix not in (1, 8):
        raise ValueError("PPPO requires N=8, G=8 (G=1 is debug only)")
    if r.response_length != p.response_cap or c.data.max_response_length != p.response_cap:
        raise ValueError("PPPO original response caps disagree")
    if r.max_model_len < c.data.max_prompt_length + p.response_cap:
        raise ValueError("PPPO context is too short")
    if a.policy_loss.loss_mode != "pppo" or a.clip_ratio_c is not None:
        raise ValueError("PPPO requires independent loss and no dual clipping")
    if a.clip_ratio_low != 0.2 or a.clip_ratio_high != 0.28:
        raise ValueError("PPPO clip must be 0.2/0.28")
    for key in ("algorithm.use_kl_in_reward", "algorithm.probe_credit.enable", "actor_rollout_ref.actor.use_kl_loss",
                "actor_rollout_ref.actor.entropy_coeff", "distillation.enabled", "critic.enable",
                "algorithm.rollout_correction.bypass_mode", "algorithm.rollout_correction.rollout_is",
                "algorithm.rollout_correction.rollout_rs", "ncbr.enable"):
        if _select(c, key):
            raise ValueError(f"PPPO incompatible option: {key}")
    if _select(c, "actor_rollout_ref.rollout.boundary_return.mode", default="off") != "off":
        raise ValueError("PPPO cannot activate NCBR")
    if _select(c, "reward.reward_kwargs.overlong_buffer_cfg.enable", default=False):
        raise ValueError("PPPO cannot use shaped length reward")
    if not c.algorithm.filter_groups.enable and c.data.gen_batch_size != c.data.train_batch_size:
        raise ValueError("unfiltered PPPO requires gen=train")
    if p.schedule.type not in ("fixed", "plateau") or p.eta_percent not in (15, 20, 25, 30, 35):
        raise ValueError("invalid PPPO schedule")
    if p.actor_view not in ("cropped", "full_mask_reference"):
        raise ValueError("invalid PPPO actor view")
    if c.data.train_batch_size % a.ppo_mini_batch_size:
        raise ValueError("partial optimizer mini batches are unsupported")
    if not p.holdout_hash:
        raise ValueError("prepare and hash the isolated holdout before launch")
