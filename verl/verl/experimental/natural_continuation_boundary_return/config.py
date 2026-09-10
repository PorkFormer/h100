"""Resolve the optional top-level switch without changing legacy mode defaults."""
from dataclasses import replace

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.rollout import BoundaryReturnConfig


def config_get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def resolve_boundary_config(config):
    rollout = config_get(config_get(config, "actor_rollout_ref"), "rollout")
    raw = config_get(rollout, "boundary_return", {})
    enable = config_get(config_get(config, "ncbr"), "enable", config_get(raw, "enable"))
    if enable is not None and not isinstance(enable, bool):
        raise ValueError("ncbr.enable must be null or boolean")
    if enable is False:
        return BoundaryReturnConfig(enable=False, mode="off")
    boundary = raw if isinstance(raw, BoundaryReturnConfig) else omega_conf_to_dataclass(raw, BoundaryReturnConfig)
    if enable is True and boundary.mode not in ("shadow", "replace"):
        raise ValueError("ncbr.enable=true requires explicit boundary_return.mode=shadow or replace")
    return replace(boundary, enable=enable)


def map_ncbr_config(config):
    """Map into the typed rollout config before worker initialization."""
    from omegaconf import OmegaConf
    boundary = resolve_boundary_config(config)
    if boundary.enable is not None:
        OmegaConf.update(config, "actor_rollout_ref.rollout.boundary_return.enable", boundary.enable, force_add=True)
        OmegaConf.update(config, "actor_rollout_ref.rollout.boundary_return.mode", boundary.mode, force_add=True)
    return boundary
