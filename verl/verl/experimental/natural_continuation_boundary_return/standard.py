"""Reward and lifecycle adapters for the existing RayPPOTrainer."""
import contextlib

import numpy as np

from verl.experimental.agent_loop.agent_loop import build_rollout_sampling_params
from verl.trainer.ppo.reward import extract_reward
from .hook import NCBRHook
from .reward_adapter import BoundaryRewardOutput
from .scoring import score_long_generations


@contextlib.contextmanager
def rollout_phase(trainer, boundary):
    """Keep replicas awake through verification; never sleep on unattested cleanup."""
    try:
        yield
    except BaseException as error:
        if boundary.mode != "off":
            if getattr(error, "boundary_remote_cleanup_attested", True):
                try:
                    trainer.checkpoint_manager.sleep_replicas()
                except BaseException as cleanup_error:
                    error.add_note(f"replica sleep failed: {cleanup_error!r}")
            else:
                error.add_note("replicas remain awake: remote continuation drain/release is unattested")
        raise
    else:
        if boundary.mode != "off":
            trainer.checkpoint_manager.sleep_replicas()


def align_standard_identity(batch):
    """Assign trajectory identities before the existing batch balancing operation."""
    versions = batch.non_tensor_batch.get("global_steps")
    if versions is None or len(versions) != len(batch):
        raise ValueError("normal rollout is missing actual server global_steps")
    batch.non_tensor_batch["rollout_policy_version"] = np.asarray(versions, dtype=object).copy()
    ordinals = {}
    ids = []
    for uid in batch.non_tensor_batch["uid"]:
        ordinal = ordinals.get(str(uid), 0)
        ids.append(f"{uid}:{ordinal}")
        ordinals[str(uid)] = ordinal + 1
    batch.non_tensor_batch["trajectory_id"] = np.asarray(ids, dtype=object)


def apply_standard_boundary(trainer, batch, raw_scores, extras, boundary, timing_raw):
    def score_batch(long_batch):
        # Same colocated reward pipeline as the original standard entry point.
        long_batch = long_batch.union(trainer._compute_reward_colocate(long_batch))
        scores, reward_extras = extract_reward(long_batch)
        return BoundaryRewardOutput(scores, reward_extras)

    def score_long(candidate, generations, config):
        manager = getattr(trainer, "reward_loop_manager", None)
        return score_long_generations(
            candidate, generations, config, score_batch=score_batch,
            pad_token_id=trainer.tokenizer.pad_token_id,
            reward_worker_count=len(getattr(manager, "reward_loop_workers", ())) or 1,
            global_step=trainer.global_steps,
        )

    rollout = trainer.config.actor_rollout_ref.rollout
    return NCBRHook().apply(
        batch, raw_scores, raw_verifier_extras=extras, config=boundary,
        policy_version=trainer._ncbr_policy_version,
        sampling_params=build_rollout_sampling_params(rollout),
        continuation_client=trainer.llm_server_manager.get_client(), score_long=score_long,
        eos_token_id=trainer.tokenizer.eos_token_id, short_response_length=rollout.response_length,
        max_model_len=rollout.max_model_len, timing_raw=timing_raw,
    )
