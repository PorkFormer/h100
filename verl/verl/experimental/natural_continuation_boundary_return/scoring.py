"""Original chunking and reward-worker padding, shared by both entry points."""
import numpy as np
import torch
from .profiling import IntervalRecorder
from .reward_adapter import BoundaryRewardOutput, build_long_reward_batch, extract_required_reward_scalars


def score_long_generations(candidate, generations, boundary, *, score_batch, pad_token_id,
                           reward_worker_count=1, global_step=0, build_batch=build_long_reward_batch):
    correctness_parts: list[np.ndarray] = []
    task_score_parts: list[np.ndarray] = []
    recorder = IntervalRecorder(f"boundary-long-reward-step-{global_step}")
    with recorder.record(
        "boundary_long_reward",
        metadata={"rows": len(generations), "chunk_size": boundary.long_reward_chunk_size},
    ):
        for chunk_index, start in enumerate(range(0, len(generations), boundary.long_reward_chunk_size)):
            chunk = generations[start : start + boundary.long_reward_chunk_size]
            full_tokens = sum(
                len(getattr(item, "prefix_token_ids", ())) + len(getattr(item, "tail_token_ids", ()))
                for item in chunk
            )
            with recorder.record(
                "boundary_long_reward_chunk",
                metadata={"chunk_index": chunk_index, "rows": len(chunk), "full_response_tokens": full_tokens},
            ):
                with recorder.record(
                    "long_reward_batch_build",
                    metadata={"chunk_index": chunk_index, "rows": len(chunk), "full_response_tokens": full_tokens},
                ):
                    long_batch = build_batch(
                        candidate,
                        chunk,
                        pad_token_id=pad_token_id,
                    )
                    long_batch.meta_info["boundary_reward_only"] = True
                    original_count = len(long_batch)
                    padding_count = (-original_count) % reward_worker_count
                    long_batch.padding(padding_count, padding_candidate="last")
                with recorder.record(
                    "long_reward_model_forward",
                    metadata={"chunk_index": chunk_index, "rows": original_count, "padded_rows": len(long_batch)},
                ):
                    normalized = score_batch(long_batch)
                scalars = extract_required_reward_scalars(
                    BoundaryRewardOutput(
                        reward_tensor=normalized.reward_tensor,
                        extra_info=normalized.extra_info,
                    ),
                    expected_count=len(long_batch),
                    correctness_key=boundary.correctness_key,
                    task_score_key=boundary.task_score_key,
                )
                correctness_parts.append(scalars.correctness[:original_count])
                task_score_parts.append(scalars.task_score[:original_count])
    return BoundaryRewardOutput(
        reward_tensor=torch.empty((len(generations), 0), dtype=torch.float32),
        extra_info={
            boundary.correctness_key: np.concatenate(correctness_parts),
            boundary.task_score_key: np.concatenate(task_score_parts),
        },
        profiling_intervals=tuple(recorder.intervals),
    )
