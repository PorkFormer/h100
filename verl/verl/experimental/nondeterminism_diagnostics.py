"""Opt-in, non-mutating boundary diagnostics for stochastic DAPO launches.

This module is diagnostic-only.  It records identities already present at four
training boundaries and deliberately never creates a sampling seed or adds a
field to a :class:`DataProto`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from verl import DataProto


SCHEMA_VERSION = "obcf-gate-d-nondeterminism-boundary-v2"
BOUNDARY_NAMES = {
    0: "post_dataloader_pre_generation",
    1: "post_rollout_pre_reward",
    2: "post_reward_pre_dynamic_filter",
    3: "post_dynamic_filter",
}


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """Resolved identity and output settings for one diagnostic launch."""

    enabled: bool = False
    output_dir: Path | str | None = None
    run_id: str | None = None
    config_sha256: str | None = None
    git_commit: str | None = None
    rank: int = 0
    ray_actor_identity: str | None = None
    vllm_engine_identity: str | None = None
    tp_group_identity: str | None = None
    dataloader_base_seed: int | None = None
    sampler_generator_hash: str | None = None

    def validate(self) -> None:
        if not self.enabled:
            return
        missing = [
            name
            for name in ("output_dir", "run_id", "config_sha256", "git_commit")
            if getattr(self, name) in (None, "")
        ]
        if missing:
            raise ValueError(f"enabled nondeterminism diagnostics missing required settings: {missing}")
        if int(self.rank) < 0:
            raise ValueError("diagnostic rank must be nonnegative")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite_float": repr(value)}
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _update_tensor_hash(hasher: Any, name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach().cpu().contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(detached.dtype).encode("ascii"))
    hasher.update(_canonical_json_bytes(list(detached.shape)))
    if detached.numel():
        hasher.update(detached.view(torch.uint8).numpy().tobytes(order="C"))


def data_proto_semantic_hash(batch: DataProto) -> str:
    """Hash DataProto content and insertion order without serializing or mutating it."""

    hasher = hashlib.sha256()
    hasher.update(b"obcf-gate-d-dataproto-semantic-hash-v1\0")
    for name, tensor in batch.batch.items():
        _update_tensor_hash(hasher, str(name), tensor)
    hasher.update(b"\0non-tensors\0")
    for name, value in batch.non_tensor_batch.items():
        hasher.update(str(name).encode("utf-8"))
        hasher.update(_canonical_json_bytes(value))
    hasher.update(b"\0meta-info\0")
    hasher.update(_canonical_json_bytes(batch.meta_info))
    return hasher.hexdigest()


def _sha256_tokens(tokens: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _array_item(batch: DataProto, key: str, index: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key)
    if values is None or len(values) <= index:
        return default
    return _jsonable(values[index])


def _extra_info_item(batch: DataProto, key: str, index: int, default: Any = None) -> Any:
    extra_info = _array_item(batch, "extra_info", index, {})
    if isinstance(extra_info, dict):
        return extra_info.get(key, default)
    return default


def _prompt_tokens(batch: DataProto, index: int) -> list[int]:
    prompts = batch.batch.get("prompts")
    if prompts is None:
        input_ids = batch.batch.get("input_ids")
        if input_ids is None:
            return []
        prompt_width = int(input_ids.shape[-1])
        tokens = input_ids[index]
    else:
        prompt_width = int(prompts.shape[-1])
        tokens = prompts[index]
    attention_mask = batch.batch.get("attention_mask")
    if attention_mask is not None:
        mask = attention_mask[index, :prompt_width].bool()
        tokens = tokens[-prompt_width:][mask]
    return [int(token) for token in tokens.detach().cpu().tolist()]


def _response_tokens(batch: DataProto, index: int) -> list[int]:
    responses = batch.batch.get("responses")
    if responses is None:
        return []
    mask = batch.batch.get("response_mask")
    tokens = responses[index]
    if mask is not None:
        tokens = tokens[mask[index].bool()]
    return [int(token) for token in tokens.detach().cpu().tolist()]


def _terminal_reward(batch: DataProto, index: int) -> float | None:
    for name in ("token_level_scores", "token_level_rewards", "rm_scores"):
        values = batch.batch.get(name)
        if values is not None:
            return float(values[index].detach().float().sum().cpu().item())
    for name in ("score", "acc", "reward"):
        value = _array_item(batch, name, index)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _prompt_id(batch: DataProto, index: int, prompt_token_hash: str) -> Any:
    for name in ("prompt_id", "dataset_row_id", "index", "uid"):
        value = _array_item(batch, name, index)
        if value is not None:
            return value
        value = _extra_info_item(batch, name, index)
        if value is not None:
            return _jsonable(value)
    return prompt_token_hash


def _dataset_row_id(batch: DataProto, index: int) -> Any:
    for name in ("dataset_row_id", "index", "row_id"):
        value = _array_item(batch, name, index)
        if value is not None:
            return value
        value = _extra_info_item(batch, name, index)
        if value is not None:
            return _jsonable(value)
    return None


def _explicit_sampling_seed(batch: DataProto, index: int) -> Any:
    for name in ("assigned_sampling_seed", "sampling_seed", "request_sampling_seed"):
        value = _array_item(batch, name, index)
        if value is not None:
            return value
    return None


class NondeterminismDiagnostics:
    """Atomic writer for the four Gate D v2 nondeterminism boundaries."""

    def __init__(self, config: DiagnosticsConfig):
        config.validate()
        self.config = config

    def _common_record(
        self,
        batch: DataProto,
        index: int,
        *,
        global_step: int,
        generation_batch_index: int,
        rollout_n: int,
    ) -> dict[str, Any]:
        prompt_tokens = _prompt_tokens(batch, index)
        prompt_token_hash = _sha256_tokens(prompt_tokens)
        prompt_id = _prompt_id(batch, index, prompt_token_hash)
        rollout_index = _array_item(batch, "rollout_index", index)
        if rollout_index is None:
            rollout_index = index % int(rollout_n)
        request_id = _array_item(batch, "request_id", index)
        if request_id is None:
            uid = _array_item(batch, "uid", index, prompt_id)
            request_id = f"{uid}:{int(rollout_index)}"
        return {
            "global_step": int(global_step),
            "generation_batch_index": int(generation_batch_index),
            "request_order": int(index),
            "dataset_row_id": _dataset_row_id(batch, index),
            "prompt_id": prompt_id,
            "prompt_hash": hashlib.sha256(_canonical_json_bytes(prompt_id)).hexdigest(),
            "prompt_token_hash": prompt_token_hash,
            "prompt_token_count": len(prompt_tokens),
            "rollout_index": int(rollout_index),
            "request_id": str(request_id),
            "dataloader_base_seed": self.config.dataloader_base_seed,
            "sampler_generator_hash": self.config.sampler_generator_hash,
            "rank": int(self.config.rank),
            "ray_actor_identity": self.config.ray_actor_identity,
            "vllm_engine_identity": self.config.vllm_engine_identity,
            "tp_group_identity": self.config.tp_group_identity,
        }

    def _records(
        self,
        boundary: int,
        batch: DataProto,
        *,
        global_step: int,
        generation_batch_index: int,
        rollout_n: int,
        filter_metric: str | None,
        completion_order: Sequence[int] | None,
        effective_training_batch: bool,
    ) -> list[dict[str, Any]]:
        records = [
            self._common_record(
                batch,
                index,
                global_step=global_step,
                generation_batch_index=generation_batch_index,
                rollout_n=rollout_n,
            )
            for index in range(len(batch))
        ]
        if boundary == 0:
            for index, record in enumerate(records):
                seed = _explicit_sampling_seed(batch, index)
                record["assigned_sampling_seed"] = seed
                record["sampling_seed_status"] = (
                    "EXPLICIT_REQUEST_SEED_PRESENT" if seed is not None else "MISSING_EXPLICIT_REQUEST_SEED"
                )
        elif boundary == 1:
            for index, record in enumerate(records):
                response_tokens = _response_tokens(batch, index)
                response_hash = _sha256_tokens(response_tokens)
                record.update(
                    {
                        "response_hash": response_hash,
                        "response_token_ids_hash": response_hash,
                        "response_length": len(response_tokens),
                        "finish_reason": _array_item(batch, "finish_reason", index),
                        "completion_order": int(completion_order[index]) if completion_order is not None else None,
                        "completion_order_status": (
                            "EXPLICIT_COMPLETION_ORDER_PRESENT"
                            if completion_order is not None
                            else "COMPLETION_ORDER_NOT_EXPOSED"
                        ),
                        "worker_identity": _array_item(batch, "worker_identity", index),
                        "engine_identity": _array_item(
                            batch, "vllm_engine_identity", index, self.config.vllm_engine_identity
                        ),
                    }
                )
        elif boundary == 2:
            group_rewards: dict[str, list[float]] = defaultdict(list)
            for index, record in enumerate(records):
                reward = _terminal_reward(batch, index)
                if reward is not None:
                    group_rewards[str(record["prompt_id"])].append(reward)
            for index, record in enumerate(records):
                response_tokens = _response_tokens(batch, index)
                rewards = group_rewards.get(str(record["prompt_id"]), [])
                variance = float(np.var(rewards)) if rewards else None
                record.update(
                    {
                        "response_hash": _sha256_tokens(response_tokens),
                        "terminal_reward": _terminal_reward(batch, index),
                        "reward_error": _array_item(batch, "reward_error", index),
                        "group_reward_pattern": rewards,
                        "group_variance": variance,
                        "candidate_decision": (
                            "keep" if variance is None or variance > 0.0 or len(rewards) == 1 else "drop"
                        ),
                        "filter_metric": filter_metric,
                    }
                )
        elif boundary == 3:
            for index, record in enumerate(records):
                record.update(
                    {
                        "retained_prompt_id": record["prompt_id"],
                        "retained_response_hash": _sha256_tokens(_response_tokens(batch, index)),
                        "keep_drop_decision": "keep",
                        "retained_order": int(index),
                        "effective_training_batch": bool(effective_training_batch),
                    }
                )
        return records

    @staticmethod
    def _audit_unique_identities(records: list[dict[str, Any]]) -> None:
        identities: set[tuple[str, int, int, int]] = set()
        for record in records:
            identity = (
                str(record["request_id"]),
                int(record["global_step"]),
                int(record["generation_batch_index"]),
                int(record["rollout_index"]),
            )
            if identity in identities:
                raise ValueError(f"duplicate sample identity: {identity}")
            identities.add(identity)

    def _target_dir(
        self,
        *,
        boundary: int,
        global_step: int,
        generation_batch_index: int,
        effective_training_batch: bool,
    ) -> Path:
        label = "effective" if boundary == 3 and effective_training_batch else "candidate"
        return (
            Path(str(self.config.output_dir))
            / str(self.config.run_id)
            / f"rank_{int(self.config.rank):05d}"
            / f"step_{int(global_step):06d}"
            / f"generation_batch_{int(generation_batch_index):04d}"
            / f"boundary_{boundary}_{label}"
        )

    def _atomic_commit(self, temporary_dir: Path, target_dir: Path) -> None:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            raise FileExistsError(f"refusing to overwrite nondeterminism diagnostic: {target_dir}")
        os.replace(temporary_dir, target_dir)

    def capture(
        self,
        *,
        boundary: int,
        batch: DataProto,
        global_step: int,
        generation_batch_index: int,
        rollout_n: int,
        filter_metric: str | None = None,
        completion_order: Sequence[int] | None = None,
        effective_training_batch: bool = False,
    ) -> None:
        """Capture one boundary; disabled diagnostics return before any work."""

        if not self.config.enabled:
            return
        if boundary not in BOUNDARY_NAMES:
            raise ValueError(f"invalid nondeterminism diagnostic boundary: {boundary}")
        if int(rollout_n) <= 0:
            raise ValueError("rollout_n must be positive")
        if completion_order is not None and len(completion_order) != len(batch):
            raise ValueError("completion_order length does not match batch")

        before_hash = data_proto_semantic_hash(batch)
        records = self._records(
            boundary,
            batch,
            global_step=global_step,
            generation_batch_index=generation_batch_index,
            rollout_n=rollout_n,
            filter_metric=filter_metric,
            completion_order=completion_order,
            effective_training_batch=effective_training_batch,
        )
        self._audit_unique_identities(records)
        after_record_hash = data_proto_semantic_hash(batch)
        if after_record_hash != before_hash:
            raise AssertionError("nondeterminism instrumentation modified DataProto while building records")

        target_dir = self._target_dir(
            boundary=boundary,
            global_step=global_step,
            generation_batch_index=generation_batch_index,
            effective_training_batch=effective_training_batch,
        )
        temp_parent = target_dir.parent
        temp_parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{target_dir.name}.", suffix=".tmp", dir=temp_parent))
        try:
            records_path = temporary_dir / "records.jsonl"
            with records_path.open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
            after_write_hash = data_proto_semantic_hash(batch)
            if after_write_hash != before_hash:
                raise AssertionError("nondeterminism instrumentation modified DataProto while writing records")
            manifest = {
                "boundary": int(boundary),
                "boundary_name": BOUNDARY_NAMES[boundary],
                "config_sha256": self.config.config_sha256,
                "data_proto_hash_after": after_write_hash,
                "data_proto_hash_before": before_hash,
                "effective_training_batch": bool(effective_training_batch),
                "generation_batch_index": int(generation_batch_index),
                "git_commit": self.config.git_commit,
                "global_step": int(global_step),
                "rank": int(self.config.rank),
                "record_count": len(records),
                "records_sha256": records_sha256,
                "run_id": self.config.run_id,
                "schema_version": SCHEMA_VERSION,
                "status": "COMPLETE",
            }
            manifest_path = temporary_dir / "manifest.json"
            with manifest_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._atomic_commit(temporary_dir, target_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
