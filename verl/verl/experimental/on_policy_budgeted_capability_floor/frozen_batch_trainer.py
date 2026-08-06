"""Distributed one-update adapter for the Gate D v2 frozen-batch harness."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.capability_constraints.identity import reference_model_fingerprint
from verl.experimental.frozen_batch_harness import (
    FrozenBatchHarness,
    FrozenBatchHarnessConfig,
)
from verl.experimental.on_policy_budgeted_capability_floor.dapo_trainer import (
    RayDAPOOnPolicyBudgetedCapabilityFloorTrainer,
)
from verl.experimental.probe_credit.dapo_trainer import (
    RayDAPOProbeCreditTrainer,
    _config_get,
)
from verl.utils.metric import reduce_metrics


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _checkpoint_tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise ValueError(f"initial checkpoint directory does not exist: {path}")
    hasher = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"initial checkpoint directory has no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix()
        hasher.update(relative.encode("utf-8"))
        hasher.update(_sha256_file(item).encode("ascii"))
    return hasher.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    hasher = hashlib.sha256()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    if value.numel():
        hasher.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return hasher.hexdigest()


class FrozenBatchTrainerMixin:
    """Replace the training loop with one verified actor update and no rollout."""

    def _frozen_config(self) -> Any:
        raw = _config_get(_config_get(self.config, "trainer"), "frozen_batch")
        if raw is None:
            raise ValueError("trainer.frozen_batch configuration is required")
        return raw

    def _frozen_mode(self) -> str:
        return str(_config_get(self._frozen_config(), "mode", ""))

    def _frozen_output_dir(self) -> Path:
        value = _config_get(self._frozen_config(), "output_dir")
        if not value:
            raise ValueError("trainer.frozen_batch.output_dir is required")
        return Path(str(value))

    def _frozen_run_id(self) -> str:
        value = _config_get(self._frozen_config(), "run_id")
        if not value:
            raise ValueError("trainer.frozen_batch.run_id is required")
        return str(value)

    def _optimizer_and_scheduler_identities(self) -> tuple[str, str]:
        actor = self.config.actor_rollout_ref.actor
        optimizer_config = OmegaConf.to_container(actor.optim, resolve=True)
        optimizer_identity = _canonical_sha256(optimizer_config)
        scheduler_config = {
            "lr": _config_get(actor.optim, "lr"),
            "lr_scheduler_type": _config_get(actor.optim, "lr_scheduler_type"),
            "lr_warmup_steps": _config_get(actor.optim, "lr_warmup_steps"),
            "lr_warmup_steps_ratio": _config_get(actor.optim, "lr_warmup_steps_ratio"),
            "min_lr_ratio": _config_get(actor.optim, "min_lr_ratio"),
            "num_cycles": _config_get(actor.optim, "num_cycles"),
            "total_training_steps": _config_get(actor.optim, "total_training_steps"),
        }
        return optimizer_identity, _canonical_sha256(scheduler_config)

    def _initial_state_identity(self, checkpoint_path: Path | None = None) -> dict[str, str]:
        if checkpoint_path is None:
            checkpoint_path = Path(str(self.config.trainer.resume_from_path))
        optimizer_identity, scheduler_identity = self._optimizer_and_scheduler_identities()
        base_model_path = getattr(self, "_obcf_base_model_local_path", None)
        if not base_model_path:
            raise ValueError("frozen harness requires the resolved local Base model path")
        return {
            "actor": reference_model_fingerprint(str(base_model_path)),
            "initial_checkpoint": _checkpoint_tree_sha256(checkpoint_path),
            "optimizer": optimizer_identity,
            "scheduler": scheduler_identity,
        }

    def _write_initial_state_manifest(self, checkpoint_path: Path) -> Path:
        raw = self._frozen_config()
        manifest_path_value = _config_get(raw, "initial_state_manifest_path")
        if not manifest_path_value:
            raise ValueError("initial_state_manifest_path is required for preparation")
        manifest_path = Path(str(manifest_path_value))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite initial state manifest: {manifest_path}")
        manifest = {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "identities": self._initial_state_identity(checkpoint_path),
            "schema_version": "obcf-frozen-initial-state-v2",
            "source_git_commit": os.environ.get("OBCF_DIAGNOSTIC_GIT_COMMIT"),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return manifest_path

    def _write_actor_input_manifest(self, batch: DataProto) -> Path:
        output_dir = self._frozen_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self._frozen_run_id()}.actor_input.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite actor-input manifest: {path}")
        tensor_fields = {
            str(name): {
                "dtype": str(tensor.dtype),
                "sha256": _tensor_sha256(tensor),
                "shape": list(tensor.shape),
            }
            for name, tensor in batch.batch.items()
        }
        response_mask = batch.batch.get("response_mask")
        manifest = {
            "actor_input_field_names": list(batch.batch.keys()),
            "non_tensor_field_names": list(batch.non_tensor_batch.keys()),
            "row_count": len(batch),
            "schema_version": "obcf-frozen-actor-input-v2",
            "tensor_fields": tensor_fields,
            "valid_response_token_count": (
                int(response_mask.sum().item()) if response_mask is not None else None
            ),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=output_dir
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    def _update_actor(self, batch: DataProto) -> DataProto:
        count = int(getattr(self, "_frozen_actor_update_count", 0))
        if count != 0:
            raise RuntimeError("frozen-batch harness attempted more than one actor update")
        self._frozen_actor_update_count = 1
        self._frozen_actor_input_manifest = self._write_actor_input_manifest(batch)
        return super()._update_actor(batch)

    def _run_frozen_actor_update(self, batch: DataProto, mode: str) -> dict[str, Any]:
        if mode != self._frozen_mode():
            raise ValueError(f"harness mode {mode!r} does not match trainer mode {self._frozen_mode()!r}")
        terminal_scores = batch.batch.get("token_level_scores")
        terminal_scores_before = terminal_scores.detach().clone() if terminal_scores is not None else None
        metrics: dict[str, float] = {}
        timing_raw: dict[str, float] = {}
        lambda_before = float(getattr(self, "_lambda", 0.0))
        constraint_updates_before = int(getattr(self, "_constraint_observation_count", 0))
        started = time.perf_counter()
        batch, actor_output = self._compute_advantage_and_actor_update(batch, metrics, timing_raw)
        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
        update_seconds = time.perf_counter() - started
        terminal_reward_unchanged = terminal_scores_before is None or torch.equal(
            terminal_scores_before, batch.batch.get("token_level_scores")
        )
        if not terminal_reward_unchanged:
            raise AssertionError("frozen-batch actor path modified terminal reward scores")

        self.global_steps = 1
        self._save_checkpoint()
        constraint_updates_after = int(getattr(self, "_constraint_observation_count", 0))
        prefix_calls = int(metrics.get("obcf/prefix_verifier_calls", 0.0))
        shadow_diagnostics = (
            {key: value for key, value in metrics.items() if key.startswith("obcf/")}
            if mode == "shadow"
            else None
        )
        return {
            "actor_input_manifest": str(self._frozen_actor_input_manifest.resolve()),
            "actor_update_count": int(self._frozen_actor_update_count),
            "cache_loaded_count": int(getattr(self, "_obcf_cache", None) is not None),
            "constraint_observation_count": int(prefix_calls > 0),
            "lambda_after": float(getattr(self, "_lambda", 0.0)),
            "lambda_before": lambda_before,
            "lambda_update_count": constraint_updates_after - constraint_updates_before,
            "metrics": metrics,
            "post_update_checkpoint": str(
                (Path(str(self.config.trainer.default_local_dir)) / "global_step_1").resolve()
            ),
            "prefix_verifier_calls": prefix_calls,
            "rollout_request_count": 0,
            "shadow_diagnostics": shadow_diagnostics,
            "terminal_reward_exactly_unchanged": terminal_reward_unchanged,
            "timing_raw": timing_raw,
            "update_wall_seconds": update_seconds,
            "verifier_error_count": 0,
            "verifier_timeout_count": 0,
        }

    def _prepare_initial_state(self) -> None:
        if self.config.trainer.resume_mode != "disable":
            raise ValueError("initial frozen state preparation requires trainer.resume_mode=disable")
        self.global_steps = 0
        self._save_checkpoint()
        checkpoint_path = Path(str(self.config.trainer.default_local_dir)) / "global_step_0"
        self._write_initial_state_manifest(checkpoint_path)

    def fit(self):
        """Prepare an initial state or replay exactly one frozen actor update."""

        self._validate_probe_credit_mode()
        raw = self._frozen_config()
        self._frozen_actor_update_count = 0
        if bool(_config_get(raw, "prepare_initial_state", False)):
            self._prepare_initial_state()
            self._shutdown_dump_executor()
            return

        if self.config.trainer.resume_mode != "resume_path":
            raise ValueError("frozen-batch runs require trainer.resume_mode=resume_path")
        self.global_steps = 0
        self._load_checkpoint()
        if int(self.global_steps) != 0:
            raise ValueError("frozen-batch initial checkpoint must have global step zero")

        self._save_checkpoint()
        core = FrozenBatchHarness(
            FrozenBatchHarnessConfig(
                mode=self._frozen_mode(),
                frozen_batch_path=_config_get(raw, "batch_path"),
                frozen_batch_manifest_path=_config_get(raw, "batch_manifest_path"),
                initial_state_manifest_path=_config_get(raw, "initial_state_manifest_path"),
                output_dir=self._frozen_output_dir(),
                run_id=self._frozen_run_id(),
                protocol_fingerprint=str(_config_get(raw, "protocol_fingerprint")),
                cache_fingerprint=(
                    getattr(getattr(self, "_obcf_cache", None), "fingerprint", None)
                    if self._frozen_mode() == "shadow"
                    else None
                ),
            ),
            initial_state_identity=self._initial_state_identity,
            actor_update=self._run_frozen_actor_update,
        )
        core.run()
        self._shutdown_dump_executor()


class RayDAPOFrozenBatchBaselineTrainer(FrozenBatchTrainerMixin, RayDAPOProbeCreditTrainer):
    """Baseline entrypoint adapter for one frozen update."""


class RayDAPOFrozenBatchOBCFTrainer(
    FrozenBatchTrainerMixin, RayDAPOOnPolicyBudgetedCapabilityFloorTrainer
):
    """OBCF off/shadow entrypoint adapter for one frozen update."""
