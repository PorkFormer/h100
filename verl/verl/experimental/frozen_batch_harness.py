"""Fail-closed core for one-update frozen DataProto diagnostics.

The core deliberately has no rollout or dynamic-filtering dependency.  A real
trainer adapter supplies the initial-state identity and exactly-one actor update;
CPU tests can supply small fakes without weakening the artifact checks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from verl import DataProto
from verl.experimental.nondeterminism_diagnostics import data_proto_semantic_hash


VALID_MODES = frozenset({"baseline", "off", "shadow"})
CAPABILITY_FIELD_PREFIXES = (
    "capability",
    "obcf_",
    "terminal_advantages",
)


@dataclasses.dataclass(frozen=True)
class FrozenBatchHarnessConfig:
    mode: str
    frozen_batch_path: Path | str
    frozen_batch_manifest_path: Path | str
    initial_state_manifest_path: Path | str
    output_dir: Path | str
    run_id: str
    protocol_fingerprint: str
    cache_fingerprint: str | None = None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class FrozenBatchHarness:
    """Verify immutable inputs and execute one injected actor-update operation."""

    def __init__(
        self,
        config: FrozenBatchHarnessConfig,
        *,
        initial_state_identity: Callable[[], dict[str, Any]],
        actor_update: Callable[[DataProto, str], dict[str, Any]],
    ):
        self.config = config
        self._initial_state_identity = initial_state_identity
        self._actor_update = actor_update

    def _validate_mode(self) -> None:
        if self.config.mode not in VALID_MODES:
            raise ValueError(
                f"invalid frozen-batch mode {self.config.mode!r}; expected one of {sorted(VALID_MODES)}"
            )
        if self.config.mode == "shadow" and not self.config.cache_fingerprint:
            raise ValueError("shadow frozen-batch mode requires an exact cache fingerprint")
        if not self.config.run_id:
            raise ValueError("frozen-batch run_id must not be empty")
        if not self.config.protocol_fingerprint:
            raise ValueError("frozen-batch protocol fingerprint must not be empty")

    @staticmethod
    def _require_manifest_fields(manifest: dict[str, Any]) -> None:
        required = {
            "file_sha256",
            "semantic_sha256",
            "row_count",
            "source_commit",
            "source_config_sha256",
            "source_checkpoint_fingerprint",
        }
        missing = sorted(required - manifest.keys())
        if missing:
            raise ValueError(f"frozen batch manifest missing fields: {missing}")

    def _verify_and_load_batch(self) -> tuple[DataProto, dict[str, Any], str]:
        batch_path = Path(self.config.frozen_batch_path)
        manifest = _load_json_object(
            Path(self.config.frozen_batch_manifest_path), "frozen batch manifest"
        )
        self._require_manifest_fields(manifest)
        file_sha256 = _sha256_file(batch_path)
        if file_sha256 != manifest["file_sha256"]:
            raise ValueError(
                "frozen batch file hash mismatch: "
                f"expected {manifest['file_sha256']}, observed {file_sha256}"
            )
        batch = DataProto.load_from_disk(batch_path)
        semantic_sha256 = data_proto_semantic_hash(batch)
        if semantic_sha256 != manifest["semantic_sha256"]:
            raise ValueError(
                "frozen batch semantic hash mismatch: "
                f"expected {manifest['semantic_sha256']}, observed {semantic_sha256}"
            )
        if len(batch) != int(manifest["row_count"]):
            raise ValueError("frozen batch row count mismatch")
        return batch, manifest, file_sha256

    def _verify_initial_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = _load_json_object(
            Path(self.config.initial_state_manifest_path), "initial state manifest"
        )
        expected = manifest.get("identities")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("initial state manifest requires nonempty identities")
        observed = _jsonable(self._initial_state_identity())
        if observed != expected:
            raise ValueError(
                "initial state identity mismatch: "
                f"expected {json.dumps(expected, sort_keys=True)}, "
                f"observed {json.dumps(observed, sort_keys=True)}"
            )
        return expected, observed

    @staticmethod
    def _capability_fields(batch: DataProto) -> list[str]:
        fields = [str(name) for name in batch.batch.keys()]
        fields.extend(str(name) for name in batch.non_tensor_batch.keys())
        return sorted(
            name for name in fields if name.startswith(CAPABILITY_FIELD_PREFIXES)
        )

    def _validate_update_result(
        self,
        result: dict[str, Any],
        *,
        capability_fields: list[str],
        total_advantage_unchanged: bool,
    ) -> None:
        if int(result.get("actor_update_count", -1)) != 1:
            raise ValueError("frozen-batch harness requires exactly one actor update")
        if int(result.get("rollout_request_count", -1)) != 0:
            raise ValueError("frozen-batch harness must not request rollout")
        if self.config.mode == "off":
            exact_zero_fields = (
                "cache_loaded_count",
                "prefix_verifier_calls",
                "constraint_observation_count",
                "lambda_update_count",
            )
            nonzero = {
                name: result.get(name)
                for name in exact_zero_fields
                if int(result.get(name, -1)) != 0
            }
            if nonzero:
                raise ValueError(f"off mode has OBCF side effects: {nonzero}")
            if capability_fields:
                raise ValueError(f"off mode added capability batch fields: {capability_fields}")
        if self.config.mode == "shadow":
            if float(result.get("lambda_before", float("nan"))) != 0.0:
                raise ValueError("shadow lambda_before must be exactly zero")
            if float(result.get("lambda_after", float("nan"))) != 0.0:
                raise ValueError("shadow lambda_after must be exactly zero")
            if int(result.get("lambda_update_count", -1)) != 0:
                raise ValueError("shadow must not update the dual")
            if not total_advantage_unchanged:
                raise ValueError("shadow modified total advantage")
            if not isinstance(result.get("shadow_diagnostics"), dict):
                raise ValueError("shadow prefix diagnostics are missing")

    def _write_result(self, result: dict[str, Any]) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{self.config.run_id}.result.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite frozen-batch result: {result_path}")
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.config.run_id}.", suffix=".tmp", dir=output_dir
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, result_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return result_path

    def run(self) -> dict[str, Any]:
        """Run once, rejecting any invalid identity/count/side-effect evidence."""

        self._validate_mode()
        batch, batch_manifest, input_file_sha256 = self._verify_and_load_batch()
        expected_initial, observed_initial = self._verify_initial_state()
        total_advantage_before = batch.batch.get("advantages")
        if total_advantage_before is None:
            raise ValueError("frozen batch is missing advantages required by actor update")
        total_advantage_before = total_advantage_before.detach().clone()

        update_result = self._actor_update(batch, self.config.mode)
        if not isinstance(update_result, dict):
            raise TypeError("actor_update must return a result dictionary")
        total_advantage_after = batch.batch.get("advantages")
        total_advantage_unchanged = total_advantage_after is not None and torch.equal(
            total_advantage_before, total_advantage_after
        )
        capability_fields = self._capability_fields(batch)
        self._validate_update_result(
            update_result,
            capability_fields=capability_fields,
            total_advantage_unchanged=total_advantage_unchanged,
        )

        final_file_sha256 = _sha256_file(Path(self.config.frozen_batch_path))
        frozen_input_unchanged = final_file_sha256 == input_file_sha256
        if not frozen_input_unchanged:
            raise ValueError("actor update modified the frozen input artifact")

        result = {
            **_jsonable(update_result),
            "cache_fingerprint": self.config.cache_fingerprint,
            "capability_batch_field_count": len(capability_fields),
            "capability_batch_fields": capability_fields,
            "decision": "PASS",
            "frozen_batch_file_sha256": input_file_sha256,
            "frozen_batch_semantic_sha256": batch_manifest["semantic_sha256"],
            "frozen_input_file_unchanged": frozen_input_unchanged,
            "initial_state_expected": expected_initial,
            "initial_state_observed": observed_initial,
            "mode": self.config.mode,
            "protocol_fingerprint": self.config.protocol_fingerprint,
            "run_id": self.config.run_id,
            "schema_version": "obcf-frozen-batch-harness-result-v2",
            "total_advantage_exactly_unchanged": total_advantage_unchanged,
        }
        result_path = self._write_result(result)
        result["result_path"] = str(result_path.resolve())
        return result
