"""Strict prompt-level cache for offline Base capability floors."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from verl.experimental.capability_constraints.identity import canonical_prompt_key
from verl.experimental.on_policy_budgeted_capability_floor.math import compute_capability_floor

SCHEMA_VERSION = 2
ALGORITHM = "on_policy_budgeted_capability_floor"

PROMPT_SCHEMA = pa.schema(
    [
        ("prompt_key", pa.string()),
        ("prompt_id", pa.int64()),
        ("original_dataset_index", pa.int64()),
        ("prompt_hash", pa.string()),
        ("prompt_token_ids", pa.list_(pa.int32())),
        ("prompt_token_count", pa.int32()),
        ("base_rollout_count", pa.int16()),
        ("base_prefix_success_count", pa.int16()),
        ("q_reference", pa.float32()),
        ("floor_count", pa.int16()),
        ("capability_floor", pa.float32()),
    ]
)

REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "algorithm",
        "reference_model_id",
        "reference_model_hash",
        "reference_budget",
        "base_rollouts_per_prompt",
        "support_threshold",
        "reference_tolerance_count",
        "prefix_reward_field",
        "tokenizer_fingerprint",
        "chat_template_fingerprint",
        "prompt_manifest_fingerprint",
        "rollout_fingerprint",
        "score_fingerprint",
        "verifier_fingerprint",
        "prefix_protocol_fingerprint",
        "created_at",
        "source_git_commit",
        "prompt_count",
    }
)


@dataclass(frozen=True)
class CacheExpectations:
    reference_budget: int
    base_rollouts_per_prompt: int
    support_threshold: int
    reference_tolerance_count: int
    tokenizer_fingerprint: str
    chat_template_fingerprint: str
    verifier_fingerprint: str
    prefix_protocol_fingerprint: str


class _FloorRows(list[dict[str, Any]]):
    def __init__(self, rows: Iterable[dict[str, Any]], *, histogram: Counter[int]) -> None:
        super().__init__(rows)
        self.success_count_histogram = dict(histogram)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=PROMPT_SCHEMA), temporary)
    os.replace(temporary, path)


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    try:
        model_id = str(row["model_id"])
        prompt_id = int(row["prompt_id"])
        rollout_index = int(row["rollout_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("rollout identity requires model_id, prompt_id, and rollout_index") from error
    return model_id, prompt_id, rollout_index


def _unique_by_identity(rows: Iterable[Mapping[str, Any]], name: str) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        identity = _identity(row)
        if identity in indexed:
            raise ValueError(f"duplicate {name} identity {identity}")
        indexed[identity] = row
    return indexed


def build_floor_rows(
    *,
    prompts: Iterable[Mapping[str, Any]],
    rollouts: Iterable[Mapping[str, Any]],
    scores: Iterable[Mapping[str, Any]],
    model_id: str,
    tokenizer_fingerprint: str,
    chat_template_fingerprint: str,
    reference_budget: int,
    base_rollouts_per_prompt: int,
    support_threshold: int,
    reference_tolerance_count: int,
    verifier_fingerprint: str | None = None,
    source_git_commit: str | None = None,
    require_prompt_provenance: bool = False,
) -> list[dict[str, Any]]:
    """Strictly join Base audit artifacts and retain only supported prompts."""
    if not model_id:
        raise ValueError("model_id must be nonempty")
    if reference_budget <= 0:
        raise ValueError("reference_budget must be positive")
    if base_rollouts_per_prompt <= 0 or not 0 < support_threshold <= base_rollouts_per_prompt:
        raise ValueError("invalid rollout count or support_threshold")
    if not 0 <= reference_tolerance_count <= base_rollouts_per_prompt:
        raise ValueError("reference_tolerance_count must be within the rollout count")
    prefix_field = f"prefix_reward_{reference_budget}"
    error_field = f"prefix_error_{reference_budget}"
    rollout_by_id = _unique_by_identity(rollouts, "rollout")
    score_by_id = _unique_by_identity(scores, "score")
    if rollout_by_id.keys() != score_by_id.keys():
        raise ValueError("rollout and score identities/counts do not match")

    prompt_rows = list(prompts)
    prompt_ids = [int(row["prompt_id"]) for row in prompt_rows]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("duplicate prompt identity")
    known_ids = set(prompt_ids)
    foreign = [identity for identity in rollout_by_id if identity[0] != model_id or identity[1] not in known_ids]
    if foreign:
        raise ValueError("rollout identity references the wrong model or an unknown prompt")

    protected: list[dict[str, Any]] = []
    histogram: Counter[int] = Counter()
    for prompt in prompt_rows:
        prompt_id = int(prompt["prompt_id"])
        if require_prompt_provenance and (
            prompt.get("tokenizer_fingerprint") != tokenizer_fingerprint
            or prompt.get("chat_template_fingerprint") != chat_template_fingerprint
        ):
            raise ValueError("prompt tokenizer/chat-template provenance does not match")
        identities = sorted(
            (identity for identity in rollout_by_id if identity[:2] == (model_id, prompt_id)),
            key=lambda identity: identity[2],
        )
        if len(identities) != base_rollouts_per_prompt or [item[2] for item in identities] != list(
            range(base_rollouts_per_prompt)
        ):
            raise ValueError(f"prompt {prompt_id} base_rollout_count is not exact")
        successes = 0
        for identity in identities:
            rollout = rollout_by_id[identity]
            score = score_by_id[identity]
            if str(rollout.get("prompt_hash", "")) != str(prompt.get("prompt_hash", "")):
                raise ValueError("rollout prompt identity/hash mismatch")
            if [int(token) for token in rollout.get("prompt_token_ids", [])] != [
                int(token) for token in prompt["prompt_token_ids"]
            ]:
                raise ValueError("rollout prompt token identity mismatch")
            if str(score.get("prompt_hash", "")) != str(prompt.get("prompt_hash", "")):
                raise ValueError("score prompt identity/hash mismatch")
            for matched_field in ("sampling_seed", "response_token_count", "response_hash"):
                if matched_field in rollout or matched_field in score:
                    if rollout.get(matched_field) != score.get(matched_field):
                        raise ValueError(f"rollout/score {matched_field} mismatch")
            if prefix_field not in score:
                raise ValueError(f"score is missing {prefix_field}")
            if not isinstance(score[prefix_field], bool):
                raise ValueError(f"{prefix_field} must be boolean")
            if error_field not in score or score[error_field] is not None:
                raise ValueError(f"{error_field} must be present and explicitly empty")
            if verifier_fingerprint is not None and score.get("verifier_fingerprint") != verifier_fingerprint:
                raise ValueError("score verifier_fingerprint does not match the audited pipeline")
            if source_git_commit is not None and score.get("source_git_commit") != source_git_commit:
                raise ValueError("score source_git_commit does not match the audit provenance")
            successes += int(bool(score[prefix_field]))
        histogram[successes] += 1
        if successes < support_threshold:
            continue
        tokens = [int(token) for token in prompt["prompt_token_ids"]]
        if len(tokens) != int(prompt["prompt_token_count"]):
            raise ValueError("prompt_token_count does not match prompt_token_ids")
        floor_count = max(successes - reference_tolerance_count, 0)
        protected.append(
            {
                "prompt_key": canonical_prompt_key(
                    tokenizer_fingerprint, chat_template_fingerprint, tokens
                ),
                "prompt_id": prompt_id,
                "original_dataset_index": int(prompt["original_dataset_index"]),
                "prompt_hash": str(prompt["prompt_hash"]),
                "prompt_token_ids": tokens,
                "prompt_token_count": len(tokens),
                "base_rollout_count": base_rollouts_per_prompt,
                "base_prefix_success_count": successes,
                "q_reference": successes / base_rollouts_per_prompt,
                "floor_count": floor_count,
                "capability_floor": compute_capability_floor(
                    base_success_count=successes,
                    base_rollout_count=base_rollouts_per_prompt,
                    tolerance_count=reference_tolerance_count,
                ),
            }
        )
    return _FloorRows(protected, histogram=histogram)


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        raise ValueError(f"cache manifest is missing required fields {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["algorithm"] != ALGORITHM:
        raise ValueError("unsupported OBCF cache schema or algorithm")
    model_hash = manifest["reference_model_hash"]
    if (
        not isinstance(model_hash, str)
        or len(model_hash) != 64
        or any(char not in "0123456789abcdef" for char in model_hash)
    ):
        raise ValueError("reference_model_hash must be a lowercase SHA-256 digest")
    for name in (
        "reference_model_id",
        "tokenizer_fingerprint",
        "chat_template_fingerprint",
        "prompt_manifest_fingerprint",
        "rollout_fingerprint",
        "score_fingerprint",
        "verifier_fingerprint",
        "prefix_protocol_fingerprint",
        "created_at",
        "source_git_commit",
    ):
        if not isinstance(manifest[name], str) or not manifest[name]:
            raise ValueError(f"{name} must be nonempty")
    source_commit = manifest["source_git_commit"]
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ValueError("source_git_commit must be a full lowercase Git commit digest")
    for name in (
        "prompt_manifest_fingerprint",
        "rollout_fingerprint",
        "score_fingerprint",
        "verifier_fingerprint",
        "prefix_protocol_fingerprint",
    ):
        digest = manifest[name]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    for name in ("reference_budget", "base_rollouts_per_prompt", "support_threshold", "prompt_count"):
        value = manifest[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    n = manifest["base_rollouts_per_prompt"]
    if manifest["support_threshold"] > n:
        raise ValueError("support_threshold cannot exceed base_rollouts_per_prompt")
    tolerance = manifest["reference_tolerance_count"]
    if not isinstance(tolerance, int) or isinstance(tolerance, bool) or not 0 <= tolerance <= n:
        raise ValueError("reference_tolerance_count must be within the rollout count")
    if manifest["prefix_reward_field"] != f"prefix_reward_{manifest['reference_budget']}":
        raise ValueError("prefix_reward_field does not match reference_budget")


def _validate_rows(manifest: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    keys: set[str] = set()
    prompt_ids: set[int] = set()
    dataset_indices: set[int] = set()
    prompt_hashes: set[str] = set()
    n = int(manifest["base_rollouts_per_prompt"])
    threshold = int(manifest["support_threshold"])
    tolerance = int(manifest["reference_tolerance_count"])
    for row in normalized:
        key = str(row["prompt_key"])
        prompt_id = int(row["prompt_id"])
        dataset_index = int(row["original_dataset_index"])
        prompt_hash = str(row["prompt_hash"])
        if (
            key in keys
            or prompt_id in prompt_ids
            or dataset_index in dataset_indices
            or prompt_hash in prompt_hashes
        ):
            raise ValueError("duplicate prompt identity")
        keys.add(key)
        prompt_ids.add(prompt_id)
        dataset_indices.add(dataset_index)
        prompt_hashes.add(prompt_hash)
        tokens = [int(token) for token in row["prompt_token_ids"]]
        expected_key = canonical_prompt_key(
            str(manifest["tokenizer_fingerprint"]),
            str(manifest["chat_template_fingerprint"]),
            tokens,
        )
        if key != expected_key or len(tokens) != int(row["prompt_token_count"]):
            raise ValueError("prompt identity/token count mismatch")
        count = int(row["base_rollout_count"])
        successes = int(row["base_prefix_success_count"])
        if count != n:
            raise ValueError("base_rollout_count does not match manifest")
        if not threshold <= successes <= count:
            raise ValueError("base_prefix_success_count is below support_threshold or invalid")
        floor_count = max(successes - tolerance, 0)
        if not math.isclose(float(row["q_reference"]), successes / count, abs_tol=1e-6):
            raise ValueError("q_reference does not match counts")
        if int(row["floor_count"]) != floor_count:
            raise ValueError("floor_count does not match counts")
        if not math.isclose(float(row["capability_floor"]), floor_count / count, abs_tol=1e-6):
            raise ValueError("capability_floor does not match floor_count")
    return normalized


def write_cache(
    root: str | Path,
    manifest: Mapping[str, Any],
    prompts: Iterable[Mapping[str, Any]],
) -> str:
    """Validate and atomically write a complete OBCF cache."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    normalized_manifest = dict(manifest)
    normalized_manifest.setdefault("schema_version", SCHEMA_VERSION)
    normalized_manifest.setdefault("algorithm", ALGORITHM)
    _validate_manifest(normalized_manifest)
    histogram = getattr(prompts, "success_count_histogram", None)
    rows = _validate_rows(normalized_manifest, prompts)
    if not rows:
        raise ValueError("OBCF cache must contain at least one protected prompt")
    normalized_manifest["protected_prompt_count"] = len(rows)
    if int(normalized_manifest["prompt_count"]) < len(rows):
        raise ValueError("prompt_count cannot be less than protected_prompt_count")
    if histogram is None and int(normalized_manifest["prompt_count"]) != len(rows):
        raise ValueError("a complete success-count histogram is required for the audit report")

    manifest_path = root / "manifest.json"
    prompts_path = root / "prompts.parquet"
    _atomic_bytes(manifest_path, _canonical_json(normalized_manifest) + b"\n")
    _atomic_parquet(prompts_path, rows)
    core_hashes = {
        "manifest.json": _sha256_file(manifest_path),
        "prompts.parquet": _sha256_file(prompts_path),
    }
    fingerprint = _sha256_bytes(_canonical_json(core_hashes))
    if histogram is None:
        histogram = Counter(int(row["base_prefix_success_count"]) for row in rows)
    if sum(histogram.values()) != int(normalized_manifest["prompt_count"]):
        raise ValueError("success-count histogram does not match prompt_count")
    floor_histogram = Counter(int(row["floor_count"]) for row in rows)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "cache_fingerprint": fingerprint,
        "prompt_count": int(normalized_manifest["prompt_count"]),
        "protected_prompt_count": len(rows),
        "base_prefix_success_count_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
        "protected_floor_count_histogram": {
            str(key): int(value) for key, value in sorted(floor_histogram.items())
        },
    }
    audit_path = root / "audit_report.json"
    _atomic_bytes(audit_path, _canonical_json(audit) + b"\n")
    file_hashes = core_hashes | {"audit_report.json": _sha256_file(audit_path)}
    _atomic_bytes(
        root / "hashes.json",
        _canonical_json({"cache_fingerprint": fingerprint, "files": file_hashes}) + b"\n",
    )
    return fingerprint


class CapabilityFloorCache:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        prompts: list[dict[str, Any]],
        audit_report: dict[str, Any],
        fingerprint: str,
    ) -> None:
        self.manifest = manifest
        self.prompts = prompts
        self.audit_report = audit_report
        self.fingerprint = fingerprint
        self._by_key = {row["prompt_key"]: row for row in prompts}

    def get(self, prompt_key: str) -> dict[str, Any] | None:
        row = self._by_key.get(prompt_key)
        return None if row is None else dict(row)

    @classmethod
    def load(cls, root: str | Path, expectations: CacheExpectations) -> "CapabilityFloorCache":
        root = Path(root)
        required = ("manifest.json", "prompts.parquet", "audit_report.json", "hashes.json")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(f"cache is incomplete; missing {missing}")
        manifest = json.loads((root / "manifest.json").read_text())
        audit = json.loads((root / "audit_report.json").read_text())
        hashes = json.loads((root / "hashes.json").read_text())
        _validate_manifest(manifest)
        actual_hashes = {name: _sha256_file(root / name) for name in required[:-1]}
        if hashes.get("files") != actual_hashes:
            raise ValueError("cache file hash mismatch")
        fingerprint = _sha256_bytes(
            _canonical_json({name: actual_hashes[name] for name in ("manifest.json", "prompts.parquet")})
        )
        if hashes.get("cache_fingerprint") != fingerprint or audit.get("cache_fingerprint") != fingerprint:
            raise ValueError("cache fingerprint mismatch")
        if audit.get("passed") is not True:
            raise ValueError("audit_report passed must be true")
        if audit.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("audit_report schema_version mismatch")
        for field in (
            "reference_budget",
            "base_rollouts_per_prompt",
            "support_threshold",
            "reference_tolerance_count",
            "tokenizer_fingerprint",
            "chat_template_fingerprint",
            "verifier_fingerprint",
            "prefix_protocol_fingerprint",
        ):
            expected = getattr(expectations, field)
            if manifest.get(field) != expected:
                raise ValueError(f"cache {field} mismatch: expected {expected!r}, got {manifest.get(field)!r}")
        table = pq.read_table(root / "prompts.parquet")
        if table.schema != PROMPT_SCHEMA:
            raise ValueError("prompts.parquet schema mismatch")
        prompts = _validate_rows(manifest, table.to_pylist())
        if manifest.get("protected_prompt_count") != len(prompts):
            raise ValueError("protected_prompt_count does not match prompts")
        if audit.get("prompt_count") != manifest.get("prompt_count") or audit.get(
            "protected_prompt_count"
        ) != len(prompts):
            raise ValueError("audit_report counts do not match cache contents")
        histogram = audit.get("base_prefix_success_count_histogram")
        if not isinstance(histogram, dict):
            raise ValueError("audit_report success-count histogram is missing")
        try:
            count_histogram = {int(key): int(value) for key, value in histogram.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("audit_report success-count histogram is malformed") from error
        n = int(manifest["base_rollouts_per_prompt"])
        if (
            any(key < 0 or key > n or value < 0 for key, value in count_histogram.items())
            or sum(count_histogram.values()) != int(manifest["prompt_count"])
            or sum(
                value
                for key, value in count_histogram.items()
                if key >= int(manifest["support_threshold"])
            )
            != len(prompts)
        ):
            raise ValueError("audit_report success-count histogram is inconsistent")
        expected_protected_histogram = Counter(
            int(row["base_prefix_success_count"]) for row in prompts
        )
        observed_protected_histogram = Counter(
            {
                key: value
                for key, value in count_histogram.items()
                if key >= int(manifest["support_threshold"]) and value
            }
        )
        if observed_protected_histogram != expected_protected_histogram:
            raise ValueError("audit_report success-count histogram does not match prompt rows")
        expected_floor_histogram = Counter(int(row["floor_count"]) for row in prompts)
        if audit.get("protected_floor_count_histogram") != {
            str(key): int(value) for key, value in sorted(expected_floor_histogram.items())
        }:
            raise ValueError("audit_report floor-count histogram is inconsistent")
        return cls(manifest=manifest, prompts=prompts, audit_report=audit, fingerprint=fingerprint)
