"""Strict on-disk contract and stateless sampler for BSSF witness caches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from verl.experimental.capability_constraints.identity import (
    canonical_prompt_key,
    reference_model_fingerprint,
    tokenizer_fingerprints,
)

SCHEMA_VERSION = 1
ALGORITHM = "budgeted_success_support_floor"
NATURAL_FINISH_REASONS = frozenset({"eos", "stop", "stop_sequence"})
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "algorithm",
        "reference_model_id",
        "reference_model_hash",
        "reference_budget",
        "base_rollouts_per_prompt",
        "support_threshold",
        "tokenizer_fingerprint",
        "chat_template_fingerprint",
        "prompt_manifest_fingerprint",
        "verifier_fingerprint",
        "logprob_temperature",
        "logprob_convention",
        "include_eos",
        "created_at",
        "source_git_commit",
    }
)

PROMPT_SCHEMA = pa.schema(
    [
        ("prompt_key", pa.string()),
        ("prompt_id", pa.int64()),
        ("original_dataset_index", pa.int64()),
        ("prompt_hash", pa.string()),
        ("prompt_token_ids", pa.list_(pa.int32())),
        ("prompt_token_count", pa.int32()),
        ("base_rollout_count", pa.int16()),
        ("eligible_success_count", pa.int16()),
        ("q_reference", pa.float32()),
    ]
)

WITNESS_SCHEMA = pa.schema(
    [
        ("prompt_key", pa.string()),
        ("witness_id", pa.int16()),
        ("source_rollout_index", pa.int16()),
        ("response_token_ids", pa.list_(pa.int32())),
        ("response_token_count", pa.int32()),
        ("reference_seq_logprob", pa.float64()),
        ("reference_mean_logprob", pa.float32()),
        ("finish_reason", pa.string()),
        ("full_reward", pa.bool_()),
        ("prefix_reward_reference_budget", pa.bool_()),
        ("response_hash", pa.string()),
    ]
)


@dataclass(frozen=True)
class CacheExpectations:
    reference_budget: int
    support_threshold: int
    tokenizer_fingerprint: str
    chat_template_fingerprint: str
    logprob_temperature: float
    include_eos: bool = True


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        raise ValueError(f"cache manifest is missing required fields {missing}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {manifest['schema_version']}")
    if manifest["algorithm"] != ALGORITHM:
        raise ValueError(f"unexpected cache algorithm {manifest['algorithm']!r}")
    if not str(manifest["reference_model_id"]):
        raise ValueError("reference_model_id must be nonempty")
    model_hash = str(manifest["reference_model_hash"])
    if len(model_hash) != 64 or any(character not in "0123456789abcdef" for character in model_hash):
        raise ValueError("reference_model_hash must be a lowercase SHA-256 digest")
    for field in ("reference_budget", "base_rollouts_per_prompt", "support_threshold"):
        value = manifest[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"cache {field} must be a positive integer")
    temperature = float(manifest["logprob_temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("cache logprob_temperature must be finite and positive")
    if manifest["logprob_convention"] != "response-token-sum":
        raise ValueError("cache logprob_convention must be response-token-sum")
    if manifest["include_eos"] is not True:
        raise ValueError("the first BSSF cache schema requires include_eos=true")
    for field in (
        "tokenizer_fingerprint",
        "chat_template_fingerprint",
        "prompt_manifest_fingerprint",
        "verifier_fingerprint",
        "created_at",
        "source_git_commit",
    ):
        if not str(manifest[field]):
            raise ValueError(f"cache {field} must be nonempty")


def witness_is_eligible(
    *,
    full_reward: bool,
    prefix_reward: bool,
    response_token_count: int,
    reference_budget: int,
    hit_token_cap: bool,
    finish_reason: str,
    generation_error: Any,
    verifier_error: Any,
) -> bool:
    """Apply the complete verifier-certified natural-completion witness gate."""
    return bool(
        full_reward
        and prefix_reward
        and 0 < response_token_count <= reference_budget
        and not hit_token_cap
        and finish_reason in NATURAL_FINISH_REASONS
        and not generation_error
        and not verifier_error
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validated_rows(
    manifest: Mapping[str, Any],
    prompts: Iterable[Mapping[str, Any]],
    witnesses: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_manifest(manifest)
    prompt_rows = sorted((dict(row) for row in prompts), key=lambda row: row["prompt_key"])
    witness_rows = sorted(
        (dict(row) for row in witnesses), key=lambda row: (row["prompt_key"], row["witness_id"])
    )
    if not prompt_rows or not witness_rows:
        raise ValueError("cache must contain protected prompts and witnesses")
    threshold = int(manifest["support_threshold"])
    budget = int(manifest["reference_budget"])
    prompt_keys = {row["prompt_key"] for row in prompt_rows}
    if len(prompt_keys) != len(prompt_rows):
        raise ValueError("prompt_key values must be unique")
    witness_counts = {key: 0 for key in prompt_keys}
    witness_ids: set[tuple[str, int]] = set()
    for row in witness_rows:
        key = row["prompt_key"]
        if key not in prompt_keys:
            raise ValueError(f"witness references unknown prompt_key {key}")
        identity = (key, int(row["witness_id"]))
        if identity in witness_ids:
            raise ValueError(f"duplicate witness identity {identity}")
        witness_ids.add(identity)
        witness_counts[key] += 1
        seq_logprob = float(row["reference_seq_logprob"])
        if not math.isfinite(seq_logprob):
            raise ValueError("reference_seq_logprob must be finite")
        token_count = int(row["response_token_count"])
        if token_count <= 0 or token_count > budget:
            raise ValueError("witness response_token_count exceeds reference budget")
        if token_count != len(row["response_token_ids"]):
            raise ValueError("witness response_token_count does not match token IDs")
        if not row["full_reward"] or not row["prefix_reward_reference_budget"]:
            raise ValueError("every cached witness must have positive full and prefix verifier rewards")
        if row["finish_reason"] not in NATURAL_FINISH_REASONS:
            raise ValueError("every cached witness must finish naturally")
    for row in prompt_rows:
        key = row["prompt_key"]
        expected_key = canonical_prompt_key(
            str(manifest["tokenizer_fingerprint"]),
            str(manifest["chat_template_fingerprint"]),
            row["prompt_token_ids"],
        )
        if key != expected_key:
            raise ValueError("prompt_key does not match canonical rendered prompt tokens")
        if len(row["prompt_token_ids"]) != int(row["prompt_token_count"]):
            raise ValueError("prompt_token_count does not match token IDs")
        base_count = int(row["base_rollout_count"])
        eligible_count = int(row["eligible_success_count"])
        if base_count <= 0 or eligible_count > base_count:
            raise ValueError("prompt rollout counts are invalid")
        if not math.isclose(
            float(row["q_reference"]), eligible_count / base_count, abs_tol=1e-6
        ):
            raise ValueError("q_reference does not match prompt rollout counts")
        if witness_counts[key] < threshold:
            raise ValueError(f"protected prompt {key} has fewer than support_threshold witnesses")
        if witness_counts[key] != int(row["eligible_success_count"]):
            raise ValueError("eligible_success_count does not match witness rows")
    return prompt_rows, witness_rows


def write_cache(
    root: str | Path,
    manifest: Mapping[str, Any],
    prompts: Iterable[Mapping[str, Any]],
    witnesses: Iterable[Mapping[str, Any]],
) -> str:
    """Validate and atomically write a complete immutable cache directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    normalized_manifest = dict(manifest)
    normalized_manifest.setdefault("schema_version", SCHEMA_VERSION)
    normalized_manifest.setdefault("algorithm", ALGORITHM)
    _validate_manifest(normalized_manifest)
    prompt_rows, witness_rows = _validated_rows(normalized_manifest, prompts, witnesses)
    normalized_manifest.setdefault("prompt_count", len(prompt_rows))
    if int(normalized_manifest["prompt_count"]) < len(prompt_rows):
        raise ValueError("prompt_count cannot be smaller than protected_prompt_count")
    normalized_manifest["protected_prompt_count"] = len(prompt_rows)
    normalized_manifest["witness_count"] = len(witness_rows)

    prompt_path = root / "prompts.parquet"
    witness_path = root / "witnesses.parquet"
    manifest_path = root / "manifest.json"
    _atomic_write_parquet(prompt_path, prompt_rows, PROMPT_SCHEMA)
    _atomic_write_parquet(witness_path, witness_rows, WITNESS_SCHEMA)
    _atomic_write_bytes(manifest_path, _canonical_json(normalized_manifest) + b"\n")
    file_hashes = {
        "manifest.json": _sha256_file(manifest_path),
        "prompts.parquet": _sha256_file(prompt_path),
        "witnesses.parquet": _sha256_file(witness_path),
    }
    fingerprint = _sha256_bytes(_canonical_json(file_hashes))
    _atomic_write_bytes(
        root / "hashes.json",
        _canonical_json({"cache_fingerprint": fingerprint, "files": file_hashes}) + b"\n",
    )
    return fingerprint


class SuccessSupportCache:
    """Validated cache contents with deterministic prompt-stratified sampling."""

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        prompts: list[dict[str, Any]],
        witnesses: list[dict[str, Any]],
        fingerprint: str,
    ) -> None:
        self.manifest = manifest
        self.prompts = prompts
        self.witnesses = witnesses
        self.fingerprint = fingerprint
        self._witnesses_by_prompt: dict[str, list[dict[str, Any]]] = {}
        for witness in witnesses:
            self._witnesses_by_prompt.setdefault(witness["prompt_key"], []).append(witness)

    @classmethod
    def load(cls, root: str | Path, expectations: CacheExpectations) -> "SuccessSupportCache":
        root = Path(root)
        required = ("manifest.json", "prompts.parquet", "witnesses.parquet", "hashes.json")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(f"cache is incomplete; missing {missing}")
        manifest = json.loads((root / "manifest.json").read_text())
        hashes = json.loads((root / "hashes.json").read_text())
        file_hashes = {name: _sha256_file(root / name) for name in required[:-1]}
        if hashes.get("files") != file_hashes:
            raise ValueError("cache file hash mismatch")
        fingerprint = _sha256_bytes(_canonical_json(file_hashes))
        if hashes.get("cache_fingerprint") != fingerprint:
            raise ValueError("cache fingerprint mismatch")
        _validate_manifest(manifest)
        for field in (
            "reference_budget",
            "support_threshold",
            "tokenizer_fingerprint",
            "chat_template_fingerprint",
            "logprob_temperature",
            "include_eos",
        ):
            expected = getattr(expectations, field)
            if manifest.get(field) != expected:
                raise ValueError(
                    f"cache {field} mismatch: expected {expected!r}, got {manifest.get(field)!r}"
                )
        prompts = pq.read_table(root / "prompts.parquet", schema=PROMPT_SCHEMA).to_pylist()
        witnesses = pq.read_table(root / "witnesses.parquet", schema=WITNESS_SCHEMA).to_pylist()
        prompts, witnesses = _validated_rows(manifest, prompts, witnesses)
        if manifest.get("protected_prompt_count") != len(prompts):
            raise ValueError("protected_prompt_count does not match cache contents")
        if int(manifest.get("prompt_count", 0)) < len(prompts):
            raise ValueError("prompt_count cannot be smaller than protected_prompt_count")
        if manifest.get("witness_count") != len(witnesses):
            raise ValueError("witness_count does not match cache contents")
        return cls(manifest=manifest, prompts=prompts, witnesses=witnesses, fingerprint=fingerprint)

    def sample(
        self,
        *,
        batch_size: int,
        seed: int,
        global_step: int,
        support_update_count: int,
    ) -> list[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.prompts):
            raise ValueError("support prompts must be sampled without replacement")
        seed_payload = [seed, self.fingerprint, global_step, support_update_count]
        step_seed = int(_sha256_bytes(_canonical_json(seed_payload))[:16], 16)
        generator = random.Random(step_seed)
        prompt_keys = generator.sample([row["prompt_key"] for row in self.prompts], batch_size)
        return [dict(generator.choice(self._witnesses_by_prompt[key])) for key in prompt_keys]
