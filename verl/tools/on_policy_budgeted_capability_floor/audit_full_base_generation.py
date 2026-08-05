"""Audit and enrich frozen full-Base generation artifacts under seed protocol v2.

Protocol v2 preserves the frozen 31-bit deterministic seed rule while treating
cross-prompt scalar collisions as an attested observation. Sample identity,
sample UID, within-prompt seed uniqueness, seed-rule reproduction, and all
artifact identity/hash checks remain fail-closed.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow as pa
import pyarrow.parquet as pq


SEED_PROTOCOL_VERSION = "obcf-full-base-generation-seed-protocol-v2"
SEED_RULE_VERSION = "offline-answer-timing-v1"
SEED_SPACE_SIZE = 2**31
RESPONSE_HASH_VERSION = "obcf-response-token-ids-v1"
SAMPLE_UID_VERSION = "obcf-full-base-sample-uid-v2"
GENERATION_PROTOCOL_VERSION = "obcf-full-base-generation-protocol-v2"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_payload(label: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(label.encode())
    digest.update(b"\0")
    digest.update(_canonical_json(value).encode())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _file_set_hash(paths: Sequence[Path]) -> str:
    files = sorted((Path(path).resolve() for path in paths), key=lambda path: path.name)
    if not files or len({path.name for path in files}) != len(files):
        raise ValueError("artifact file set must be nonempty with unique names")
    digest = hashlib.sha256()
    digest.update(b"obcf-artifact-file-set-v1\0")
    for path in files:
        identity = path.name.encode()
        payload = path.read_bytes()
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _token_ids(value: Any, name: str, *, allow_empty: bool = True) -> list[int]:
    if not isinstance(value, (list, tuple)) or (not allow_empty and not value):
        raise ValueError(f"{name} must be a token-ID list")
    result = []
    for token in value:
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError(f"{name} must contain nonnegative integer token IDs")
        result.append(token)
    return result


def frozen_sampling_seed(master_seed: int, prompt_id: int, rollout_index: int) -> int:
    """Reproduce the immutable offline-answer-timing-v1 31-bit seed rule."""
    master_seed = _nonnegative_int(master_seed, "master_seed")
    prompt_id = _nonnegative_int(prompt_id, "prompt_id")
    rollout_index = _nonnegative_int(rollout_index, "rollout_index")
    payload = f"{SEED_RULE_VERSION}\0{master_seed}\0{prompt_id}\0{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def response_token_hash(response_token_ids: Any) -> str:
    tokens = _token_ids(response_token_ids, "response_token_ids")
    digest = hashlib.sha256()
    digest.update(f"{RESPONSE_HASH_VERSION}\0".encode())
    digest.update(json.dumps(tokens, separators=(",", ":")).encode())
    return digest.hexdigest()


def derive_sample_uid(
    *,
    dataset_fingerprint: str,
    prompt_id: int,
    rollout_index: int,
    prompt_hash: str,
    generation_config_fingerprint: str,
) -> str:
    """Bind a globally unique sample identity to frozen generation provenance."""
    if not all(
        isinstance(value, str) and value
        for value in (dataset_fingerprint, prompt_hash, generation_config_fingerprint)
    ):
        raise ValueError("sample UID fingerprints and prompt_hash must be nonempty strings")
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "generation_config_fingerprint": generation_config_fingerprint,
        "prompt_hash": prompt_hash,
        "prompt_id": _nonnegative_int(prompt_id, "prompt_id"),
        "rollout_index": _nonnegative_int(rollout_index, "rollout_index"),
        "version": SAMPLE_UID_VERSION,
    }
    return _sha256_payload(SAMPLE_UID_VERSION, payload)


def _collision_attestation_payload(collisions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol_version": SEED_PROTOCOL_VERSION,
        "collision_semantics": "cross-prompt scalar seed collisions are reported but not rewritten",
        "cross_prompt_seed_collisions": collisions,
    }


def _generation_protocol_fingerprint(
    *,
    master_seed: int,
    dataset_fingerprint: str,
    generation_config_fingerprint: str,
    seed_collision_attestation_sha256: str,
    tokenizer_fingerprint: str = "",
    chat_template_fingerprint: str = "",
    model_fingerprint: str = "",
) -> str:
    return _sha256_payload(
        GENERATION_PROTOCOL_VERSION,
        {
            "dataset_fingerprint": dataset_fingerprint,
            "generation_config_fingerprint": generation_config_fingerprint,
            "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
            "master_seed": master_seed,
            "model_fingerprint": model_fingerprint,
            "seed_collision_attestation_sha256": seed_collision_attestation_sha256,
            "seed_protocol_version": SEED_PROTOCOL_VERSION,
            "seed_rule_version": SEED_RULE_VERSION,
            "chat_template_fingerprint": chat_template_fingerprint,
            "tokenizer_fingerprint": tokenizer_fingerprint,
        },
    )


def audit_generation_rows_v2(
    *,
    rows: Sequence[Mapping[str, Any]],
    master_seed: int,
    dataset_fingerprint: str,
    generation_config_fingerprint: str,
    expected_prompt_count: int | None = None,
    expected_rollout_count: int | None = None,
    expected_rollouts_per_prompt: int | None = None,
    expected_prompts_by_id: Mapping[int, Mapping[str, Any]] | None = None,
    tokenizer_fingerprint: str = "",
    chat_template_fingerprint: str = "",
    model_fingerprint: str = "",
) -> dict[str, Any]:
    """Return a fail-closed seed/identity audit without mutating source rows."""
    if not rows:
        raise ValueError("generation rows must be nonempty")
    identity_counts: Counter[tuple[int, int]] = Counter()
    identity_shards: dict[tuple[int, int], set[str]] = defaultdict(set)
    prompt_rollout_indices: dict[int, set[int]] = defaultdict(set)
    prompt_seed_identities: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seed_identities: dict[int, list[tuple[int, int]]] = defaultdict(list)
    uid_counts: Counter[str] = Counter()
    invalid_identity_count = 0
    sample_uid_mismatch_count = 0
    wrong_seed_rule_count = 0
    generation_error_count = 0
    prompt_hash_missing_count = 0
    prompt_hash_mismatch_count = 0
    prompt_token_id_mismatch_count = 0
    response_hash_mismatch_count = 0
    response_token_count_mismatch_count = 0
    config_fingerprint_mismatch_count = 0
    shard_identity_mismatch_count = 0
    tokenizer_fingerprint_mismatch_count = 0
    chat_template_fingerprint_mismatch_count = 0
    model_fingerprint_mismatch_count = 0

    for source in rows:
        row = dict(source)
        try:
            prompt_id = _nonnegative_int(row.get("prompt_id"), "prompt_id")
            rollout_index = _nonnegative_int(row.get("rollout_index"), "rollout_index")
            seed = _nonnegative_int(row.get("sampling_seed"), "sampling_seed")
        except ValueError:
            invalid_identity_count += 1
            continue
        identity = (prompt_id, rollout_index)
        identity_counts[identity] += 1
        prompt_rollout_indices[prompt_id].add(rollout_index)
        prompt_seed_identities[prompt_id][seed].append(identity)
        seed_identities[seed].append(identity)
        expected_seed = frozen_sampling_seed(master_seed, prompt_id, rollout_index)
        wrong_seed_rule_count += seed != expected_seed

        prompt_hash = row.get("prompt_hash")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            prompt_hash_missing_count += 1
            prompt_hash = "<missing>"
        if expected_prompts_by_id is not None:
            prompt = expected_prompts_by_id.get(prompt_id)
            if prompt is None:
                prompt_hash_mismatch_count += 1
                prompt_token_id_mismatch_count += 1
            else:
                prompt_hash_mismatch_count += prompt_hash != prompt.get("prompt_hash")
                prompt_token_id_mismatch_count += row.get("prompt_token_ids") != prompt.get(
                    "prompt_token_ids"
                )
        expected_uid = derive_sample_uid(
            dataset_fingerprint=dataset_fingerprint,
            prompt_id=prompt_id,
            rollout_index=rollout_index,
            prompt_hash=prompt_hash,
            generation_config_fingerprint=generation_config_fingerprint,
        )
        declared_uid = row.get("sample_uid", expected_uid)
        if not isinstance(declared_uid, str) or not declared_uid:
            declared_uid = "<invalid>"
        sample_uid_mismatch_count += declared_uid != expected_uid
        uid_counts[declared_uid] += 1

        error = row.get("generation_error")
        generation_error_count += error not in (None, "", False, 0)
        if row.get("config_fingerprint") != generation_config_fingerprint:
            config_fingerprint_mismatch_count += 1
        if tokenizer_fingerprint:
            tokenizer_fingerprint_mismatch_count += (
                row.get("tokenizer_fingerprint") != tokenizer_fingerprint
            )
        if chat_template_fingerprint:
            chat_template_fingerprint_mismatch_count += (
                row.get("chat_template_fingerprint") != chat_template_fingerprint
            )
        if model_fingerprint:
            model_fingerprint_mismatch_count += row.get("model_fingerprint") != model_fingerprint
        try:
            response_ids = _token_ids(row.get("response_token_ids"), "response_token_ids")
            response_token_count_mismatch_count += row.get("response_token_count") != len(response_ids)
            response_hash_mismatch_count += row.get("response_hash") != response_token_hash(response_ids)
        except ValueError:
            response_token_count_mismatch_count += 1
            response_hash_mismatch_count += 1

        shard = row.get("prompt_shard", row.get("shard_id"))
        identity_shards[identity].add(str(shard))
        if isinstance(shard, str) and "/" in shard:
            try:
                shard_index_text, shard_count_text = shard.split("/", 1)
                shard_index, shard_count = int(shard_index_text), int(shard_count_text)
                shard_identity_mismatch_count += (
                    shard_count <= 0
                    or not 0 <= shard_index < shard_count
                    or prompt_id % shard_count != shard_index
                )
            except ValueError:
                shard_identity_mismatch_count += 1

    shard_overlap_count = sum(len(shards) > 1 for shards in identity_shards.values())

    duplicate_sample_identity_count = sum(count - 1 for count in identity_counts.values())
    duplicate_sample_uid_count = sum(count - 1 for count in uid_counts.values())
    within_prompt_collisions = []
    within_prompt_seed_collision_count = 0
    for prompt_id, by_seed in sorted(prompt_seed_identities.items()):
        for seed, identities in sorted(by_seed.items()):
            unique = sorted(set(identities))
            if len(unique) > 1:
                pair_count = math.comb(len(unique), 2)
                within_prompt_seed_collision_count += pair_count
                within_prompt_collisions.append(
                    {
                        "prompt_id": prompt_id,
                        "sampling_seed": seed,
                        "pair_count": pair_count,
                        "identities": [
                            {"prompt_id": value[0], "rollout_index": value[1]} for value in unique
                        ],
                    }
                )

    cross_prompt_collisions = []
    collision_rows: set[tuple[int, int]] = set()
    cross_prompt_seed_collision_pair_count = 0
    for seed, identities in sorted(seed_identities.items()):
        unique = sorted(set(identities))
        prompts = {identity[0] for identity in unique}
        if len(prompts) < 2:
            continue
        by_prompt = Counter(identity[0] for identity in unique)
        pair_count = math.comb(len(unique), 2) - sum(
            math.comb(count, 2) for count in by_prompt.values() if count > 1
        )
        cross_prompt_seed_collision_pair_count += pair_count
        collision_rows.update(unique)
        cross_prompt_collisions.append(
            {
                "sampling_seed": seed,
                "pair_count": pair_count,
                "identities": [
                    {"prompt_id": identity[0], "rollout_index": identity[1]} for identity in unique
                ],
            }
        )

    observed_prompt_count = len(prompt_rollout_indices)
    missing_prompt_count = 0
    extra_prompt_count = 0
    if expected_prompt_count is not None:
        expected_ids = set(range(expected_prompt_count))
        observed_ids = set(prompt_rollout_indices)
        missing_prompt_count = len(expected_ids - observed_ids)
        extra_prompt_count = len(observed_ids - expected_ids)
    rollout_index_set_mismatch_count = 0
    if expected_rollouts_per_prompt is not None:
        expected_indices = set(range(expected_rollouts_per_prompt))
        rollout_index_set_mismatch_count = sum(
            indices != expected_indices for indices in prompt_rollout_indices.values()
        ) + missing_prompt_count
    rollout_count_mismatch = int(
        expected_rollout_count is not None and len(rows) != expected_rollout_count
    )
    prompt_count_mismatch = int(
        expected_prompt_count is not None and observed_prompt_count != expected_prompt_count
    )
    collision_payload = _collision_attestation_payload(cross_prompt_collisions)
    collision_hash = _sha256_payload(SEED_PROTOCOL_VERSION, collision_payload)
    protocol_fp = _generation_protocol_fingerprint(
        master_seed=master_seed,
        dataset_fingerprint=dataset_fingerprint,
        generation_config_fingerprint=generation_config_fingerprint,
        seed_collision_attestation_sha256=collision_hash,
        tokenizer_fingerprint=tokenizer_fingerprint,
        chat_template_fingerprint=chat_template_fingerprint,
        model_fingerprint=model_fingerprint,
    )
    hard_failure_counts = {
        "config_fingerprint_mismatch_count": config_fingerprint_mismatch_count,
        "duplicate_sample_identity_count": duplicate_sample_identity_count,
        "duplicate_sample_uid_count": duplicate_sample_uid_count,
        "extra_prompt_count": extra_prompt_count,
        "generation_error_count": generation_error_count,
        "invalid_identity_count": invalid_identity_count,
        "missing_prompt_count": missing_prompt_count,
        "prompt_count_mismatch": prompt_count_mismatch,
        "prompt_hash_missing_count": prompt_hash_missing_count,
        "prompt_hash_mismatch_count": prompt_hash_mismatch_count,
        "prompt_token_id_mismatch_count": prompt_token_id_mismatch_count,
        "response_hash_mismatch_count": response_hash_mismatch_count,
        "response_token_count_mismatch_count": response_token_count_mismatch_count,
        "rollout_count_mismatch": rollout_count_mismatch,
        "rollout_index_set_mismatch_count": rollout_index_set_mismatch_count,
        "sample_uid_mismatch_count": sample_uid_mismatch_count,
        "shard_identity_mismatch_count": shard_identity_mismatch_count,
        "shard_overlap_count": shard_overlap_count,
        "tokenizer_fingerprint_mismatch_count": tokenizer_fingerprint_mismatch_count,
        "chat_template_fingerprint_mismatch_count": chat_template_fingerprint_mismatch_count,
        "model_fingerprint_mismatch_count": model_fingerprint_mismatch_count,
        "within_prompt_seed_collision_count": within_prompt_seed_collision_count,
        "wrong_seed_rule_count": wrong_seed_rule_count,
    }
    return {
        "schema_version": 2,
        "protocol_version": SEED_PROTOCOL_VERSION,
        "seed_rule_version": SEED_RULE_VERSION,
        "passed": all(count == 0 for count in hard_failure_counts.values()),
        "row_count": len(rows),
        "prompt_count": observed_prompt_count,
        "master_seed": master_seed,
        "dataset_fingerprint": dataset_fingerprint,
        "generation_config_fingerprint": generation_config_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "chat_template_fingerprint": chat_template_fingerprint,
        "model_fingerprint": model_fingerprint,
        **hard_failure_counts,
        "cross_prompt_seed_collision_pair_count": cross_prompt_seed_collision_pair_count,
        "cross_prompt_seed_collision_row_count": len(collision_rows),
        "cross_prompt_seed_collisions": cross_prompt_collisions,
        "within_prompt_seed_collisions": within_prompt_collisions,
        "expected_birthday_collision_pairs": len(rows) * (len(rows) - 1) / (2 * SEED_SPACE_SIZE),
        "observed_vs_expected_ratio": (
            cross_prompt_seed_collision_pair_count
            / (len(rows) * (len(rows) - 1) / (2 * SEED_SPACE_SIZE))
            if len(rows) > 1
            else 0.0
        ),
        "seed_collision_attestation_sha256": collision_hash,
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "generation_protocol_fingerprint": protocol_fp,
    }


def enrich_generation_rows_v2(
    *,
    rows: Sequence[Mapping[str, Any]],
    prompts_by_id: Mapping[int, Mapping[str, Any]],
    master_seed: int,
    dataset_fingerprint: str,
    generation_config_fingerprint: str,
    tokenizer_fingerprint: str,
    chat_template_fingerprint: str,
    model_fingerprint: str,
) -> list[dict[str, Any]]:
    """Add protocol-v2 provenance while preserving every frozen source field."""
    enriched = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        prompt_id = _nonnegative_int(row.get("prompt_id"), "prompt_id")
        rollout_index = _nonnegative_int(row.get("rollout_index"), "rollout_index")
        prompt = prompts_by_id.get(prompt_id)
        if prompt is None:
            raise ValueError(f"rollout references missing prompt {prompt_id}")
        if row.get("prompt_hash") != prompt.get("prompt_hash"):
            raise ValueError(f"prompt_hash mismatch for prompt {prompt_id}")
        if "prompt_token_ids" in row and row["prompt_token_ids"] != prompt.get("prompt_token_ids"):
            raise ValueError(f"prompt_token_ids mismatch for prompt {prompt_id}")
        response_ids = _token_ids(row.get("response_token_ids"), "response_token_ids")
        if row.get("response_token_count") != len(response_ids):
            raise ValueError(f"response_token_count mismatch for prompt {prompt_id}")
        computed_response_hash = response_token_hash(response_ids)
        if "response_hash" in row and row["response_hash"] != computed_response_hash:
            raise ValueError(f"response_hash mismatch for prompt {prompt_id}")
        if row.get("sampling_seed") != frozen_sampling_seed(master_seed, prompt_id, rollout_index):
            raise ValueError(f"sampling_seed rule mismatch for prompt {prompt_id}")
        if row.get("config_fingerprint") != generation_config_fingerprint:
            raise ValueError(f"generation config fingerprint mismatch for prompt {prompt_id}")
        row.update(
            {
                "sample_uid": derive_sample_uid(
                    dataset_fingerprint=dataset_fingerprint,
                    prompt_id=prompt_id,
                    rollout_index=rollout_index,
                    prompt_hash=row["prompt_hash"],
                    generation_config_fingerprint=generation_config_fingerprint,
                ),
                "response_hash": computed_response_hash,
                "generation_error": row.get("generation_error"),
                "dataset_fingerprint": dataset_fingerprint,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "chat_template_fingerprint": chat_template_fingerprint,
                "model_fingerprint": model_fingerprint,
                "seed_protocol_version": SEED_PROTOCOL_VERSION,
                "seed_rule_version": SEED_RULE_VERSION,
                "config_provenance_fingerprint": generation_config_fingerprint,
            }
        )
        if "raw_prompt" in prompt:
            row["raw_prompt"] = copy.deepcopy(prompt["raw_prompt"])
        if "extra_info" in prompt:
            row["extra_info"] = copy.deepcopy(prompt["extra_info"])
        enriched.append(row)
    return enriched


def validate_generation_protocol_attestation_v2(attestation: Mapping[str, Any]) -> None:
    """Reject legacy, malformed, failed, or collision-unbound protocol artifacts."""
    if attestation.get("protocol_version") != SEED_PROTOCOL_VERSION:
        raise ValueError("generation protocol attestation protocol_version mismatch")
    required = {
        "schema_version": 2,
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "passed": True,
    }
    for field, expected in required.items():
        if attestation.get(field) != expected:
            raise ValueError(f"generation protocol attestation {field} mismatch")
    collision_hash = attestation.get("seed_collision_attestation_sha256")
    if not isinstance(collision_hash, str) or len(collision_hash) != 64:
        raise ValueError("generation protocol attestation collision hash is invalid")
    expected_fp = _generation_protocol_fingerprint(
        master_seed=_nonnegative_int(attestation.get("master_seed"), "master_seed"),
        dataset_fingerprint=str(attestation.get("dataset_fingerprint", "")),
        generation_config_fingerprint=str(attestation.get("generation_config_fingerprint", "")),
        seed_collision_attestation_sha256=collision_hash,
        tokenizer_fingerprint=str(attestation.get("tokenizer_fingerprint", "")),
        chat_template_fingerprint=str(attestation.get("chat_template_fingerprint", "")),
        model_fingerprint=str(attestation.get("model_fingerprint", "")),
    )
    if attestation.get("generation_protocol_fingerprint") != expected_fp:
        raise ValueError("generation protocol attestation fingerprint mismatch")


def _source_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _load_prompt_rows(path: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = pq.read_table(path).to_pylist()
    by_id = {}
    for source in rows:
        row = dict(source)
        prompt_id = _nonnegative_int(row.get("prompt_id"), "prompt_id")
        if prompt_id in by_id:
            raise ValueError(f"duplicate prompt identity {prompt_id}")
        raw_prompt = row.get("raw_prompt", row.get("canonical_prompt"))
        if isinstance(raw_prompt, str):
            raw_prompt = json.loads(raw_prompt)
        extra_info = row.get("extra_info", row.get("extra_info_json", {}))
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        row["raw_prompt"] = raw_prompt
        row["extra_info"] = extra_info
        by_id[prompt_id] = row
    return rows, by_id


def _write_collision_artifacts(output: Path, report: Mapping[str, Any], rows: list[dict]) -> None:
    identities = {
        (identity["prompt_id"], identity["rollout_index"])
        for collision in report["cross_prompt_seed_collisions"]
        for identity in collision["identities"]
    }
    collision_rows = sorted(
        (row for row in rows if (row["prompt_id"], row["rollout_index"]) in identities),
        key=lambda row: (row["sampling_seed"], row["prompt_id"], row["rollout_index"]),
    )
    selected_fields = (
        "sampling_seed",
        "prompt_id",
        "rollout_index",
        "sample_uid",
        "prompt_hash",
        "response_hash",
        "prompt_token_count",
        "response_token_count",
        "finish_reason",
        "stop_reason",
        "prompt_shard",
        "config_fingerprint",
    )
    serialized = []
    for row in collision_rows:
        value = {name: row.get(name) for name in selected_fields}
        value["expected_seed_from_frozen_rule"] = frozen_sampling_seed(
            report["master_seed"], row["prompt_id"], row["rollout_index"]
        )
        value["seed_rule_match"] = value["sampling_seed"] == value["expected_seed_from_frozen_rule"]
        serialized.append(value)
    pq.write_table(pa.Table.from_pylist(serialized), output / "seed_collision_rows.parquet", compression="zstd")
    _atomic_json(output / "seed_collision_rows.json", serialized)
    lines = [
        "# Seed Collision Analysis",
        "",
        f"Protocol: `{SEED_PROTOCOL_VERSION}`",
        "",
        f"- Cross-prompt collision pairs: {report['cross_prompt_seed_collision_pair_count']}",
        f"- Affected rows: {report['cross_prompt_seed_collision_row_count']}",
        f"- Within-prompt collision pairs: {report['within_prompt_seed_collision_count']}",
        f"- Wrong frozen seeds: {report['wrong_seed_rule_count']}",
        "- Collision rows were retained without deletion, resampling, seed modification, or reindexing.",
        "",
    ]
    (output / "seed_collision_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run_artifact_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output}")
    output.mkdir(parents=True)
    enriched_dir = output / "full_base_rollouts_enriched"
    enriched_dir.mkdir()
    prompt_rows, prompts_by_id = _load_prompt_rows(args.prompts)
    if len(prompt_rows) != args.expected_prompt_count:
        raise ValueError("prompt manifest row count mismatch")
    shutil.copyfile(args.prompts, output / "full_base_prompt_manifest.parquet")

    raw_files = sorted(args.rollouts.glob("part-*.parquet"))
    if not raw_files:
        raise ValueError("rollout directory contains no parquet parts")
    all_enriched: list[dict[str, Any]] = []
    enriched_files = []
    source_rows = 0
    for raw_file in raw_files:
        source = pq.read_table(raw_file).to_pylist()
        source_rows += len(source)
        enriched = enrich_generation_rows_v2(
            rows=source,
            prompts_by_id=prompts_by_id,
            master_seed=args.master_seed,
            dataset_fingerprint=args.dataset_fingerprint,
            generation_config_fingerprint=args.generation_config_fingerprint,
            tokenizer_fingerprint=args.tokenizer_fingerprint,
            chat_template_fingerprint=args.chat_template_fingerprint,
            model_fingerprint=args.model_fingerprint,
        )
        if any(
            before["response_token_ids"] != after["response_token_ids"]
            or before["prompt_id"] != after["prompt_id"]
            or before["rollout_index"] != after["rollout_index"]
            for before, after in zip(source, enriched, strict=True)
        ):
            raise ValueError("enrichment changed frozen identity or response token IDs")
        target = enriched_dir / raw_file.name
        pq.write_table(pa.Table.from_pylist(enriched), target, compression="zstd", use_dictionary=True)
        enriched_files.append(target)
        all_enriched.extend(enriched)
    if source_rows != args.expected_rollout_count:
        raise ValueError("rollout artifact row count mismatch")

    report = audit_generation_rows_v2(
        rows=all_enriched,
        master_seed=args.master_seed,
        dataset_fingerprint=args.dataset_fingerprint,
        generation_config_fingerprint=args.generation_config_fingerprint,
        expected_prompt_count=args.expected_prompt_count,
        expected_rollout_count=args.expected_rollout_count,
        expected_rollouts_per_prompt=args.rollouts_per_prompt,
        expected_prompts_by_id=prompts_by_id,
        tokenizer_fingerprint=args.tokenizer_fingerprint,
        chat_template_fingerprint=args.chat_template_fingerprint,
        model_fingerprint=args.model_fingerprint,
    )
    report.update(
        {
            "created_at_utc": _now(),
            "source_git_commit": _source_commit(),
            "raw_rollout_fingerprint": _file_set_hash(raw_files),
            "enriched_rollout_fingerprint": _file_set_hash(enriched_files),
            "prompt_manifest_sha256": _sha256_file(output / "full_base_prompt_manifest.parquet"),
            "raw_rollout_file_count": len(raw_files),
            "enriched_rollout_file_count": len(enriched_files),
        }
    )
    _write_collision_artifacts(output, report, all_enriched)
    collision_attestation = {
        "schema_version": 2,
        "protocol_version": SEED_PROTOCOL_VERSION,
        "passed": report["within_prompt_seed_collision_count"] == 0
        and report["wrong_seed_rule_count"] == 0,
        "cross_prompt_seed_collision_pair_count": report["cross_prompt_seed_collision_pair_count"],
        "cross_prompt_seed_collision_row_count": report["cross_prompt_seed_collision_row_count"],
        "cross_prompt_seed_collisions": report["cross_prompt_seed_collisions"],
        "within_prompt_seed_collision_count": report["within_prompt_seed_collision_count"],
        "wrong_seed_rule_count": report["wrong_seed_rule_count"],
        "expected_birthday_collision_pairs": report["expected_birthday_collision_pairs"],
        "observed_vs_expected_ratio": report["observed_vs_expected_ratio"],
        "seed_collision_attestation_sha256": report["seed_collision_attestation_sha256"],
        "collision_rows_sha256": _sha256_file(output / "seed_collision_rows.parquet"),
    }
    protocol_attestation = {
        **report,
        "artifact_hashes": {
            "prompts": report["prompt_manifest_sha256"],
            "raw_rollouts": report["raw_rollout_fingerprint"],
            "enriched_rollouts": report["enriched_rollout_fingerprint"],
            "seed_collision_rows": collision_attestation["collision_rows_sha256"],
        },
        "artifact_row_counts": {
            "prompts": len(prompt_rows),
            "raw_rollouts": source_rows,
            "enriched_rollouts": len(all_enriched),
            "seed_collision_rows": report["cross_prompt_seed_collision_row_count"],
        },
    }
    if protocol_attestation["passed"]:
        validate_generation_protocol_attestation_v2(protocol_attestation)
    manifest = {
        "schema_version": 2,
        "created_at_utc": _now(),
        "passed": report["passed"],
        "protocol_version": SEED_PROTOCOL_VERSION,
        "generation_protocol_version": GENERATION_PROTOCOL_VERSION,
        "generation_protocol_fingerprint": report["generation_protocol_fingerprint"],
        "model_id": args.model_id,
        "prompt_count": len(prompt_rows),
        "rollout_count": len(all_enriched),
        "rollouts_per_prompt": args.rollouts_per_prompt,
        "raw_rollout_directory": str(args.rollouts.resolve()),
        "enriched_rollout_directory": str(enriched_dir),
        "raw_rollout_fingerprint": report["raw_rollout_fingerprint"],
        "enriched_rollout_fingerprint": report["enriched_rollout_fingerprint"],
        "seed_collision_attestation_sha256": report["seed_collision_attestation_sha256"],
    }
    _atomic_json(output / "full_base_generation_manifest_v2.json", manifest)
    _atomic_json(output / "full_base_generation_audit_v2.json", report)
    _atomic_json(output / "full_base_seed_collision_attestation_v2.json", collision_attestation)
    _atomic_json(output / "full_base_generation_protocol_attestation_v2.json", protocol_attestation)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--generation-config-fingerprint", required=True)
    parser.add_argument("--tokenizer-fingerprint", required=True)
    parser.add_argument("--chat-template-fingerprint", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--model-id", default="base")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--expected-prompt-count", type=int, required=True)
    parser.add_argument("--expected-rollout-count", type=int, required=True)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    args = parser.parse_args()
    report = run_artifact_audit(args)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
