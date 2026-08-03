"""Build a strict prompt-level OBCF floor cache from Base audit artifacts."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq

from verl.experimental.capability_constraints.identity import (
    reference_model_fingerprint,
    tokenizer_fingerprints,
)
from verl.experimental.on_policy_budgeted_capability_floor.cache import build_floor_rows, write_cache
from verl.experimental.on_policy_budgeted_capability_floor.reward_adapter import (
    verifier_pipeline_fingerprint,
)


def _artifact_files(paths: Path | Sequence[Path]) -> list[Path]:
    inputs = [paths] if isinstance(paths, Path) else list(paths)
    files: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if glob.has_magic(str(path)):
            matches = [Path(match) for match in glob.glob(str(path))]
            if not matches:
                raise ValueError(f"artifact glob {path} matched no files")
            files.extend(matches)
        elif path.is_dir():
            parts = sorted(path.glob("part-*.parquet")) or sorted(path.glob("*.parquet"))
            if not parts:
                raise ValueError(f"artifact directory {path} contains no parquet parts")
            files.extend(parts)
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"artifact path {path} does not exist")
    resolved = sorted({path.resolve() for path in files}, key=lambda path: str(path))
    if not resolved:
        raise ValueError("artifact input contains no files")
    return resolved


def _rows(paths: Path | Sequence[Path]) -> list[dict]:
    files = _artifact_files(paths)
    if len(files) > 1 or files[0].suffix == ".parquet":
        if any(path.suffix != ".parquet" for path in files):
            raise ValueError("multi-file artifacts must contain only parquet files")
        rows: list[dict] = []
        expected_schema = None
        for path in files:
            table = pq.read_table(path)
            if expected_schema is None:
                expected_schema = table.schema
            elif table.schema != expected_schema:
                raise ValueError("artifact parquet parts have incompatible schemas")
            rows.extend(table.to_pylist())
        return rows
    path = files[0]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a row list")
    return payload


def _file_hash(paths: Path | Sequence[Path]) -> str:
    files = _artifact_files(paths)
    if len({file.name for file in files}) != len(files):
        raise ValueError("artifact parts must have unique file names")
    digest = hashlib.sha256()
    digest.update(b"obcf-artifact-file-set-v1\0")
    for file in files:
        identity = file.name.encode()
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        payload = file.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_tokenizer_fingerprints(
    tokenizer, expected_tokenizer: str | None, expected_template: str | None
) -> tuple[str, str]:
    actual_tokenizer, actual_template = tokenizer_fingerprints(tokenizer)
    if expected_tokenizer is not None and expected_tokenizer != actual_tokenizer:
        raise ValueError("provided tokenizer fingerprint does not match the local model tokenizer")
    if expected_template is not None and expected_template != actual_template:
        raise ValueError("provided chat-template fingerprint does not match the local model tokenizer")
    return actual_tokenizer, actual_template


def _json_object(value: str, name: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _apply_legacy_attestation(
    *,
    prompts: list[dict],
    rollouts: list[dict],
    scores: list[dict],
    attestation: dict,
    tokenizer_fingerprint: str,
    chat_template_fingerprint: str,
    verifier_fingerprint: str,
    source_git_commit: str,
    artifact_fingerprints: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Bind immutable legacy rows through a hash- and config-bound audit attestation."""
    required = {
        "schema_version": 1,
        "passed": True,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "chat_template_fingerprint": chat_template_fingerprint,
        "verifier_fingerprint": verifier_fingerprint,
        "source_git_commit": source_git_commit,
        **artifact_fingerprints,
    }
    for field, expected in required.items():
        if attestation.get(field) != expected:
            raise ValueError(f"legacy artifact attestation {field} mismatch")
    config_fingerprint = attestation.get("config_fingerprint")
    if (
        not isinstance(config_fingerprint, str)
        or len(config_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in config_fingerprint)
    ):
        raise ValueError("legacy artifact attestation config_fingerprint is invalid")
    for name, rows in (("prompt", prompts), ("rollout", rollouts), ("score", scores)):
        if not rows or any(row.get("config_fingerprint") != config_fingerprint for row in rows):
            raise ValueError(f"legacy {name} rows do not match attested config_fingerprint")

    converted_prompts = [
        dict(row)
        | {
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "chat_template_fingerprint": chat_template_fingerprint,
        }
        for row in prompts
    ]
    converted_scores = [
        dict(row)
        | {
            "verifier_fingerprint": verifier_fingerprint,
            "source_git_commit": source_git_commit,
        }
        for row in scores
    ]
    return converted_prompts, converted_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-fingerprint")
    parser.add_argument("--chat-template-fingerprint")
    parser.add_argument("--verifier-fingerprint", required=True)
    parser.add_argument("--reward-manager-source", choices=("register", "importlib"), default="register")
    parser.add_argument("--reward-manager-name", default="naive")
    parser.add_argument("--reward-manager-module-path", type=Path)
    parser.add_argument("--reward-manager-module-name", default="custom_reward_manager")
    parser.add_argument("--custom-reward-function", type=Path)
    parser.add_argument("--custom-reward-function-name", default="compute_score")
    parser.add_argument("--custom-reward-kwargs", default="{}")
    parser.add_argument("--reward-kwargs", default="{}")
    parser.add_argument("--sandbox-fusion-url")
    parser.add_argument("--sandbox-max-concurrent", type=int, default=64)
    parser.add_argument("--sandbox-memory-limit-mb", type=int, default=1024)
    parser.add_argument("--reference-budget", type=int, default=2048)
    parser.add_argument("--base-rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--support-threshold", type=int, default=2)
    parser.add_argument("--reference-tolerance-count", type=int, default=1)
    parser.add_argument("--source-git-commit", required=True)
    parser.add_argument(
        "--legacy-artifact-attestation",
        type=Path,
        help="hash-bound attestation for immutable artifacts with only config_fingerprint",
    )
    args = parser.parse_args()

    from verl.utils.tokenizer import hf_tokenizer

    tokenizer = hf_tokenizer(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    tokenizer_fp, template_fp = _validate_tokenizer_fingerprints(
        tokenizer, args.tokenizer_fingerprint, args.chat_template_fingerprint
    )
    custom_reward_kwargs = _json_object(args.custom_reward_kwargs, "custom reward kwargs")
    reward_kwargs = _json_object(args.reward_kwargs, "reward kwargs")
    sandbox_fusion = {
        "url": args.sandbox_fusion_url,
        "max_concurrent": args.sandbox_max_concurrent,
        "memory_limit_mb": args.sandbox_memory_limit_mb,
    }
    verifier_fp = verifier_pipeline_fingerprint(
        reward_manager_name=args.reward_manager_name,
        reward_manager_source=args.reward_manager_source,
        reward_manager_module_path=(
            str(args.reward_manager_module_path) if args.reward_manager_module_path else None
        ),
        reward_manager_module_name=args.reward_manager_module_name,
        custom_reward_function_path=(
            str(args.custom_reward_function) if args.custom_reward_function else None
        ),
        custom_reward_function_name=args.custom_reward_function_name,
        custom_reward_kwargs=custom_reward_kwargs,
        reward_kwargs=reward_kwargs,
        sandbox_fusion=sandbox_fusion,
    )
    if args.verifier_fingerprint != verifier_fp:
        raise ValueError("provided verifier fingerprint does not match local reward pipeline")

    artifact_fingerprints = {
        "prompt_manifest_fingerprint": _file_hash(args.prompts),
        "rollout_fingerprint": _file_hash(args.rollouts),
        "score_fingerprint": _file_hash(args.scores),
    }
    prompts = _rows(args.prompts)
    rollouts = _rows(args.rollouts)
    scores = _rows(args.scores)
    if args.legacy_artifact_attestation is not None:
        attestation = json.loads(args.legacy_artifact_attestation.read_text())
        if not isinstance(attestation, dict):
            raise ValueError("legacy artifact attestation must be a JSON object")
        prompts, scores = _apply_legacy_attestation(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            attestation=attestation,
            tokenizer_fingerprint=tokenizer_fp,
            chat_template_fingerprint=template_fp,
            verifier_fingerprint=verifier_fp,
            source_git_commit=args.source_git_commit,
            artifact_fingerprints=artifact_fingerprints,
        )
    rows = build_floor_rows(
        prompts=prompts,
        rollouts=rollouts,
        scores=scores,
        model_id=args.model_id,
        tokenizer_fingerprint=tokenizer_fp,
        chat_template_fingerprint=template_fp,
        reference_budget=args.reference_budget,
        base_rollouts_per_prompt=args.base_rollouts_per_prompt,
        support_threshold=args.support_threshold,
        reference_tolerance_count=args.reference_tolerance_count,
        verifier_fingerprint=verifier_fp,
        source_git_commit=args.source_git_commit,
        require_prompt_provenance=True,
    )
    manifest = {
        "schema_version": 1,
        "algorithm": "on_policy_budgeted_capability_floor",
        "reference_model_id": args.model_id,
        "reference_model_hash": reference_model_fingerprint(args.model_path),
        "reference_budget": args.reference_budget,
        "base_rollouts_per_prompt": args.base_rollouts_per_prompt,
        "support_threshold": args.support_threshold,
        "reference_tolerance_count": args.reference_tolerance_count,
        "prefix_reward_field": f"prefix_reward_{args.reference_budget}",
        "tokenizer_fingerprint": tokenizer_fp,
        "chat_template_fingerprint": template_fp,
        **artifact_fingerprints,
        "verifier_fingerprint": verifier_fp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": args.source_git_commit,
        "prompt_count": len(prompts),
    }
    fingerprint = write_cache(args.output, manifest, rows)
    print(json.dumps({"cache_fingerprint": fingerprint, "protected_prompt_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
