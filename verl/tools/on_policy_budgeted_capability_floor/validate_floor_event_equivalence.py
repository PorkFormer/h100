"""Recompute frozen Base prefixes through the online reward pipeline and attest equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from omegaconf import OmegaConf, open_dict

from verl.experimental.capability_constraints.identity import tokenizer_fingerprints
from verl.experimental.on_policy_budgeted_capability_floor.artifact_batch import (
    build_full_rollout_batch_from_artifacts,
)
from verl.experimental.on_policy_budgeted_capability_floor.event_equivalence import (
    PREFIX_PROTOCOL_VERSION,
    compare_prefix_events,
    prefix_protocol_fingerprint,
)
from verl.experimental.on_policy_budgeted_capability_floor.prefix_batch import (
    build_exact_prefix_batch,
)
from verl.experimental.on_policy_budgeted_capability_floor.reward_adapter import (
    verifier_pipeline_fingerprint,
)
from verl.experimental.reward_loop import RewardLoopWorker, migrate_legacy_reward_impl
from verl.utils.model import compute_position_id_with_mask
from tools.on_policy_budgeted_capability_floor.build_floor_cache import _file_hash, _rows


def extract_binary_acc_from_reward_result(result: Mapping[str, Any]) -> bool:
    """Extract verifier ``acc`` only; shaped reward_score is deliberately ignored."""
    extra = result.get("reward_extra_info")
    if not isinstance(extra, Mapping) or "acc" not in extra:
        raise ValueError("reward result is missing verifier acc")
    acc = extra["acc"]
    if isinstance(acc, bool):
        return acc
    if (
        not isinstance(acc, Real)
        or isinstance(acc, bool)
        or not math.isfinite(float(acc))
        or float(acc) not in (0.0, 1.0)
    ):
        raise ValueError("reward result must contain finite binary acc")
    return bool(acc)


def _select(config: Any, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path, default=default)
    return default if value is None else value


def _verifier_fingerprint(config: Any) -> str:
    return verifier_pipeline_fingerprint(
        reward_manager_name=str(_select(config, "reward.reward_manager.name", "naive")),
        reward_manager_source=str(
            _select(config, "reward.reward_manager.source", "register")
        ),
        reward_manager_module_path=_select(config, "reward.reward_manager.module.path"),
        reward_manager_module_name=_select(config, "reward.reward_manager.module.name"),
        custom_reward_function_path=_select(
            config,
            "reward.custom_reward_function.path",
        ),
        custom_reward_function_name=str(
            _select(
                config,
                "reward.custom_reward_function.name",
                "compute_score",
            )
        ),
        custom_reward_kwargs=_select(
            config,
            "reward.custom_reward_function.reward_kwargs",
            {},
        ),
        reward_kwargs=_select(config, "reward.reward_kwargs", {}),
        sandbox_fusion={
            "url": _select(config, "reward.sandbox_fusion.url"),
            "max_concurrent": _select(
                config,
                "reward.sandbox_fusion.max_concurrent",
                64,
            ),
            "memory_limit_mb": _select(
                config,
                "reward.sandbox_fusion.memory_limit_mb",
                1024,
            ),
        },
    )


def _score_prefixes(
    *,
    worker: RewardLoopWorker,
    prefix_batch: Any,
    reference_budget: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    reward_field = f"prefix_reward_{reference_budget}"
    error_field = f"prefix_error_{reference_budget}"
    rows: list[dict[str, Any]] = []
    for start in range(0, len(prefix_batch), batch_size):
        chunk = prefix_batch[start : start + batch_size]
        try:
            outputs = worker.loop.run_until_complete(worker.compute_score_batch(chunk))
            if len(outputs) != len(chunk):
                raise ValueError("reward pipeline returned the wrong row count")
        except Exception as error:  # Preserve a row-wise failed attestation.
            outputs = [error] * len(chunk)
        for offset in range(len(chunk)):
            identity = {
                name: chunk.non_tensor_batch[name][offset].item()
                if hasattr(chunk.non_tensor_batch[name][offset], "item")
                else chunk.non_tensor_batch[name][offset]
                for name in (
                    "model_id",
                    "prompt_id",
                    "rollout_index",
                    "prompt_hash",
                    "response_hash",
                    "sampling_seed",
                    "response_token_count",
                )
            }
            output = outputs[offset]
            try:
                if isinstance(output, Exception):
                    raise output
                event = extract_binary_acc_from_reward_result(output)
                error_text = None
            except Exception as error:
                event = None
                error_text = f"{type(error).__name__}: {error}"
            rows.append(
                identity
                | {
                    reward_field: event,
                    error_field: error_text,
                    f"prefix_token_count_{reference_budget}": int(
                        chunk.batch["response_mask"][offset].sum().item()
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["model_id"], row["prompt_id"], row["rollout_index"]),
    )


def _ensure_response_width(batch: Any, *, width: int, pad_token_id: int) -> Any:
    """Pad frozen responses to B when every response ended before B."""
    current_width = int(batch.batch["responses"].shape[1])
    if current_width >= width:
        return batch
    padding_width = width - current_width
    row_count = len(batch)
    token_padding = torch.full(
        (row_count, padding_width),
        pad_token_id,
        dtype=batch.batch["responses"].dtype,
        device=batch.batch["responses"].device,
    )
    mask_padding = torch.zeros(
        (row_count, padding_width),
        dtype=batch.batch["response_mask"].dtype,
        device=batch.batch["response_mask"].device,
    )
    batch.batch["responses"] = torch.cat(
        (batch.batch["responses"], token_padding),
        dim=-1,
    )
    batch.batch["response_mask"] = torch.cat(
        (batch.batch["response_mask"], mask_padding),
        dim=-1,
    )
    batch.batch["input_ids"] = torch.cat(
        (batch.batch["prompts"], batch.batch["responses"]),
        dim=-1,
    )
    prompt_width = batch.batch["prompts"].shape[1]
    prompt_mask = batch.batch["attention_mask"][:, :prompt_width]
    batch.batch["attention_mask"] = torch.cat(
        (prompt_mask, batch.batch["response_mask"]),
        dim=-1,
    )
    batch.batch["position_ids"] = compute_position_id_with_mask(
        batch.batch["attention_mask"]
    )
    return batch


def _source_commit(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("source commit must be a full Git digest")
    return commit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    encoded = "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    path.write_text(encoded)


def _mismatch_details(
    historical_rows: list[dict[str, Any]],
    recomputed_rows: list[dict[str, Any]],
    reference_budget: int,
) -> list[dict[str, Any]]:
    identity_names = ("model_id", "prompt_id", "rollout_index")
    historical = {
        tuple(row[name] for name in identity_names): row for row in historical_rows
    }
    recomputed = {
        tuple(row[name] for name in identity_names): row for row in recomputed_rows
    }
    reward_field = f"prefix_reward_{reference_budget}"
    error_field = f"prefix_error_{reference_budget}"
    details = []
    for identity in sorted(historical.keys() & recomputed.keys()):
        old = historical[identity]
        new = recomputed[identity]
        if (
            old.get(reward_field) == new.get(reward_field)
            and not old.get(error_field)
            and not new.get(error_field)
        ):
            continue
        details.append(
            {
                "model_id": identity[0],
                "prompt_id": identity[1],
                "rollout_index": identity[2],
                "response_token_count": new.get("response_token_count"),
                "historical_prefix_reward": old.get(reward_field),
                "recomputed_prefix_reward": new.get(reward_field),
                "historical_error": old.get(error_field),
                "recomputed_error": new.get(error_field),
                "response_token_hash": new.get("response_hash"),
            }
        )
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--historical-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-budget", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for recomputed scores and event_equivalence_attestation.json",
    )
    args = parser.parse_args()
    if args.reference_budget <= 0 or args.batch_size <= 0:
        raise ValueError("reference_budget and batch_size must be positive")
    if not args.model_path.exists():
        raise ValueError("model_path must be a local Base model path")

    config = OmegaConf.load(args.resolved_config)
    config = migrate_legacy_reward_impl(config)
    with open_dict(config):
        config.actor_rollout_ref.model.path = str(args.model_path.resolve())
        config.actor_rollout_ref.model.tokenizer_path = None
    worker = RewardLoopWorker(config)
    tokenizer = worker.input_tokenizer
    tokenizer_fp, template_fp = tokenizer_fingerprints(tokenizer)
    verifier_fp = _verifier_fingerprint(config)
    protocol_fp = prefix_protocol_fingerprint(
        reference_budget=args.reference_budget,
        tokenizer_fingerprint=tokenizer_fp,
        chat_template_fingerprint=template_fp,
        verifier_fingerprint=verifier_fp,
    )

    prompts = _rows(args.prompts)
    rollouts = _rows(args.rollouts)
    historical = _rows(args.historical_scores)
    full_batch = build_full_rollout_batch_from_artifacts(
        prompt_rows=prompts,
        rollout_rows=rollouts,
        tokenizer=tokenizer,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    full_batch = _ensure_response_width(
        full_batch,
        width=args.reference_budget,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    prefix_batch = build_exact_prefix_batch(
        batch=full_batch,
        rollout_indices=torch.arange(len(full_batch), dtype=torch.long),
        reference_budget=args.reference_budget,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    recomputed = _score_prefixes(
        worker=worker,
        prefix_batch=prefix_batch,
        reference_budget=args.reference_budget,
        batch_size=args.batch_size,
    )
    report = compare_prefix_events(
        historical_rows=historical,
        recomputed_rows=recomputed,
        reference_budget=args.reference_budget,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    recomputed_path = args.output / "recomputed_prefix_scores.jsonl"
    _write_jsonl(recomputed_path, recomputed)
    repository_root = Path(__file__).resolve().parents[2]
    artifact_hashes = {
        "prompts": _file_hash(args.prompts),
        "rollouts": _file_hash(args.rollouts),
        "historical_scores": _file_hash(args.historical_scores),
        "resolved_config": _file_hash(args.resolved_config),
        "recomputed_scores": _sha256(recomputed_path),
    }
    attestation = {
        "schema_version": 1,
        "passed": report.passed,
        "prefix_protocol_version": PREFIX_PROTOCOL_VERSION,
        "prefix_protocol_fingerprint": protocol_fp,
        "reference_budget": args.reference_budget,
        "tokenizer_fingerprint": tokenizer_fp,
        "chat_template_fingerprint": template_fp,
        "verifier_fingerprint": verifier_fp,
        "prompt_manifest_fingerprint": artifact_hashes["prompts"],
        "rollout_fingerprint": artifact_hashes["rollouts"],
        "historical_score_fingerprint": artifact_hashes["historical_scores"],
        "score_fingerprint": artifact_hashes["historical_scores"],
        "artifact_hashes": artifact_hashes,
        **asdict(report),
        "mismatches": _mismatch_details(
            historical,
            recomputed,
            args.reference_budget,
        ),
        "source_git_commit": _source_commit(repository_root),
    }
    attestation_path = args.output / "event_equivalence_attestation.json"
    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(json.dumps(attestation, sort_keys=True))
    if not report.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
