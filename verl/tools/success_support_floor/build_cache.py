#!/usr/bin/env python3
"""Build a verifier-certified BSSF cache and calibrate Base log probabilities."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from verl.experimental.success_support_floor.cache import (
    canonical_prompt_key,
    reference_model_fingerprint,
    tokenizer_fingerprints,
    witness_is_eligible,
    write_cache,
)
from verl.experimental.success_support_floor.logprobs import load_reference_model, sequence_logprobs


def _files(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [Path(name) for name in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern!r}")
    return files


def _read(pattern: str) -> list[dict[str, Any]]:
    tables = [pq.read_table(path) for path in _files(pattern)]
    return pa.concat_tables(tables, promote_options="default").to_pylist()


def _file_set_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-glob", required=True)
    parser.add_argument("--score-glob", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--reference-model-path", required=True)
    parser.add_argument("--reference-budget", type=int, default=2048)
    parser.add_argument("--base-rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--support-threshold", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--logprob-temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-eos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if not args.strict:
        raise ValueError("the first BSSF cache builder requires --strict")
    if (
        args.reference_budget <= 0
        or args.base_rollouts_per_prompt <= 0
        or args.support_threshold <= 0
        or args.batch_size <= 0
    ):
        raise ValueError("budget, threshold, and batch size must be positive")
    if not args.include_eos:
        raise ValueError("the first BSSF cache convention requires --include-eos")
    rollout_files = _files(args.rollout_glob)
    score_files = _files(args.score_glob)
    prompt_file = Path(args.prompt_manifest)
    rollouts = _read(args.rollout_glob)
    scores = _read(args.score_glob)
    prompt_rows = pq.read_table(prompt_file).to_pylist()
    model_fp = reference_model_fingerprint(args.reference_model_path)
    score_by_key = {}
    for row in scores:
        key = (row.get("model_id"), int(row["prompt_id"]), int(row["rollout_index"]))
        if key in score_by_key:
            raise ValueError(f"duplicate score identity {key}")
        score_by_key[key] = row

    model, tokenizer, device = load_reference_model(
        args.reference_model_path, args.tokenizer_path, device=args.device
    )
    tokenizer_fp, template_fp = tokenizer_fingerprints(tokenizer)
    prompt_by_id = {}
    for row in prompt_rows:
        prompt_id = int(row["prompt_id"])
        if prompt_id in prompt_by_id:
            raise ValueError(f"duplicate prompt_id {prompt_id}")
        prompt_by_id[prompt_id] = row
    rollout_keys = set()
    rollout_counts = {prompt_id: 0 for prompt_id in prompt_by_id}
    eligible_by_prompt: dict[int, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        key = (rollout.get("model_id"), int(rollout["prompt_id"]), int(rollout["rollout_index"]))
        if key in rollout_keys:
            raise ValueError(f"duplicate rollout identity {key}")
        rollout_keys.add(key)
        prompt_id = int(rollout["prompt_id"])
        if prompt_id not in prompt_by_id:
            raise ValueError(f"rollout references unknown prompt_id {prompt_id}")
        rollout_counts[prompt_id] += 1
        score = score_by_key.get(key)
        if score is None:
            raise ValueError(f"rollout is missing its score row: {key}")
        response_tokens = [int(token) for token in rollout["response_token_ids"]]
        if int(rollout["response_token_count"]) != len(response_tokens):
            raise ValueError(f"response_token_count mismatch for rollout {key}")
        budget = args.reference_budget
        prefix_reward = score.get(f"prefix_reward_{budget}")
        prefix_error = score.get(f"prefix_error_{budget}")
        eligible = witness_is_eligible(
            full_reward=bool(score.get("full_reward")),
            prefix_reward=bool(prefix_reward),
            response_token_count=int(rollout["response_token_count"]),
            reference_budget=budget,
            hit_token_cap=bool(rollout.get("hit_token_cap")),
            finish_reason=rollout.get("finish_reason") or "",
            generation_error=rollout.get("generation_error"),
            verifier_error=prefix_error or score.get("full_error"),
        )
        if eligible:
            eligible_by_prompt.setdefault(prompt_id, []).append(rollout | {"_score": score})

    wrong_counts = {
        prompt_id: count
        for prompt_id, count in rollout_counts.items()
        if count != args.base_rollouts_per_prompt
    }
    if wrong_counts:
        sample = list(sorted(wrong_counts.items()))[:10]
        raise ValueError(
            "Base rollout count does not match --base-rollouts-per-prompt; "
            f"first mismatches: {sample}"
        )

    protected_ids = sorted(
        prompt_id for prompt_id, rows in eligible_by_prompt.items() if len(rows) >= args.support_threshold
    )
    if not protected_ids:
        raise ValueError("no protected prompts satisfy the support threshold")
    cache_prompts: list[dict[str, Any]] = []
    cache_witnesses: list[dict[str, Any]] = []
    scoring_inputs: list[tuple[list[int], list[int]]] = []
    rollout_logprobs: list[float | None] = []
    for prompt_id in protected_ids:
        prompt = prompt_by_id[prompt_id]
        prompt_tokens = [int(token) for token in prompt["prompt_token_ids"]]
        prompt_key = canonical_prompt_key(tokenizer_fp, template_fp, prompt_tokens)
        rows = sorted(eligible_by_prompt[prompt_id], key=lambda row: int(row["rollout_index"]))
        base_count = rollout_counts[prompt_id]
        cache_prompts.append(
            {
                "prompt_key": prompt_key,
                "prompt_id": prompt_id,
                "original_dataset_index": int(prompt["original_dataset_index"]),
                "prompt_hash": str(prompt["prompt_hash"]),
                "prompt_token_ids": prompt_tokens,
                "prompt_token_count": len(prompt_tokens),
                "base_rollout_count": base_count,
                "eligible_success_count": len(rows),
                "q_reference": len(rows) / base_count,
            }
        )
        for witness_id, row in enumerate(rows):
            response = [int(token) for token in row["response_token_ids"]]
            scoring_inputs.append((prompt_tokens, response))
            cumulative = row.get("cumulative_logprob")
            rollout_logprobs.append(float(cumulative) if cumulative is not None else None)
            cache_witnesses.append(
                {
                    "prompt_key": prompt_key,
                    "witness_id": witness_id,
                    "source_rollout_index": int(row["rollout_index"]),
                    "response_token_ids": response,
                    "response_token_count": len(response),
                    "reference_seq_logprob": 0.0,
                    "reference_mean_logprob": 0.0,
                    "finish_reason": str(row["finish_reason"]),
                    "full_reward": True,
                    "prefix_reward_reference_budget": True,
                    "response_hash": hashlib.sha256(bytes(json.dumps(response), "utf-8")).hexdigest(),
                }
            )
    reference_values: list[float] = []
    for start in range(0, len(scoring_inputs), args.batch_size):
        reference_values.extend(
            sequence_logprobs(
                model,
                scoring_inputs[start : start + args.batch_size],
                pad_token_id=tokenizer.pad_token_id,
                temperature=args.logprob_temperature,
                device=device,
            )
        )
    for witness, value in zip(cache_witnesses, reference_values, strict=True):
        witness["reference_seq_logprob"] = value
        witness["reference_mean_logprob"] = value / witness["response_token_count"]

    manifest = {
        "schema_version": 1,
        "algorithm": "budgeted_success_support_floor",
        "reference_model_id": str(args.reference_model_path),
        "reference_model_hash": model_fp,
        "reference_budget": args.reference_budget,
        "base_rollouts_per_prompt": args.base_rollouts_per_prompt,
        "support_threshold": args.support_threshold,
        "tokenizer_fingerprint": tokenizer_fp,
        "chat_template_fingerprint": template_fp,
        "prompt_manifest_fingerprint": _file_set_fingerprint([prompt_file]),
        "verifier_fingerprint": _file_set_fingerprint(score_files),
        "rollout_fingerprint": _file_set_fingerprint(rollout_files),
        "logprob_temperature": args.logprob_temperature,
        "logprob_convention": "response-token-sum",
        "include_eos": args.include_eos,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_git_commit": _git_commit(),
        "prompt_count": len(prompt_rows),
    }
    output = Path(args.output_dir)
    fingerprint = write_cache(output, manifest, cache_prompts, cache_witnesses)
    differences = [
        abs(reference - rollout) / len(scoring_inputs[index][1])
        for index, (reference, rollout) in enumerate(zip(reference_values, rollout_logprobs, strict=True))
        if rollout is not None
    ]
    report = {
        "cache_fingerprint": fingerprint,
        "protected_prompt_count": len(cache_prompts),
        "witness_count": len(cache_witnesses),
        "rollout_cumulative_logprob_comparison_count": len(differences),
        "rollout_cumulative_logprob_mean_absolute_per_token_error": (
            sum(differences) / len(differences) if differences else None
        ),
        "rollout_cumulative_logprob_max_absolute_per_token_error": max(differences, default=None),
        "note": "rollout cumulative_logprob is diagnostic only; cache values are teacher-forced",
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
