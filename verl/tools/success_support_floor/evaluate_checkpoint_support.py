#!/usr/bin/env python3
"""Evaluate one or more checkpoints against a BSSF cache with paired prompt statistics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from verl.experimental.success_support_floor.analysis import (
    binary_auroc,
    paired_bootstrap_mean_difference,
    spearman_correlation,
)
from verl.experimental.success_support_floor.cache import (
    CacheExpectations,
    SuccessSupportCache,
    tokenizer_fingerprints,
)
from verl.experimental.success_support_floor.logprobs import load_reference_model, sequence_logprobs


def _summary(log_ratio: np.ndarray, shortfall: np.ndarray) -> dict[str, float | int]:
    if len(log_ratio) == 0:
        return {"count": 0}
    return {
        "count": len(log_ratio),
        "log_ratio_mean": float(log_ratio.mean()),
        "log_ratio_p10": float(np.quantile(log_ratio, 0.1)),
        "log_ratio_p50": float(np.quantile(log_ratio, 0.5)),
        "log_ratio_p90": float(np.quantile(log_ratio, 0.9)),
        "shortfall_mean": float(shortfall.mean()),
        "active_fraction": float((shortfall > 0).mean()),
    }


def _parse_checkpoints(values: list[str], legacy_path: str | None) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    if legacy_path:
        parsed.append(("checkpoint", legacy_path))
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must have LABEL=PATH form")
        label, path = value.split("=", 1)
        if not label or not path:
            raise ValueError("--checkpoint must have nonempty LABEL=PATH")
        parsed.append((label, path))
    if not parsed:
        raise ValueError("provide --checkpoint-path or at least one --checkpoint LABEL=PATH")
    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    return parsed


def _annotations(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return pq.read_table(path).to_pylist()


def _annotation_map(
    rows: list[dict[str, Any]], cache: SuccessSupportCache
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    prompt_key_by_id = {int(row["prompt_id"]): row["prompt_key"] for row in cache.prompts}
    output = {}
    for row in rows:
        key = row.get("prompt_key")
        if key is None and row.get("prompt_id") is not None:
            key = prompt_key_by_id.get(int(row["prompt_id"]))
        if key in prompt_key_by_id.values():
            output[str(key)] = row
    return output


def _score_checkpoint(
    *,
    label: str,
    path: str,
    cache: SuccessSupportCache,
    manifest: dict[str, Any],
    tokenizer_path: str | None,
    alpha: float,
    batch_size: int,
    device_name: str,
    annotation_by_key: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    prompt_by_key = {row["prompt_key"]: row for row in cache.prompts}
    examples = [
        (prompt_by_key[row["prompt_key"]]["prompt_token_ids"], row["response_token_ids"])
        for row in cache.witnesses
    ]
    model, tokenizer, device = load_reference_model(path, tokenizer_path, device=device_name)
    tokenizer_fp, template_fp = tokenizer_fingerprints(tokenizer)
    if tokenizer_fp != manifest["tokenizer_fingerprint"]:
        raise ValueError(f"checkpoint {label} tokenizer fingerprint does not match cache")
    if template_fp != manifest["chat_template_fingerprint"]:
        raise ValueError(f"checkpoint {label} chat template fingerprint does not match cache")
    current: list[float] = []
    for start in range(0, len(examples), batch_size):
        current.extend(
            sequence_logprobs(
                model,
                examples[start : start + batch_size],
                pad_token_id=tokenizer.pad_token_id,
                temperature=manifest["logprob_temperature"],
                device=device,
            )
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reference = np.asarray(
        [row["reference_seq_logprob"] for row in cache.witnesses], dtype=np.float64
    )
    ratios = np.asarray(current, dtype=np.float64) - reference
    shortfalls = np.maximum(math.log(alpha) - ratios, 0.0)
    if not np.isfinite(ratios).all():
        raise ValueError(f"checkpoint {label} produced non-finite support scores")

    witness_indices: dict[str, list[int]] = defaultdict(list)
    for index, witness in enumerate(cache.witnesses):
        witness_indices[witness["prompt_key"]].append(index)
    prompt_scores: dict[str, dict[str, float]] = {}
    for key, indices in witness_indices.items():
        prompt_scores[key] = {
            "log_ratio": float(ratios[indices].mean()),
            "shortfall": float(shortfalls[indices].mean()),
        }

    support_strata = {}
    for threshold in (1, 2, 4):
        keys = [
            row["prompt_key"]
            for row in cache.prompts
            if int(row["eligible_success_count"]) >= threshold
        ]
        support_strata[f"S_base_{threshold}"] = _summary(
            np.asarray([prompt_scores[key]["log_ratio"] for key in keys]),
            np.asarray([prompt_scores[key]["shortfall"] for key in keys]),
        )

    transition_groups: dict[str, list[str]] = defaultdict(list)
    for key, annotation in annotation_by_key.items():
        group = annotation.get("transition_group") or annotation.get("group")
        if group in {"retained", "delayed", "lost"} and key in prompt_scores:
            transition_groups[str(group)].append(key)
    group_report = {
        group: _summary(
            np.asarray([prompt_scores[key]["log_ratio"] for key in keys]),
            np.asarray([prompt_scores[key]["shortfall"] for key in keys]),
        )
        for group, keys in sorted(transition_groups.items())
    }
    annotated_keys = [key for keys in transition_groups.values() for key in keys]
    auroc = None
    if {"retained", "delayed", "lost"}.intersection(transition_groups) and len(annotated_keys) >= 2:
        labels = [
            (annotation_by_key[key].get("transition_group") or annotation_by_key[key].get("group"))
            == "retained"
            for key in annotated_keys
        ]
        if any(labels) and not all(labels):
            auroc = binary_auroc([prompt_scores[key]["log_ratio"] for key in annotated_keys], labels)

    correlation = None
    correlation_keys = []
    q_shifts = []
    for key, annotation in annotation_by_key.items():
        value = annotation.get("q2048_shift")
        if value is None:
            value = annotation.get("observed_q2048_shift")
        if key in prompt_scores and value is not None and np.isfinite(float(value)):
            correlation_keys.append(key)
            q_shifts.append(float(value))
    if len(correlation_keys) >= 2:
        try:
            correlation = spearman_correlation(
                [prompt_scores[key]["log_ratio"] for key in correlation_keys], q_shifts
            )
        except ValueError:
            correlation = None

    report = {
        "label": label,
        "checkpoint_path": str(Path(path).resolve()),
        "witness": _summary(ratios, shortfalls),
        "prompt": _summary(
            np.asarray([value["log_ratio"] for value in prompt_scores.values()]),
            np.asarray([value["shortfall"] for value in prompt_scores.values()]),
        ),
        "support_strata": support_strata,
        "transition_groups": group_report,
        "retained_vs_delayed_lost_auroc": auroc,
        "support_vs_q2048_shift_spearman": correlation,
        "annotated_prompt_count": len(annotation_by_key),
        "non_finite_count": 0,
    }
    return report, prompt_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--checkpoint-path", help="Legacy single-checkpoint path")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for Base/H2048/H4096 paired comparisons",
    )
    parser.add_argument("--prompt-annotations", help="Parquet with prompt key/id, transition group, and q shift")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    checkpoints = _parse_checkpoints(args.checkpoint, args.checkpoint_path)
    root = Path(args.cache_path)
    manifest = json.loads((root / "manifest.json").read_text())
    cache = SuccessSupportCache.load(
        root,
        CacheExpectations(
            reference_budget=manifest["reference_budget"],
            support_threshold=manifest["support_threshold"],
            tokenizer_fingerprint=manifest["tokenizer_fingerprint"],
            chat_template_fingerprint=manifest["chat_template_fingerprint"],
            logprob_temperature=manifest["logprob_temperature"],
            include_eos=manifest["include_eos"],
        ),
    )
    annotation_by_key = _annotation_map(_annotations(args.prompt_annotations), cache)
    checkpoint_reports = {}
    prompt_scores_by_label = {}
    for label, path in checkpoints:
        checkpoint_reports[label], prompt_scores_by_label[label] = _score_checkpoint(
            label=label,
            path=path,
            cache=cache,
            manifest=manifest,
            tokenizer_path=args.tokenizer_path,
            alpha=args.alpha,
            batch_size=args.batch_size,
            device_name=args.device,
            annotation_by_key=annotation_by_key,
        )

    paired = {}
    labels = [label for label, _ in checkpoints]
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            keys = sorted(set(prompt_scores_by_label[left]) & set(prompt_scores_by_label[right]))
            if len(keys) < 2:
                continue
            paired[f"{left}_minus_{right}"] = {
                "log_ratio": paired_bootstrap_mean_difference(
                    [prompt_scores_by_label[left][key]["log_ratio"] for key in keys],
                    [prompt_scores_by_label[right][key]["log_ratio"] for key in keys],
                    seed=args.bootstrap_seed,
                    resamples=args.bootstrap_resamples,
                ),
                "shortfall": paired_bootstrap_mean_difference(
                    [prompt_scores_by_label[left][key]["shortfall"] for key in keys],
                    [prompt_scores_by_label[right][key]["shortfall"] for key in keys],
                    seed=args.bootstrap_seed,
                    resamples=args.bootstrap_resamples,
                ),
            }
    report = {
        "cache_fingerprint": cache.fingerprint,
        "alpha": args.alpha,
        "checkpoints": checkpoint_reports,
        "paired_prompt_bootstrap": paired,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
