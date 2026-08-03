#!/usr/bin/env python3
"""Recompute cached Base witness log probabilities and enforce calibrated tolerances."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from verl.experimental.success_support_floor.cache import (
    CacheExpectations,
    SuccessSupportCache,
    reference_model_fingerprint,
    tokenizer_fingerprints,
)
from verl.experimental.success_support_floor.logprobs import load_reference_model, sequence_logprobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--reference-model-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--mean-threshold", type=float, default=1e-5)
    parser.add_argument("--max-threshold", type=float, default=1e-4)
    args = parser.parse_args()
    if args.sample_size <= 0 or args.batch_size <= 0:
        raise ValueError("sample size and batch size must be positive")
    if args.mean_threshold < 0.0 or args.max_threshold < 0.0:
        raise ValueError("validation thresholds must be nonnegative")
    manifest = json.loads((Path(args.cache_path) / "manifest.json").read_text())
    expected = CacheExpectations(
        reference_budget=manifest["reference_budget"],
        support_threshold=manifest["support_threshold"],
        tokenizer_fingerprint=manifest["tokenizer_fingerprint"],
        chat_template_fingerprint=manifest["chat_template_fingerprint"],
        logprob_temperature=manifest["logprob_temperature"],
        include_eos=manifest["include_eos"],
    )
    cache = SuccessSupportCache.load(args.cache_path, expected)
    actual_model_hash = reference_model_fingerprint(args.reference_model_path)
    if actual_model_hash != manifest["reference_model_hash"]:
        raise ValueError("reference model weight hash does not match cache manifest")
    generator = random.Random(args.seed)
    rows = generator.sample(cache.witnesses, min(args.sample_size, len(cache.witnesses)))
    prompt_by_key = {row["prompt_key"]: row for row in cache.prompts}
    model, tokenizer, device = load_reference_model(
        args.reference_model_path, args.tokenizer_path, device=args.device
    )
    tokenizer_fp, template_fp = tokenizer_fingerprints(tokenizer)
    if tokenizer_fp != manifest["tokenizer_fingerprint"]:
        raise ValueError("loaded tokenizer fingerprint does not match cache")
    if template_fp != manifest["chat_template_fingerprint"]:
        raise ValueError("loaded chat template fingerprint does not match cache")
    recomputed: list[float] = []
    examples = [
        (prompt_by_key[row["prompt_key"]]["prompt_token_ids"], row["response_token_ids"])
        for row in rows
    ]
    for start in range(0, len(examples), args.batch_size):
        recomputed.extend(
            sequence_logprobs(
                model,
                examples[start : start + args.batch_size],
                pad_token_id=tokenizer.pad_token_id,
                temperature=manifest["logprob_temperature"],
                device=device,
            )
        )
    errors = [
        abs(value - float(row["reference_seq_logprob"])) / int(row["response_token_count"])
        for row, value in zip(rows, recomputed, strict=True)
    ]
    report = {
        "cache_fingerprint": cache.fingerprint,
        "reference_model_hash": actual_model_hash,
        "sample_count": len(rows),
        "mean_absolute_per_token_error": sum(errors) / len(errors),
        "max_absolute_per_token_error": max(errors),
        "non_finite_count": 0,
        "token_mismatch_count": 0,
        "mean_threshold": args.mean_threshold,
        "max_threshold": args.max_threshold,
    }
    report["passed"] = bool(
        report["mean_absolute_per_token_error"] <= args.mean_threshold
        and report["max_absolute_per_token_error"] <= args.max_threshold
    )
    path = Path(args.cache_path) / "validation_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("cached Base log-probability calibration failed")


if __name__ == "__main__":
    main()
