#!/usr/bin/env python3
"""Evaluate witness-wise BSSF support statistics for an offline checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from verl.experimental.success_support_floor.cache import CacheExpectations, SuccessSupportCache
from verl.experimental.success_support_floor.logprobs import load_reference_model, sequence_logprobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
    prompt_by_key = {row["prompt_key"]: row for row in cache.prompts}
    examples = [
        (prompt_by_key[row["prompt_key"]]["prompt_token_ids"], row["response_token_ids"])
        for row in cache.witnesses
    ]
    model, tokenizer, device = load_reference_model(
        args.checkpoint_path, args.tokenizer_path, device=args.device
    )
    current: list[float] = []
    for start in range(0, len(examples), args.batch_size):
        current.extend(
            sequence_logprobs(
                model,
                examples[start : start + args.batch_size],
                pad_token_id=tokenizer.pad_token_id,
                temperature=manifest["logprob_temperature"],
                device=device,
            )
        )
    reference = np.asarray([row["reference_seq_logprob"] for row in cache.witnesses], dtype=np.float64)
    ratios = np.asarray(current, dtype=np.float64) - reference
    shortfall = np.maximum(math.log(args.alpha) - ratios, 0.0)
    report = {
        "cache_fingerprint": cache.fingerprint,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "alpha": args.alpha,
        "witness_count": len(ratios),
        "log_ratio_mean": float(ratios.mean()),
        "log_ratio_p10": float(np.quantile(ratios, 0.1)),
        "log_ratio_p50": float(np.quantile(ratios, 0.5)),
        "log_ratio_p90": float(np.quantile(ratios, 0.9)),
        "shortfall_mean": float(shortfall.mean()),
        "active_fraction": float((shortfall > 0).mean()),
        "non_finite_count": int((~np.isfinite(ratios)).sum()),
        "note": "group/bootstrap/AUROC analyses require a prompt transition annotation file and are reported downstream",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["non_finite_count"]:
        raise SystemExit("non-finite support score")


if __name__ == "__main__":
    main()
