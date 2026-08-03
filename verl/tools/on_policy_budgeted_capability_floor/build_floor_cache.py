"""Build a strict prompt-level OBCF floor cache from Base audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow.parquet as pq

from verl.experimental.capability_constraints.identity import reference_model_fingerprint
from verl.experimental.on_policy_budgeted_capability_floor.cache import build_floor_rows, write_cache


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a row list")
    return payload


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-fingerprint", required=True)
    parser.add_argument("--chat-template-fingerprint", required=True)
    parser.add_argument("--verifier-fingerprint", required=True)
    parser.add_argument("--reference-budget", type=int, default=2048)
    parser.add_argument("--base-rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--support-threshold", type=int, default=2)
    parser.add_argument("--reference-tolerance-count", type=int, default=1)
    parser.add_argument("--source-git-commit", required=True)
    args = parser.parse_args()

    prompts = _rows(args.prompts)
    rows = build_floor_rows(
        prompts=prompts,
        rollouts=_rows(args.rollouts),
        scores=_rows(args.scores),
        model_id=args.model_id,
        tokenizer_fingerprint=args.tokenizer_fingerprint,
        chat_template_fingerprint=args.chat_template_fingerprint,
        reference_budget=args.reference_budget,
        base_rollouts_per_prompt=args.base_rollouts_per_prompt,
        support_threshold=args.support_threshold,
        reference_tolerance_count=args.reference_tolerance_count,
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
        "tokenizer_fingerprint": args.tokenizer_fingerprint,
        "chat_template_fingerprint": args.chat_template_fingerprint,
        "prompt_manifest_fingerprint": _file_hash(args.prompts),
        "rollout_fingerprint": _file_hash(args.rollouts),
        "score_fingerprint": _file_hash(args.scores),
        "verifier_fingerprint": args.verifier_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": args.source_git_commit,
        "prompt_count": len(prompts),
    }
    fingerprint = write_cache(args.output, manifest, rows)
    print(json.dumps({"cache_fingerprint": fingerprint, "protected_prompt_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
