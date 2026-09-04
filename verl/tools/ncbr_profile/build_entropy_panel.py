#!/usr/bin/env python3
"""Freeze the prescribed four rollout indices per AIME question."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROLLOUT_INDICES = (0, 8, 16, 24)
BENCHMARKS = ("AIME2024", "AIME2025")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"benchmark", "prompt_id", "rollout_index", "prompt_token_ids", "response_token_ids"}
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    prompts: dict[str, set[str]] = {benchmark: set() for benchmark in BENCHMARKS}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"entropy rollout row is missing fields: {sorted(missing)}")
        benchmark = str(row["benchmark"])
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unsupported or pooled benchmark label: {benchmark}")
        prompt_id = str(row["prompt_id"])
        prompts[benchmark].add(prompt_id)
        rollout_index = int(row["rollout_index"])
        if rollout_index not in ROLLOUT_INDICES:
            continue
        key = (benchmark, prompt_id, rollout_index)
        if key in selected:
            raise ValueError(f"duplicate entropy rollout identity: {key}")
        normalized = dict(row)
        normalized["benchmark"] = benchmark
        normalized["prompt_id"] = prompt_id
        normalized["rollout_index"] = rollout_index
        normalized["prompt_token_ids"] = [int(token) for token in row["prompt_token_ids"]]
        normalized["response_token_ids"] = [int(token) for token in row["response_token_ids"]]
        if not normalized["prompt_token_ids"] or not normalized["response_token_ids"]:
            raise ValueError(f"empty entropy token sequence: {key}")
        selected[key] = normalized
    missing = [
        (benchmark, prompt_id, index)
        for benchmark in BENCHMARKS
        for prompt_id in sorted(prompts[benchmark])
        for index in ROLLOUT_INDICES
        if (benchmark, prompt_id, index) not in selected
    ]
    if missing:
        raise ValueError(f"entropy panel is missing prescribed rollout identities: {missing[:10]}")
    if any(not prompts[benchmark] for benchmark in BENCHMARKS):
        raise ValueError("AIME2024 and AIME2025 must both be present and remain separate")
    return [selected[key] for key in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-kind", choices=("on_policy", "shared_base_teacher_forced"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite a frozen entropy panel")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "qwen3-1p7b-checkpoint-entropy-panel-v1",
        "panel_kind": args.panel_kind,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "rollout_indices": list(ROLLOUT_INDICES),
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "panel": str(args.output.resolve()),
        "panel_sha256": sha256(args.output),
        "benchmarks": {
            benchmark: {
                "prompt_count": len({row["prompt_id"] for row in selected if row["benchmark"] == benchmark}),
                "trajectory_count": sum(row["benchmark"] == benchmark for row in selected),
            }
            for benchmark in BENCHMARKS
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
