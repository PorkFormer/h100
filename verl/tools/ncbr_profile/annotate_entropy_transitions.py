#!/usr/bin/env python3
"""Attach Base-to-checkpoint answer transitions without pooling benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def annotate(base_rows: list[dict], checkpoint_rows: list[dict], threshold: float = 0.5) -> list[dict]:
    def identity(row):
        return str(row["benchmark"]), str(row["prompt_id"]), int(row["rollout_index"])

    base = {identity(row): row for row in base_rows}
    checkpoint = {identity(row): row for row in checkpoint_rows}
    if len(base) != len(base_rows) or len(checkpoint) != len(checkpoint_rows) or base.keys() != checkpoint.keys():
        raise ValueError("Base/checkpoint entropy rollout identities differ or contain duplicates")
    output = []
    for key in sorted(base):
        if key[0] not in {"AIME2024", "AIME2025"}:
            raise ValueError(f"unsupported or pooled benchmark label: {key[0]}")
        old = float(base[key]["correctness"]) >= threshold
        new = float(checkpoint[key]["correctness"]) >= threshold
        transition = {
            (False, False): "wrong_to_wrong",
            (False, True): "wrong_to_correct",
            (True, True): "correct_to_correct",
            (True, False): "correct_to_wrong",
        }[(old, new)]
        output.append(
            {
                **checkpoint[key],
                "base_correctness": float(base[key]["correctness"]),
                "answer_transition": transition,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite annotated entropy rollouts")
    base = [json.loads(line) for line in args.base.read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint = [
        json.loads(line) for line in args.checkpoint.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = annotate(base, checkpoint, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
