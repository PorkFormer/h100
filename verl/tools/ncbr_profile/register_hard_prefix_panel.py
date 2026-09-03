#!/usr/bin/env python3
"""Freeze a deterministic H=2048 cap-prefix panel from Base rollout JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"JSONL row {line_number} is not an object")
            yield value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--response-length", type=int, default=2048)
    args = parser.parse_args()
    if args.count < 20:
        raise SystemExit("mechanism panel must contain at least 20 requests")
    required = {"prompt_id", "prompt_token_ids", "response_token_ids", "trajectory_id", "finish_reason"}
    eligible = []
    for row in iter_jsonl(args.input):
        missing = required - row.keys()
        if missing:
            raise SystemExit(f"panel source row is missing fields: {sorted(missing)}")
        response = [int(token) for token in row["response_token_ids"]]
        if str(row["finish_reason"]).lower() != "length" or len(response) != args.response_length:
            continue
        eligible.append(
            {
                "prompt_id": str(row["prompt_id"]),
                "prompt_token_ids": [int(token) for token in row["prompt_token_ids"]],
                "prefix_token_ids": response,
                "trajectory_id": str(row["trajectory_id"]),
            }
        )
    eligible.sort(key=lambda row: (row["prompt_id"], row["trajectory_id"]))
    panel = eligible[: args.count]
    if len(panel) != args.count:
        raise SystemExit(f"insufficient H={args.response_length} cap-prefix rows: {len(panel)} < {args.count}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite a frozen mechanism panel")
    with args.output.open("x", encoding="utf-8") as stream:
        for row in panel:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "qwen3-1p7b-hard-prefix-panel-v1",
        "source": str(args.input.resolve()),
        "source_sha256": file_sha256(args.input),
        "panel": str(args.output.resolve()),
        "panel_sha256": file_sha256(args.output),
        "model_revision": args.model_revision,
        "seed": args.seed,
        "response_length": args.response_length,
        "request_count": len(panel),
        "selection": "lexicographic_prompt_id_then_trajectory_id",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
