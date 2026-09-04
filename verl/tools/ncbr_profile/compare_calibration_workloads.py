#!/usr/bin/env python3
"""Prove that two calibration launches processed the same token workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical_record(boundary: int, record: dict[str, Any]) -> dict[str, Any]:
    common = {
        "generation_batch_index": int(record["generation_batch_index"]),
        "request_order": int(record["request_order"]),
        "prompt_token_hash": record["prompt_token_hash"],
        "prompt_token_count": int(record["prompt_token_count"]),
        "rollout_index": int(record["rollout_index"]),
    }
    if boundary == 0:
        return common
    if boundary == 1:
        return {
            **common,
            "response_token_ids_hash": record["response_token_ids_hash"],
            "response_length": int(record["response_length"]),
            "finish_reason": record.get("finish_reason"),
        }
    if boundary == 2:
        return {
            **common,
            "response_hash": record["response_hash"],
            "terminal_reward": record.get("terminal_reward"),
            "group_reward_pattern": record.get("group_reward_pattern"),
            "candidate_decision": record.get("candidate_decision"),
            "filter_metric": record.get("filter_metric"),
        }
    if boundary == 3:
        return {
            **common,
            "retained_response_hash": record["retained_response_hash"],
            "retained_order": int(record["retained_order"]),
            "effective_training_batch": bool(record["effective_training_batch"]),
        }
    raise ValueError(f"unsupported boundary: {boundary}")


def workload_manifest(root: Path) -> dict[str, Any]:
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"no calibration boundary manifests found under {root}")
    batches: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, bool]] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE":
            raise ValueError(f"incomplete calibration boundary: {manifest_path}")
        boundary = int(manifest["boundary"])
        if boundary not in {0, 1, 2, 3}:
            raise ValueError(f"unexpected calibration boundary: {boundary}")
        identity = (
            int(manifest["global_step"]),
            int(manifest["generation_batch_index"]),
            boundary,
            bool(manifest["effective_training_batch"]),
        )
        if identity in seen:
            raise ValueError(f"duplicate calibration boundary identity: {identity}")
        seen.add(identity)
        records_path = manifest_path.with_name("records.jsonl")
        raw = records_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["records_sha256"]:
            raise ValueError(f"records hash mismatch: {records_path}")
        records = [_canonical_record(boundary, item) for item in _records(records_path)]
        if len(records) != int(manifest["record_count"]):
            raise ValueError(f"record count mismatch: {records_path}")
        batches.append(
            {
                "global_step": identity[0],
                "generation_batch_index": identity[1],
                "boundary": boundary,
                "effective_training_batch": identity[3],
                "record_count": len(records),
                "records_sha256": _sha256_json(records),
            }
        )
    batches.sort(
        key=lambda item: (
            item["global_step"],
            item["generation_batch_index"],
            item["boundary"],
            item["effective_training_batch"],
        )
    )
    return {
        "schema_version": "qwen3-1p7b-calibration-workload-v1",
        "batch_count": len(batches),
        "batches": batches,
        "workload_sha256": _sha256_json(batches),
    }


def compare(node_a: Path, node_b: Path) -> dict[str, Any]:
    left = workload_manifest(node_a)
    right = workload_manifest(node_b)
    equal = left["workload_sha256"] == right["workload_sha256"] and left["batches"] == right["batches"]
    return {
        "schema_version": "qwen3-1p7b-calibration-workload-comparison-v1",
        "status": "PASS" if equal else "FAIL",
        "exact_prompt_response_reward_and_retained_workload_match": equal,
        "node_A": left,
        "node_B": right,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-a", type=Path, required=True)
    parser.add_argument("--node-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.node_a, args.node_b)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit("cross-node calibration workloads differ")


if __name__ == "__main__":
    main()
