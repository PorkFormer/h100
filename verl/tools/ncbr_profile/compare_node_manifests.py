#!/usr/bin/env python3
"""Fail closed unless two local-node manifests bind identical immutable inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-a", type=Path, required=True)
    parser.add_argument("--node-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in (args.node_a, args.node_b)]
    left, right = manifests
    checks = {
        "different_node_labels": left.get("node") != right.get("node"),
        "same_code_sha": left.get("code_sha") == right.get("code_sha"),
        "same_arm": left.get("arm") == right.get("arm"),
        "same_candidate": left.get("candidate") == right.get("candidate"),
        "same_stage": left.get("stage") == right.get("stage"),
        "same_diagnostics_mode": left.get("diagnostics") == right.get("diagnostics"),
        "same_model_identity": {key: left["model"].get(key) for key in ("repo_id", "revision", "model_type")}
        == {key: right["model"].get(key) for key in ("repo_id", "revision", "model_type")},
        "same_model_file_hashes": left["model"].get("files") == right["model"].get("files"),
        "same_model_revision_provenance": left["model"].get("revision_provenance")
        == right["model"].get("revision_provenance"),
        "same_data_hashes": {key: value.get("sha256") for key, value in left.get("data", {}).items()}
        == {key: value.get("sha256") for key, value in right.get("data", {}).items()},
        "same_frozen_artifact_hashes": {
            key: value.get("sha256") for key, value in left.get("frozen_artifacts", {}).items()
        }
        == {key: value.get("sha256") for key, value in right.get("frozen_artifacts", {}).items()},
    }
    result = {
        "schema_version": "qwen3-1p7b-cross-node-manifest-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "node_paths": {
            str(left.get("node")): {
                "model": left["model"].get("local_path"),
                "data": {key: value.get("path") for key, value in left.get("data", {}).items()},
            },
            str(right.get("node")): {
                "model": right["model"].get("local_path"),
                "data": {key: value.get("path") for key, value in right.get("data", {}).items()},
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(f"cross-node immutable-input gate failed: {checks}")


if __name__ == "__main__":
    main()
