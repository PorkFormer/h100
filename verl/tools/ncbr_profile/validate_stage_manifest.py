#!/usr/bin/env python3
"""Verify the complete stage manifest immediately before any GPU launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and ".cache" not in path.relative_to(root).parts
    }


def revision_metadata(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted((root / ".cache" / "huggingface" / "download").glob("*.metadata"))
        if path.is_file() and not path.is_symlink()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline", "v1"), required=True)
    parser.add_argument("--candidate", choices=("P0", "P1", "P2"), required=True)
    parser.add_argument(
        "--stage", choices=("calibration", "gate0", "profile", "acceptance", "formal_s300"), required=True
    )
    parser.add_argument("--node", choices=("A", "B"), required=True)
    parser.add_argument("--diagnostics", choices=("on", "off"), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_identity = (args.arm, args.candidate, args.stage)
    actual_identity = (manifest.get("arm"), manifest.get("candidate"), manifest.get("stage"))
    if actual_identity != expected_identity:
        raise SystemExit(f"stage identity mismatch: expected {expected_identity}, got {actual_identity}")
    if manifest.get("node") != args.node:
        raise SystemExit(f"stage node mismatch: expected {args.node}, got {manifest.get('node')}")
    if manifest.get("diagnostics") != args.diagnostics:
        raise SystemExit(f"diagnostics mode mismatch: expected {args.diagnostics}, got {manifest.get('diagnostics')}")
    repo = args.repo.resolve()
    actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if actual_sha != manifest.get("code_sha"):
        raise SystemExit(f"code SHA mismatch: manifest {manifest.get('code_sha')}, checkout {actual_sha}")
    if status:
        raise SystemExit("GPU stage requires a clean worktree")
    model = manifest["model"]
    if model.get("repo_id") != "Qwen/Qwen3-1.7B-Base" or model.get("model_type") != "qwen3":
        raise SystemExit("model identity mismatch")
    if model.get("revision") != "ea980cb0a6c2ae4b936e82123acc929f1cec04c1":
        raise SystemExit("unapproved model revision")
    model_root = Path(model["local_path"])
    if model_files(model_root) != model.get("files"):
        raise SystemExit("model file set or hash differs from the complete manifest")
    provenance = model.get("revision_provenance", {})
    if not provenance:
        raise SystemExit("model revision provenance is missing")
    if revision_metadata(model_root) != provenance:
        raise SystemExit("model revision metadata set or hash differs from the complete manifest")
    for relative in provenance:
        path = model_root / relative
        if path.read_text(encoding="utf-8").splitlines()[0].strip() != model["revision"]:
            raise SystemExit(f"model revision metadata disagrees with manifest: {path}")
    required_data = {"train", "AIME2024", "AIME2025"}
    if set(manifest["data"]) != required_data:
        raise SystemExit("manifest must keep train, AIME2024, and AIME2025 separate")
    for name, record in manifest["data"].items():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise SystemExit(f"data hash mismatch: {name}: {path}")
    for name, record in manifest.get("frozen_artifacts", {}).items():
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise SystemExit(f"frozen artifact hash mismatch: {name}: {path}")
    print(json.dumps({"status": "PASS", "code_sha": actual_sha, "identity": actual_identity}))


if __name__ == "__main__":
    main()
