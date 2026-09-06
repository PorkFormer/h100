#!/usr/bin/env python3
"""Create a fail-closed stage manifest from explicit immutable inputs."""

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


def files_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and ".cache" not in path.relative_to(root).parts
    }


def revision_provenance(root: Path, expected_revision: str) -> dict[str, str]:
    metadata = sorted((root / ".cache" / "huggingface" / "download").glob("*.metadata"))
    if not metadata:
        raise SystemExit("model revision provenance is unavailable: no Hugging Face download metadata")
    records = {}
    for path in metadata:
        revision = path.read_text(encoding="utf-8").splitlines()[0].strip()
        if revision != expected_revision:
            raise SystemExit(f"model revision mismatch in {path}: expected {expected_revision}, got {revision}")
        records[str(path.relative_to(root))] = sha256(path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--remote-ref", required=True)
    parser.add_argument("--arm", choices=("baseline", "v1"), required=True)
    parser.add_argument("--candidate", choices=("P0", "P1", "P2", "P_SAFE"), required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "calibration",
            "gate0",
            "profile",
            "mechanism_panel",
            "fixed_replay",
            "acceptance",
            "smoke",
            "formal_s300",
        ),
        required=True,
    )
    parser.add_argument("--node", choices=("A", "B"), required=True)
    parser.add_argument("--diagnostics", choices=("on", "off"), required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/models/Qwen3-1.7B-Base"))
    parser.add_argument("--model-repo-id", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--train-data", type=Path, default=Path("/workspace/rl/data/dapo_math_17k_train.parquet"))
    parser.add_argument("--aime2024", type=Path, default=Path("/workspace/rl/data/aime-2024-verl.parquet"))
    parser.add_argument("--aime2025", type=Path, default=Path("/workspace/rl/data/aime-2025-verl.parquet"))
    parser.add_argument("--frozen-artifact", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    repo = args.repo.resolve()
    actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if actual_sha != args.code_sha:
        raise SystemExit(f"code SHA mismatch: expected {args.code_sha}, got {actual_sha}")
    if status:
        raise SystemExit("refusing to manifest a dirty worktree")
    remote_output = subprocess.check_output(
        ["git", "ls-remote", "--exit-code", args.remote, args.remote_ref],
        cwd=repo,
        text=True,
    ).splitlines()
    remote_shas = {line.split()[0] for line in remote_output if line.split()}
    if remote_shas != {actual_sha}:
        raise SystemExit(
            f"remote ref is not pinned to the local code SHA: {args.remote}/{args.remote_ref}: {sorted(remote_shas)}"
        )
    model_config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen3":
        raise SystemExit("model config model_type is not qwen3")
    provenance = revision_provenance(args.model_path.resolve(), args.model_revision)
    data_paths = {
        "train": args.train_data,
        "AIME2024": args.aime2024,
        "AIME2025": args.aime2025,
    }
    frozen_artifacts = {}
    for item in args.frozen_artifact:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or name in frozen_artifacts:
            raise SystemExit(f"invalid or duplicate frozen artifact: {item!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SystemExit(f"frozen artifact is not a file: {path}")
        frozen_artifacts[name] = {"path": str(path), "sha256": sha256(path)}
    approved_models = {
        "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    }
    if approved_models.get(args.model_repo_id) != args.model_revision:
        raise SystemExit("model repository/revision pair is not approved")
    if args.model_repo_id.endswith("8B-Base") and args.candidate == "P2":
        raise SystemExit("Qwen3-8B uses P0, P1, or P_SAFE, not P2")
    if args.model_repo_id.endswith("1.7B-Base") and args.candidate == "P_SAFE":
        raise SystemExit("Qwen3-1.7B uses P0, P1, or P2, not P_SAFE")
    manifest = {
        "schema_version": "qwen3-ncbr-stage-manifest-v2",
        "code_sha": actual_sha,
        "code_remote": {"name": args.remote, "ref": args.remote_ref, "sha": actual_sha},
        "arm": args.arm,
        "candidate": args.candidate,
        "stage": args.stage,
        "node": args.node,
        "diagnostics": args.diagnostics,
        "model": {
            "repo_id": args.model_repo_id,
            "revision": args.model_revision,
            "local_path": str(args.model_path.resolve()),
            "model_type": "qwen3",
            "files": files_manifest(args.model_path.resolve()),
            "revision_provenance": provenance,
        },
        "data": {
            name: {"path": str(path.resolve()), "sha256": sha256(path.resolve())} for name, path in data_paths.items()
        },
        "frozen_artifacts": frozen_artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
