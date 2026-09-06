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


def validate_stage_artifacts(stage: str, candidate: str, diagnostics: str, artifacts: dict) -> None:
    if stage in {"fixed_replay", "acceptance"}:
        selection = json.loads(Path(artifacts["profile_selection"]["path"]).read_text(encoding="utf-8"))
        if selection.get("status") != "PASS" or selection.get("selected_candidate") != candidate:
            raise SystemExit("stage candidate does not match the PASS profile selection")
    if stage == "acceptance":
        overhead = json.loads(Path(artifacts["diagnostics_overhead_gate"]["path"]).read_text(encoding="utf-8"))
        if overhead.get("status") != "PASS":
            raise SystemExit("acceptance requires a PASS diagnostics overhead gate")
    if stage in {"fixed_replay", "acceptance"} and diagnostics != "on":
        raise SystemExit(f"{stage} requires diagnostics=on")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
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
    remote = manifest.get("code_remote", {})
    if remote.get("sha") != actual_sha or not remote.get("name") or not remote.get("ref"):
        raise SystemExit("stage manifest is missing pinned remote code provenance")
    remote_output = subprocess.check_output(
        ["git", "ls-remote", "--exit-code", remote["name"], remote["ref"]],
        cwd=repo,
        text=True,
    ).splitlines()
    remote_shas = {line.split()[0] for line in remote_output if line.split()}
    if remote_shas != {actual_sha}:
        raise SystemExit(f"remote code ref moved or is not pinned to checkout SHA: {sorted(remote_shas)}")
    model = manifest["model"]
    approved_models = {
        "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
    }
    if model.get("repo_id") not in approved_models or model.get("model_type") != "qwen3":
        raise SystemExit("model identity mismatch")
    if model.get("revision") != approved_models[model["repo_id"]]:
        raise SystemExit("unapproved model revision")
    is_8b = model["repo_id"] == "Qwen/Qwen3-8B-Base"
    if is_8b and args.candidate not in {"P0", "P1", "P_SAFE"}:
        raise SystemExit("unapproved Qwen3-8B profiling candidate")
    if not is_8b and args.candidate == "P_SAFE":
        raise SystemExit("unapproved Qwen3-1.7B profiling candidate")
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
    artifacts = manifest.get("frozen_artifacts", {})
    required_by_stage = {
        "gate0": {
            "hard_prefix_panel",
            "hard_prefix_panel_manifest",
            "calibration_workload_comparison",
            "calibration_performance_comparison",
        },
        "profile": {
            "hard_prefix_panel",
            "hard_prefix_panel_manifest",
            "calibration_workload_comparison",
            "calibration_performance_comparison",
        },
        "mechanism_panel": {"hard_prefix_panel", "hard_prefix_panel_manifest"},
        "fixed_replay": {"hard_prefix_panel", "hard_prefix_panel_manifest", "profile_selection"},
        "acceptance": {"profile_selection", "diagnostics_overhead_gate"},
        "formal_s300": {"s300_gate_receipt"},
    }
    if is_8b:
        required_by_stage = {
            "profile": {"gate0_receipt"},
            "smoke": {"profile_selection"},
            "formal_s300": {
                "profile_selection",
                "smoke_gate_receipt",
                "resolved_config_diff_receipt",
                "readiness_receipt",
            },
        }
    missing_artifacts = required_by_stage.get(args.stage, set()) - artifacts.keys()
    if missing_artifacts:
        raise SystemExit(f"stage is missing required frozen artifacts: {sorted(missing_artifacts)}")
    validate_stage_artifacts(args.stage, args.candidate, args.diagnostics, artifacts)
    if args.stage == "formal_s300" and not is_8b and args.diagnostics != "on":
        raise SystemExit("formal_s300 requires diagnostics=on for Qwen3-1.7B")
    if args.stage == "formal_s300" and is_8b and args.diagnostics != "off":
        raise SystemExit("formal_s300 requires diagnostics=off for the frozen Qwen3-8B recipe")
    if args.stage == "formal_s300" and not is_8b:
        gate = json.loads(Path(artifacts["s300_gate_receipt"]["path"]).read_text(encoding="utf-8"))
        if gate.get("status") != "PASS" or not gate.get("s300_authorized", False):
            raise SystemExit("formal S300 gate receipt is not PASS/authorized")
        if gate.get("code_sha") != actual_sha:
            raise SystemExit("formal S300 gate receipt code SHA mismatch")
        if gate.get("selected_candidate") != args.candidate:
            raise SystemExit("formal S300 candidate does not match the gate selection")
    if args.stage == "formal_s300" and is_8b:
        for artifact_name in (
            "profile_selection",
            "smoke_gate_receipt",
            "resolved_config_diff_receipt",
            "readiness_receipt",
        ):
            receipt = json.loads(Path(artifacts[artifact_name]["path"]).read_text(encoding="utf-8"))
            if receipt.get("status") != "PASS":
                raise SystemExit(f"formal S300 requires PASS {artifact_name}")
    print(json.dumps({"status": "PASS", "code_sha": actual_sha, "identity": actual_identity}))


if __name__ == "__main__":
    main()
