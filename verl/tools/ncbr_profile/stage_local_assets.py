#!/usr/bin/env python3
"""Idempotently stage and attest Qwen3-1.7B model/data on one local node."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def write_hashes(path: Path, records: dict[str, str]) -> None:
    path.write_text("".join(f"{digest}  ./{name}\n" for name, digest in records.items()), encoding="utf-8")


def copy_file_once(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or sha256(destination) != sha256(source):
            raise SystemExit(f"refusing to overwrite mismatched asset: {destination}")
        return
    partial = destination.with_name(destination.name + f".partial.{os.getpid()}")
    shutil.copy2(source, partial)
    os.replace(partial, destination)


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def stage(args: argparse.Namespace) -> None:
    destination = args.destination.resolve()
    model_source = args.model_source.resolve()
    data_source = args.data_source.resolve()
    model_destination = destination / "model" / "Qwen3-1.7B-Base"
    data_destination = destination / "data"
    destination.mkdir(parents=True, exist_ok=True)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    data_destination.mkdir(parents=True, exist_ok=True)

    source_model_hashes = file_hashes(model_source)
    if model_destination.exists():
        if file_hashes(model_destination) != source_model_hashes:
            raise SystemExit(f"refusing to replace mismatched model tree: {model_destination}")
    else:
        partial = model_destination.with_name(model_destination.name + f".partial.{os.getpid()}")
        shutil.copytree(model_source, partial)
        os.replace(partial, model_destination)

    data_names = ("dapo_math_17k_train.parquet", "aime-2024-verl.parquet", "aime-2025-verl.parquet")
    source_data_hashes = {name: sha256(data_source / name) for name in data_names}
    for name in data_names:
        copy_file_once(data_source / name, data_destination / name)
    local_data_hashes = {name: sha256(data_destination / name) for name in data_names}
    local_model_hashes = file_hashes(model_destination)
    if local_model_hashes != source_model_hashes or local_data_hashes != source_data_hashes:
        raise SystemExit("post-copy SHA256 verification failed")

    make_read_only(model_destination)
    make_read_only(data_destination)
    write_hashes(destination / f"node_{args.node}_model_source.sha256", source_model_hashes)
    write_hashes(destination / f"node_{args.node}_model_local.sha256", local_model_hashes)
    write_hashes(destination / f"node_{args.node}_data_source.sha256", source_data_hashes)
    write_hashes(destination / f"node_{args.node}_data_local.sha256", local_data_hashes)
    writable = [
        str(path.relative_to(destination))
        for root in (model_destination, data_destination)
        for path in (root, *root.rglob("*"))
        if path.stat().st_mode & 0o222
    ]
    (destination / f"node_{args.node}_writable_files.txt").write_text("\n".join(writable), encoding="utf-8")
    if writable:
        raise SystemExit(f"read-only asset gate failed: {writable}")
    print(f"PASS node={args.node} model_files={len(local_model_hashes)} data_files={len(local_data_hashes)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=("A", "B"), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, default=Path("/workspace/models/Qwen3-1.7B-Base"))
    parser.add_argument("--data-source", type=Path, default=Path("/workspace/rl/data"))
    stage(parser.parse_args())


if __name__ == "__main__":
    main()
