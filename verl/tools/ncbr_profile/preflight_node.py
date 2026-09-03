#!/usr/bin/env python3
"""Fail-closed local-node sizing and GPU preflight for standalone Ray."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import subprocess
from pathlib import Path

GIB = 1024**3


def mem_total_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal is unavailable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=("A", "B"), required=True)
    parser.add_argument("--require-wandb", action="store_true")
    args = parser.parse_args()
    shm = os.statvfs("/dev/shm")
    shm_available = shm.f_bavail * shm.f_frsize
    total_memory = mem_total_bytes()
    object_store_gib = int(min(0.20 * total_memory, 0.50 * shm_available, 128 * GIB) // GIB)
    online_cpus = os.cpu_count() or 0
    ray_cpus = min(240, online_cpus - 16)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    compute_rows = [line.strip() for line in compute.stdout.splitlines() if line.strip()]
    host_ips = sorted(set(subprocess.check_output(["hostname", "-I"], text=True).split()))
    expected_ip = {"A": "10.8.191.127", "B": "10.8.191.131"}[args.node]
    wandb_status: dict[str, object] = {"required": bool(args.require_wandb), "verified": False}
    if args.require_wandb:
        try:
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                import wandb

                viewer = wandb.Api(timeout=20).viewer
            wandb_status.update({"verified": True, "viewer": getattr(viewer, "username", None)})
        except Exception as error:
            wandb_status["error"] = f"{type(error).__name__}: {error}"
    result = {
        "schema_version": "qwen3-1p7b-node-preflight-v1",
        "node": args.node,
        "hostname": socket.gethostname(),
        "host_ips": host_ips,
        "expected_ip": expected_ip,
        "mem_total_bytes": total_memory,
        "shm_available_bytes": shm_available,
        "object_store_gib": object_store_gib,
        "object_store_bytes": object_store_gib * GIB,
        "online_cpus": online_cpus,
        "ray_cpus": ray_cpus,
        "gpu_query_exit_code": query.returncode,
        "gpus": gpu_rows,
        "compute_process_query_exit_code": compute.returncode,
        "compute_processes": compute_rows,
        "wandb": wandb_status,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if object_store_gib < 32:
        raise SystemExit("object store gate failed: computed capacity is below 32 GiB")
    if ray_cpus <= 0:
        raise SystemExit("CPU capacity gate failed")
    if query.returncode != 0 or len(gpu_rows) != 8:
        raise SystemExit("GPU gate failed: exactly 8 visible GPUs are required")
    if compute.returncode != 0:
        raise SystemExit("GPU compute-process query failed")
    if compute_rows:
        raise SystemExit("GPU gate failed: compute processes are still present")
    if expected_ip not in host_ips:
        raise SystemExit(f"node identity gate failed: {expected_ip} is not a local address")
    if args.require_wandb and not wandb_status["verified"]:
        raise SystemExit("W&B connectivity/identity gate failed")


if __name__ == "__main__":
    main()
