#!/usr/bin/env python3
"""Fail closed unless the shared two-node Ray cluster is idle and correctly labelled."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path

import ray

EXPECTED = {
    "A": {"resource": "ncbr_node_A", "ip": "10.8.191.127", "hostname": "p-kt-dgx-a100-03"},
    "B": {"resource": "ncbr_node_B", "ip": "10.8.191.131", "hostname": "p-kt-dgx-a100-07"},
}


def _node_probe(asset_root: str, ray_root: str, model_dir_name: str) -> dict[str, object]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    root = Path(asset_root)
    model = root / "model" / model_dir_name
    data = root / "data"
    port_errors = []
    for path in (Path(ray_root) / "session_latest" / "logs").glob("*"):
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "No available ports" in line or "Failed to register worker" in line:
                    port_errors.append(f"{path.name}:{line}")
        except OSError:
            continue
    return {
        "hostname": socket.gethostname(),
        "ips": sorted(set(subprocess.check_output(["hostname", "-I"], text=True).split())),
        "gpu_exit_code": gpu.returncode,
        "gpus": [line.strip() for line in gpu.stdout.splitlines() if line.strip()],
        "compute_exit_code": compute.returncode,
        "compute_processes": [line.strip() for line in compute.stdout.splitlines() if line.strip()],
        "asset_paths": {
            "model": str(model),
            "train": str(data / "dapo_math_17k_train.parquet"),
            "AIME2024": str(data / "aime-2024-verl.parquet"),
            "AIME2025": str(data / "aime-2025-verl.parquet"),
        },
        "asset_exists": model.is_dir()
        and all(
            (data / name).is_file()
            for name in ("dapo_math_17k_train.parquet", "aime-2024-verl.parquet", "aime-2025-verl.parquet")
        ),
        "ray_worker_port_errors": port_errors[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="10.8.191.127:6395")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", default="/tmp/qwen17-ncbr-assets-5904152e")
    parser.add_argument("--node-a-ray-root", default="/tmp/q17r590h2")
    parser.add_argument("--node-b-ray-root", default="/tmp/q17r590w2")
    parser.add_argument("--max-idle-memory-mib", type=int, default=1024)
    parser.add_argument("--model-dir-name", default="Qwen3-1.7B-Base")
    args = parser.parse_args()
    ray.init(address=args.address, namespace="qwen17-ncbr-shared-preflight", log_to_driver=False)
    try:
        resources = ray.cluster_resources()
        available = ray.available_resources()
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        remote_probe = ray.remote(num_cpus=1)(_node_probe)
        refs = {
            node: remote_probe.options(resources={expected["resource"]: 1e-3}).remote(
                args.asset_root,
                args.node_a_ray_root if node == "A" else args.node_b_ray_root,
                args.model_dir_name,
            )
            for node, expected in EXPECTED.items()
        }
        probes = {node: ray.get(ref, timeout=60) for node, ref in refs.items()}
    finally:
        ray.shutdown()

    checks = {
        "two_alive_nodes": len(alive_nodes) == 2,
        "sixteen_total_gpus": float(resources.get("GPU", 0)) == 16.0,
        "sixteen_available_gpus": float(available.get("GPU", 0)) == 16.0,
        "common_cpu_capacity": float(resources.get("CPU", 0)) == 480.0,
        "common_object_store": int(resources.get("object_store_memory", 0)) == 256 * 1024**3,
    }
    for node, expected in EXPECTED.items():
        probe = probes[node]
        gpu_memory = [int(row.split(",")[2].strip()) for row in probe["gpus"]]
        checks.update(
            {
                f"node_{node}_identity": probe["hostname"] == expected["hostname"]
                and expected["ip"] in probe["ips"],
                f"node_{node}_eight_gpus": probe["gpu_exit_code"] == 0 and len(probe["gpus"]) == 8,
                f"node_{node}_no_compute_process": probe["compute_exit_code"] == 0 and not probe["compute_processes"],
                f"node_{node}_idle_memory": len(gpu_memory) == 8
                and max(gpu_memory, default=10**9) <= args.max_idle_memory_mib,
                f"node_{node}_assets": bool(probe["asset_exists"]),
                f"node_{node}_worker_ports": not probe["ray_worker_port_errors"],
                f"node_{node}_resource_label": float(resources.get(expected["resource"], 0)) == 1.0,
            }
        )
    result = {
        "schema_version": "qwen3-1p7b-shared-ray-preflight-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cluster_resources": resources,
        "available_resources": available,
        "alive_node_count": len(alive_nodes),
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit("shared Ray preflight failed")


if __name__ == "__main__":
    main()
