#!/usr/bin/env python3
"""Verify that one standalone Ray head exposes only the intended local 8-GPU node."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path

import ray


def _gpu_uuids() -> list[str]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"], text=True)
    return sorted(line.strip() for line in output.splitlines() if line.strip())


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=("A", "B"), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ports = {
        "A": {"gcs": 6397, "object_manager": 7111, "node_manager": 7112, "dashboard": 8267, "metrics": 9087},
        "B": {"gcs": 6398, "object_manager": 7211, "node_manager": 7212, "dashboard": 8268, "metrics": 9088},
    }[args.node]
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    address = f"127.0.0.1:{ports['gcs']}"
    ray.init(address=address, namespace=f"qwen17-ncbr-node-{args.node}", log_to_driver=False)
    try:
        alive_nodes = [record for record in ray.nodes() if record.get("Alive")]
        resources = ray.cluster_resources()

        @ray.remote(num_cpus=1)
        def identity_probe() -> dict[str, object]:
            return {
                "hostname": socket.gethostname(),
                "gpu_uuids": _gpu_uuids(),
            }

        probe = ray.get(identity_probe.remote(), timeout=30)
    finally:
        ray.shutdown()
    expected_uuids = sorted(row.split(",")[1].strip() for row in preflight["gpus"])
    listening = {name: _port_open(port) for name, port in ports.items()}
    result = {
        "schema_version": "qwen3-1p7b-local-ray-probe-v1",
        "node": args.node,
        "address": address,
        "alive_node_count": len(alive_nodes),
        "cluster_resources": resources,
        "identity_probe": probe,
        "expected_hostname": preflight["hostname"],
        "expected_gpu_uuids": expected_uuids,
        "listening": listening,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(alive_nodes) != 1:
        raise SystemExit("Ray topology gate failed: expected exactly one live node")
    if float(resources.get("GPU", 0.0)) != 8.0:
        raise SystemExit("Ray resource gate failed: expected exactly eight GPUs")
    if probe["hostname"] != preflight["hostname"] or probe["gpu_uuids"] != expected_uuids:
        raise SystemExit("Ray identity probe crossed the intended hostname/GPU inventory")
    if not all(listening.values()):
        raise SystemExit(f"Ray port gate failed: {listening}")


if __name__ == "__main__":
    main()
