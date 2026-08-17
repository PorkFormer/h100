#!/usr/bin/env python3
"""Execution-time gate for the isolated eight-A100 FA-RAR smoke."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_HOST = "p-kt-dgx-a100-05"
EXPECTED_IP = "10.8.191.129"
CAC_IPS = {"10.8.191.127", "10.8.191.131"}
BUSY_PATTERN = re.compile(r"(ray::|raylet|gcs_server|vllm|verl|torchrun|wandb.*service|python.*train)", re.I)


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    hostname = socket.gethostname()
    addresses = set(_run("hostname", "-I").split())
    if hostname != EXPECTED_HOST:
        failures.append(f"hostname={hostname!r}, expected {EXPECTED_HOST!r}")
    if EXPECTED_IP not in addresses:
        failures.append(f"expected private host IP {EXPECTED_IP} is absent")
    if addresses & CAC_IPS:
        failures.append(f"local host unexpectedly exposes CAC IPs {sorted(addresses & CAC_IPS)}")
    if os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_REDIS_ADDRESS"):
        failures.append("RAY_ADDRESS and RAY_REDIS_ADDRESS must be unset")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            probe.bind((EXPECTED_IP, args.port))
        except OSError as exc:
            failures.append(f"smoke-only GCS port {EXPECTED_IP}:{args.port} is unavailable: {exc}")

    own_ancestry = {os.getpid(), os.getppid()}
    process_rows = _run("ps", "-eo", "pid=,ppid=,args=").splitlines()
    busy_processes = []
    for row in process_rows:
        fields = row.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid, ppid, command = int(fields[0]), int(fields[1]), fields[2]
        if pid in own_ancestry or ppid in own_ancestry:
            continue
        if BUSY_PATTERN.search(command):
            busy_processes.append({"pid": pid, "ppid": ppid, "command": command})
    if busy_processes:
        failures.append(f"trainer/Ray/vLLM processes are active: {busy_processes}")

    query = _run(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    gpu_rows = []
    for line in query.splitlines():
        index, name, uuid, used, total, utilization, temperature = [item.strip() for item in line.split(",")]
        row = {
            "index": int(index),
            "name": name,
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": int(utilization),
            "temperature_c": int(temperature),
        }
        gpu_rows.append(row)
    if len(gpu_rows) != 8:
        failures.append(f"expected 8 GPUs, found {len(gpu_rows)}")
    for row in gpu_rows:
        if "A100" not in row["name"]:
            failures.append(f"GPU {row['index']} is not an A100: {row['name']}")
        if row["memory_used_mib"] != 0 or row["utilization_percent"] != 0:
            failures.append(f"GPU {row['index']} is not idle: {row}")

    compute_apps = _run(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    if compute_apps:
        failures.append(f"GPU compute applications are active: {compute_apps}")

    cuda_error = None
    device_count = 0
    device_results = []
    count_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count())",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if count_check.returncode == 0:
        available, raw_count = count_check.stdout.strip().split()
        device_count = int(raw_count)
        if available != "1" or device_count != 8:
            failures.append(
                f"torch CUDA discovery returned available={available}, device_count={device_count}; expected 1, 8"
            )
    else:
        failures.append(f"torch CUDA discovery failed: {count_check.stderr.strip()}")
    for index in range(8):
        device_env = os.environ.copy()
        device_env["CUDA_VISIBLE_DEVICES"] = str(index)
        check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; x=torch.ones(1, device='cuda:0'); torch.cuda.synchronize(); print(x.item())",
            ],
            env=device_env,
            text=True,
            capture_output=True,
            check=False,
        )
        device_result = {
            "physical_index": index,
            "status": "PASS" if check.returncode == 0 and check.stdout.strip() == "1.0" else "FAIL",
            "stdout": check.stdout.strip(),
            "stderr": check.stderr.strip(),
        }
        device_results.append(device_result)
        if device_result["status"] == "FAIL":
            failures.append(f"CUDA initialization failed on physical GPU {index}: {device_result['stderr']}")
    failed_devices = [result for result in device_results if result["status"] == "FAIL"]
    if failed_devices:
        cuda_error = f"per-device failures: {[result['physical_index'] for result in failed_devices]}"

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "hostname": hostname,
        "addresses": sorted(addresses),
        "expected_private_ip": EXPECTED_IP,
        "excluded_cac_ips": sorted(CAC_IPS),
        "gcs_port": args.port,
        "busy_processes": busy_processes,
        "gpus": gpu_rows,
        "compute_apps": compute_apps,
        "torch_cuda_device_count": device_count,
        "cuda_device_results": device_results,
        "cuda_error": cuda_error,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("private GPU/Ray isolation gate failed")


if __name__ == "__main__":
    main()
