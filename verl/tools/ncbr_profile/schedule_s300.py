#!/usr/bin/env python3
"""Launch formal S300 arms Baseline-first on complete idle Ray nodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

NODES = ("A", "B")
RESOURCES = {"A": "ncbr_node_A", "B": "ncbr_node_B"}


def choose_assignments(free_nodes: list[str], launched: dict[str, str]) -> list[tuple[str, str]]:
    """Return new assignments while preserving Baseline-first semantics."""
    free = [node for node in NODES if node in free_nodes and node not in launched.values()]
    assignments = []
    if "baseline" not in launched and free:
        assignments.append(("baseline", free.pop(0)))
    if "baseline" in launched or any(arm == "baseline" for arm, _ in assignments):
        if "v1" not in launched and free:
            assignments.append(("v1", free.pop(0)))
    return assignments


def _gpu_probe(max_idle_memory_mib: int) -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    parsed = [[part.strip() for part in line.split(",")] for line in rows]
    process_rows = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    idle = (
        gpu.returncode == 0
        and processes.returncode == 0
        and len(parsed) == 8
        and not process_rows
        and all(int(row[2]) <= max_idle_memory_mib and int(row[3]) == 0 for row in parsed)
    )
    return {"idle": idle, "gpus": rows, "compute_processes": process_rows}


def _probe_nodes(ray, max_idle_memory_mib: int, timeout_seconds: float) -> tuple[list[str], dict[str, Any]]:
    probe = ray.remote(num_cpus=0.1, num_gpus=8)(_gpu_probe)
    refs = {
        node: probe.options(resources={RESOURCES[node]: 1e-3}).remote(max_idle_memory_mib) for node in NODES
    }
    free = []
    evidence = {}
    for node, ref in refs.items():
        ready, _ = ray.wait([ref], timeout=timeout_seconds)
        if not ready:
            ray.cancel(ref, force=True)
            evidence[node] = {"idle": False, "reason": "eight_ray_gpus_unavailable"}
            continue
        result = ray.get(ref)
        evidence[node] = result
        if result["idle"]:
            free.append(node)
    return free, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    gate_path = Path(spec["gate_receipt"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or not gate.get("s300_authorized", False):
        raise SystemExit("formal S300 gate receipt is not PASS/authorized")
    if not args.launch:
        raise SystemExit("dry resolution complete; pass --launch only for the automatic post-gate transition")

    import ray

    ray.init(address=spec.get("ray_address", "10.8.191.127:6395"), namespace="qwen17-ncbr-s300-scheduler")
    launched: dict[str, str] = {}
    launches = []
    snapshots = []
    deadline = time.monotonic() + float(spec.get("monitor_seconds", 1800))
    try:
        while len(launched) < 2:
            free, evidence = _probe_nodes(
                ray,
                int(spec.get("max_idle_memory_mib", 1024)),
                float(spec.get("ray_probe_timeout_seconds", 10)),
            )
            snapshots.append({"wall_time": time.time(), "free_nodes": free, "evidence": evidence})
            for arm, node in choose_assignments(free, launched):
                launch = spec["launches"][arm][node]
                environment = os.environ.copy()
                environment.update({str(key): str(value) for key, value in launch["env"].items()})
                environment["NCBR_NODE"] = node
                environment["NCBR_ARM"] = arm
                environment["NCBR_STAGE"] = "formal_s300"
                environment["NCBR_AUTHORIZE_S300"] = "AUTHORIZE_S300"
                log_path = Path(launch["log"])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open("x", encoding="utf-8")
                process = subprocess.Popen(
                    [str(launch["launcher"])],
                    cwd=str(launch["repo"]),
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                stream.close()
                time.sleep(2)
                immediate_exit = process.poll()
                if immediate_exit is not None:
                    raise RuntimeError(
                        f"formal S300 {arm} launcher exited immediately with code {immediate_exit}; see {log_path}"
                    )
                launched[arm] = node
                launches.append(
                    {"arm": arm, "node": node, "pid": process.pid, "log": str(log_path), "start_time": time.time()}
                )
            if len(launched) == 2 or time.monotonic() >= deadline:
                break
            time.sleep(float(spec.get("poll_seconds", 30)))
    finally:
        ray.shutdown()
    status = "PASS" if len(launched) == 2 else ("PARTIAL" if launched else "NO_LAUNCH")
    result = {
        "schema_version": "qwen3-1p7b-ncbr-s300-scheduler-v1",
        "status": status,
        "policy": "complete_idle_node_baseline_first_30_minute_monitor",
        "gate_receipt": str(gate_path.resolve()),
        "launched": launched,
        "launches": launches,
        "resource_snapshots": snapshots,
        "shared_ray_stopped": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "PASS":
        raise SystemExit(f"formal S300 scheduling ended with status {status}")


if __name__ == "__main__":
    main()
