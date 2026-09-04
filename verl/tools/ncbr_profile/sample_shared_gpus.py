#!/usr/bin/env python3
"""Sample both labelled Ray nodes without reserving or stopping any GPU."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import ray


EXPECTED = {
    "A": {"resource": "ncbr_node_A", "hostname": "p-kt-dgx-a100-03"},
    "B": {"resource": "ncbr_node_B", "hostname": "p-kt-dgx-a100-07"},
}


def _parse_csv(stdout: str) -> list[dict[str, object]]:
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
                "utilization_percent": float(fields[4]),
            }
        )
    return rows


def _sample_node() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    return {
        "timestamp": time.time(),
        "hostname": socket.gethostname(),
        "exit_code": result.returncode,
        "stderr": result.stderr,
        "gpus": _parse_csv(result.stdout) if result.returncode == 0 else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="10.8.191.127:6395")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-duration-seconds", type=float, default=1800.0)
    parser.add_argument("--minimum-samples", type=int, default=2)
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.max_duration_seconds <= 0 or args.minimum_samples < 1:
        raise SystemExit("sampling interval, duration, and minimum samples must be positive")
    if args.stop_file.exists():
        raise SystemExit(f"refusing stale stop file: {args.stop_file}")

    ray.init(address=args.address, namespace="q17ncbr-gpu-sampling", log_to_driver=False)
    remote_sample = ray.remote(num_cpus=0)(_sample_node)
    samples: dict[str, list[dict[str, object]]] = {"A": [], "B": []}
    started = time.monotonic()
    timed_out = False
    try:
        while True:
            refs = {
                node: remote_sample.options(resources={expected["resource"]: 1e-3}).remote()
                for node, expected in EXPECTED.items()
            }
            for node, ref in refs.items():
                samples[node].append(ray.get(ref, timeout=30))
            if args.stop_file.exists():
                break
            if time.monotonic() - started >= args.max_duration_seconds:
                timed_out = True
                break
            time.sleep(args.interval_seconds)
    finally:
        ray.shutdown()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    statuses = {}
    for node, records in samples.items():
        expected = EXPECTED[node]
        valid = [
            record
            for record in records
            if record["exit_code"] == 0
            and record["hostname"] == expected["hostname"]
            and len(record["gpus"]) == 8
        ]
        payload = {
            "schema_version": "qwen3-1p7b-shared-gpu-samples-v1",
            "node": node,
            "expected_hostname": expected["hostname"],
            "timed_out": timed_out,
            "sample_count": len(records),
            "valid_sample_count": len(valid),
            "gpu_utilization_percent": [
                gpu["utilization_percent"] for record in valid for gpu in record["gpus"]
            ],
            "samples": records,
        }
        payload["status"] = (
            "PASS" if not timed_out and len(valid) >= args.minimum_samples and len(valid) == len(records) else "FAIL"
        )
        statuses[node] = payload["status"]
        (args.output_dir / f"gpu_samples_{node}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary = {
        "schema_version": "qwen3-1p7b-shared-gpu-sampling-summary-v1",
        "shared_ray_stopped": False,
        "node_status": statuses,
        "status": "PASS" if all(status == "PASS" for status in statuses.values()) else "FAIL",
    }
    (args.output_dir / "gpu_sampling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
