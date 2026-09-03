#!/usr/bin/env python3
"""Strictly attest that the one-shot node stage left no target process or port."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
from pathlib import Path


def target_processes(ps_output: str, *, current_pid: int) -> list[str]:
    target = re.compile(r"raylet|gcs_server|vllm|EngineCore|main_dapo_boundary_return", re.IGNORECASE)
    matches = []
    for line in ps_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=2)
        if not fields or not fields[0].isdigit() or int(fields[0]) == current_pid:
            continue
        if target.search(stripped):
            matches.append(stripped)
    return matches


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ports = {
        "A": [6397, 7111, 7112, 8267, 9087],
        "B": [6398, 7211, 7212, 8268, 9088],
    }[args.node]
    process_query = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        text=True,
        capture_output=True,
    )
    if process_query.returncode != 0:
        raise SystemExit(f"process inventory failed: {process_query.stderr.strip()}")
    processes = target_processes(process_query.stdout, current_pid=os.getpid())
    listening = {str(port): port_open(port) for port in ports}
    result = {
        "schema_version": "qwen3-1p7b-stage-teardown-v1",
        "node": args.node,
        "target_processes": processes,
        "listening": listening,
        "status": "PASS" if not processes and not any(listening.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(f"strict teardown gate failed: processes={processes}, ports={listening}")


if __name__ == "__main__":
    main()
