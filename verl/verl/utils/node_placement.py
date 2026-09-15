# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional hard node pinning; unset environment preserves normal placement."""

import json
import os
from pathlib import Path

import ray


def pinned_node_ip():
    return os.environ.get("VERL_TRAINING_NODE_IP")


def training_node_ids():
    target = pinned_node_ip()
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0]
    if target:
        nodes = [n for n in nodes if n["NodeManagerAddress"] == target]
        if len(nodes) != 1:
            raise RuntimeError(f"Pinned training node unavailable: {target}")
    return [n["NodeID"] for n in nodes]


def pin_actor(actor_class):
    if not pinned_node_ip():
        return actor_class
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return actor_class.options(scheduling_strategy=NodeAffinitySchedulingStrategy(training_node_ids()[0], soft=False))


def audit_placement(role):
    if not pinned_node_ip():
        return
    actual = ray.get_runtime_context().get_node_id()
    assert actual == training_node_ids()[0], f"{role} escaped pinned node"
    directory = os.environ.get("GRPO_PLACEMENT_AUDIT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        record = {
            "role": role,
            "pid": os.getpid(),
            "node_id": actual,
            "node_ip": pinned_node_ip(),
            "gpu_ids": ray.get_gpu_ids(),
        }
        target = path / f"{role}_{os.getpid()}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True))
        temporary.replace(target)
