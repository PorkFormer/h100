# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auditable interval-DAG and workload-normalized profiling primitives."""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4 as _profile_uuid4


@dataclass(frozen=True)
class ProfileInterval:
    interval_id: str
    name: str
    wall_start: float
    wall_end: float
    parent_id: str | None
    asynchronous: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.wall_end - self.wall_start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntervalRecorder:
    def __init__(self, namespace: str):
        self.namespace = str(namespace)
        self.intervals: list[ProfileInterval] = []
        self._stack: list[str] = []

    @contextlib.contextmanager
    def record(
        self,
        name: str,
        *,
        parent_id: str | None = None,
        asynchronous: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        resolved_parent = parent_id if parent_id is not None else (self._stack[-1] if self._stack else None)
        interval_id = f"{self.namespace}:{name}:{_profile_uuid4().hex}"
        start = time.perf_counter()
        self._stack.append(interval_id)
        try:
            yield interval_id
        finally:
            popped = self._stack.pop()
            if popped != interval_id:
                raise RuntimeError("profiling interval stack corruption")
            self.intervals.append(
                ProfileInterval(
                    interval_id=interval_id,
                    name=str(name),
                    wall_start=start,
                    wall_end=time.perf_counter(),
                    parent_id=resolved_parent,
                    asynchronous=bool(asynchronous),
                    metadata=dict(metadata or {}),
                )
            )

    def add(self, interval: ProfileInterval) -> None:
        self.intervals.append(interval)


def interval_union_seconds(intervals: Sequence[ProfileInterval]) -> float:
    ranges = sorted((item.wall_start, item.wall_end) for item in intervals if item.wall_end > item.wall_start)
    if not ranges:
        return 0.0
    total = 0.0
    start, end = ranges[0]
    for next_start, next_end in ranges[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def analyze_interval_dag(
    intervals: Sequence[ProfileInterval],
    *,
    full_step_interval_id: str | None = None,
    modeled_interval_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate nesting and derive exclusive/critical-path time without naive sums."""
    by_id = {item.interval_id: item for item in intervals}
    errors: list[str] = []
    if len(by_id) != len(intervals):
        errors.append("duplicate_interval_id")
    children: dict[str, list[ProfileInterval]] = {key: [] for key in by_id}
    for item in intervals:
        if item.wall_end < item.wall_start:
            errors.append(f"negative_duration:{item.interval_id}")
        if item.parent_id is not None:
            parent = by_id.get(item.parent_id)
            if parent is None:
                errors.append(f"missing_parent:{item.interval_id}")
            else:
                children[parent.interval_id].append(item)
                if item.wall_start < parent.wall_start or item.wall_end > parent.wall_end:
                    errors.append(f"child_outside_parent:{item.interval_id}")

    exclusive: dict[str, float] = {}
    sibling_overlap: list[str] = []
    for interval_id, item_children in children.items():
        parent = by_id[interval_id]
        child_union = interval_union_seconds(item_children)
        exclusive[interval_id] = max(parent.duration - child_union, 0.0)
        if sum(child.duration for child in item_children) > child_union + 1.0e-9:
            sibling_overlap.append(interval_id)
    top_level = [item for item in intervals if item.parent_id is None]
    critical_path_seconds = interval_union_seconds(top_level)

    fixed: float | None = None
    fixed_reason: str | None = None
    full = by_id.get(full_step_interval_id) if full_step_interval_id is not None else None
    modeled = [by_id[item_id] for item_id in modeled_interval_ids if item_id in by_id]
    if errors:
        fixed_reason = "invalid_interval_dag"
    elif full is None:
        fixed_reason = "full_step_coverage_unavailable"
    elif len(modeled) != len(modeled_interval_ids):
        fixed_reason = "modeled_interval_missing"
    elif any(item.wall_start < full.wall_start or item.wall_end > full.wall_end for item in modeled):
        fixed_reason = "modeled_interval_outside_step"
    else:
        fixed = max(full.duration - interval_union_seconds(modeled), 0.0)

    return {
        "valid": not errors,
        "errors": errors,
        "exclusive_seconds": exclusive,
        "overlapping_parent_ids": sibling_overlap,
        "top_level_interval_union_seconds": critical_path_seconds,
        "critical_path_seconds": critical_path_seconds,
        "u_fixed": fixed if fixed is not None else "unavailable",
        "u_fixed_unavailable_reason": fixed_reason,
    }


def normalized_unit_cost(exclusive_seconds: float, workload: float) -> float | str:
    if workload <= 0:
        return "unavailable"
    return float(exclusive_seconds) / float(workload)


def coefficient_of_variation(values: Sequence[float]) -> float | str:
    if not values:
        return "unavailable"
    mean = sum(values) / len(values)
    if mean == 0:
        return "unavailable"
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5 / mean


def coordinate_profile_extension(
    coordination_dir: Path,
    *,
    arm: str,
    candidate: str,
    diagnostics_mode: str,
    step_walls_2_4: Sequence[float],
    threshold: float = 0.10,
    timeout_seconds: float = 900,
    poll_seconds: float = 1.0,
) -> bool:
    """Return whether both live arm processes must extend from Step 4 to Step 6."""
    if arm not in {"baseline", "v1"} or candidate not in {"P0", "P1", "P2"}:
        raise ValueError("paired profiling identity is invalid")
    if diagnostics_mode not in {"on", "off"}:
        raise ValueError("paired profiling diagnostics identity is invalid")
    cv = coefficient_of_variation(step_walls_2_4)
    if not isinstance(cv, float):
        raise ValueError("paired profiling Step 2-4 CV is unavailable")
    coordination_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = coordination_dir / f"{candidate}_{diagnostics_mode}_{arm}.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing stale paired-profile CV receipt: {receipt_path}")
    payload = {
        "schema_version": "ncbr-paired-profile-cv-v1",
        "arm": arm,
        "candidate": candidate,
        "diagnostics": diagnostics_mode,
        "step_walls_2_4": list(step_walls_2_4),
        "cv_2_4": cv,
        "threshold": float(threshold),
    }
    temporary = coordination_dir / f".{receipt_path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt_path)
    peer = "v1" if arm == "baseline" else "baseline"
    peer_path = coordination_dir / f"{candidate}_{diagnostics_mode}_{peer}.json"
    deadline = time.monotonic() + float(timeout_seconds)
    while not peer_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for paired-profile receipt: {peer_path}")
        time.sleep(poll_seconds)
    peer_payload = json.loads(peer_path.read_text(encoding="utf-8"))
    expected_peer = (peer, candidate, diagnostics_mode)
    actual_peer = (
        peer_payload.get("arm"),
        peer_payload.get("candidate"),
        peer_payload.get("diagnostics"),
    )
    if actual_peer != expected_peer:
        raise ValueError(f"paired-profile receipt identity mismatch: {actual_peer} != {expected_peer}")
    peer_threshold = float(peer_payload.get("threshold", float("nan")))
    if peer_threshold != threshold:
        raise ValueError("paired-profile CV thresholds disagree")
    extend = max(cv, float(peer_payload["cv_2_4"])) > threshold
    decision = {
        "schema_version": "ncbr-paired-profile-decision-v1",
        "candidate": candidate,
        "diagnostics": diagnostics_mode,
        "extend_to_step_6": extend,
        "cv": {arm: cv, peer: float(peer_payload["cv_2_4"])},
    }
    decision_path = coordination_dir / f"{candidate}_{diagnostics_mode}_decision_{arm}.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return extend
