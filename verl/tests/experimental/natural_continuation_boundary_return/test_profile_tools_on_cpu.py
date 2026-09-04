from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import subprocess
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.ncbr_profile.aggregate_fixed_actor_replay import aggregate as aggregate_replay
from tools.ncbr_profile.analyze_mechanism_panel import analyze as analyze_mechanism_panel
from tools.ncbr_profile.annotate_entropy_transitions import annotate
from tools.ncbr_profile.audit_checkpoint_entropy import aggregate
from tools.ncbr_profile.build_cumulative_axes import build as build_cumulative_axes
from tools.ncbr_profile.build_entropy_panel import select_rows
from tools.ncbr_profile.build_s300_gate import REQUIRED as S300_REQUIRED
from tools.ncbr_profile.build_s300_gate import build as build_s300_gate
from tools.ncbr_profile.compare_calibration import compare
from tools.ncbr_profile.compare_calibration_workloads import compare as compare_workloads
from tools.ncbr_profile.create_stage_manifest import files_manifest, revision_provenance
from tools.ncbr_profile.estimate_s300 import estimate
from tools.ncbr_profile.extract_calibration import extract as extract_calibration
from tools.ncbr_profile.evaluate_overhead import evaluate, overhead_fraction
from tools.ncbr_profile.schedule_s300 import choose_assignments
from tools.ncbr_profile.sample_shared_gpus import _parse_csv
from tools.ncbr_profile.select_profile_candidate import UNIT_NAMES, select
from tools.ncbr_profile.stage_local_assets import file_hashes, stage
from tools.ncbr_profile.validate_gate0 import validate as validate_gate0
from tools.ncbr_profile.validate_stage_manifest import model_files, revision_metadata, validate_stage_artifacts
from tools.ncbr_profile.verify_teardown import target_processes
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.natural_continuation_boundary_return.dapo_trainer import (
    RayDAPOBoundaryReturnTrainer,
    _profile_json_default,
)
from verl.experimental.natural_continuation_boundary_return.main_dapo_boundary_return import task_runner_options
from verl.experimental.natural_continuation_boundary_return.runtime import run_boundary_continuations
from verl.experimental.reward_loop import RewardLoopManager
from verl.single_controller.ray import base as ray_base
from verl.utils import ray_utils
from verl.workers.rollout import llm_server


def test_profile_json_default_serializes_numpy_and_torch_telemetry():
    payload = {
        "numpy_scalar": np.int32(7),
        "numpy_array": np.asarray([1.25, 2.5], dtype=np.float32),
        "torch_tensor": torch.tensor([3, 4], dtype=torch.int64),
    }
    encoded = json.dumps(payload, default=_profile_json_default, sort_keys=True)
    assert json.loads(encoded) == {
        "numpy_array": [1.25, 2.5],
        "numpy_scalar": 7,
        "torch_tensor": [3, 4],
    }
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"unsupported": object()}, default=_profile_json_default)


def test_mechanism_panel_rejects_missing_engine_timers():
    receipt = {
        "request_count": 1,
        "continuation_input_tokens": 4,
        "tail_decode_tokens": 2,
        "long_reward_rows": 1,
        "long_reward_full_response_tokens": 6,
        "profiling_intervals": [
            {
                "interval_id": "continuation",
                "name": "boundary_continuation",
                "wall_start": 1.0,
                "wall_end": 2.0,
                "parent_id": None,
                "asynchronous": True,
                "metadata": {},
            },
            {
                "interval_id": "long-row",
                "name": "long_reward_batch_build",
                "wall_start": 2.0,
                "wall_end": 2.1,
                "parent_id": None,
                "asynchronous": False,
                "metadata": {},
            },
            {
                "interval_id": "long-token",
                "name": "long_reward_model_forward",
                "wall_start": 2.1,
                "wall_end": 2.2,
                "parent_id": None,
                "asynchronous": False,
                "metadata": {},
            },
        ],
    }
    result = analyze_mechanism_panel(receipt)
    assert result["status"] == "FAIL"
    assert result["unit_costs"]["u_cont_input"] == 0.0
    assert result["unit_costs"]["u_tail_decode"] == 0.0


def test_shared_gpu_sampler_parses_exact_inventory():
    rows = _parse_csv(
        "0, GPU-a, 123, 40960, 87\n"
        "1, GPU-b, 456, 40960, 0\n"
    )
    assert rows == [
        {
            "index": 0,
            "uuid": "GPU-a",
            "memory_used_mib": 123,
            "memory_total_mib": 40960,
            "utilization_percent": 87.0,
        },
        {
            "index": 1,
            "uuid": "GPU-b",
            "memory_used_mib": 456,
            "memory_total_mib": 40960,
            "utilization_percent": 0.0,
        },
    ]
    with pytest.raises(ValueError, match="unexpected nvidia-smi row"):
        _parse_csv("0, too, few")


def test_shared_ray_node_resource_pins_controller_and_every_gpu_bundle(monkeypatch):
    captured = []

    class FakePlacementGroup:
        def ready(self):
            return object()

    def fake_placement_group(**kwargs):
        captured.append(kwargs)
        return FakePlacementGroup()

    monkeypatch.setattr(ray_base, "placement_group", fake_placement_group)
    monkeypatch.setattr(ray_base.ray, "get", lambda refs: refs)
    monkeypatch.setattr(ray_base, "sort_placement_group_by_node_ip", lambda groups: groups)
    pool = ray_base.RayResourcePool(
        process_on_nodes=[2], max_colocate_count=1, required_resource="ncbr_node_B"
    )
    pool.get_placement_groups()
    assert captured[0]["strategy"] == "STRICT_PACK"
    assert captured[0]["bundles"] == [
        {"CPU": 1, "GPU": 1, "ncbr_node_B": 1e-4},
        {"CPU": 1, "GPU": 1, "ncbr_node_B": 1e-4},
    ]

    class Trainer:
        @staticmethod
        def get(name):
            assert name == "ray_node_resource"
            return "ncbr_node_B"

    class Config:
        trainer = Trainer()

    assert task_runner_options(Config()) == {"num_cpus": 1, "resources": {"ncbr_node_B": 1e-3}}


def test_shared_ray_node_resource_pins_auxiliary_cpu_actors_and_fails_closed(monkeypatch):
    node_a = "a" * 56
    node_b = "b" * 56
    nodes = [
        {"NodeID": node_a, "Alive": True, "Resources": {"CPU": 240, "ncbr_node_A": 1}},
        {"NodeID": node_b, "Alive": True, "Resources": {"CPU": 240, "ncbr_node_B": 1}},
    ]
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: nodes)

    config = {"trainer": {"ray_node_resource": "ncbr_node_A"}}
    assert ray_utils.required_ray_node_resource(config) == "ncbr_node_A"
    assert ray_utils.alive_cpu_node_ids("ncbr_node_A") == [node_a]
    assert ray_utils.required_resource_options(config) == {"resources": {"ncbr_node_A": 1e-3}}

    with pytest.raises(RuntimeError, match="no live CPU node advertises"):
        ray_utils.alive_cpu_node_ids("missing_node")

    actor_options = []

    class FakeRemoteActor:
        @classmethod
        def options(cls, **options):
            actor_options.append(options)
            return cls

        @staticmethod
        def remote(*args, **kwargs):
            return object()

    config = SimpleNamespace(
        trainer={"ray_node_resource": "ncbr_node_A"},
        reward=SimpleNamespace(num_workers=2),
    )
    agent_manager = object.__new__(AgentLoopManager)
    agent_manager.config = config
    agent_manager.rollout_config = SimpleNamespace(agent=SimpleNamespace(num_workers=2))
    agent_manager.llm_client = object()
    agent_manager.teacher_client = None
    agent_manager.reward_loop_worker_handles = None
    agent_manager.agent_loop_workers_class = FakeRemoteActor
    asyncio.run(agent_manager._init_agent_loop_workers())
    assert len(actor_options) == 2
    assert all(item["scheduling_strategy"].node_id == node_a for item in actor_options)
    assert all(item["scheduling_strategy"].soft is False for item in actor_options)

    actor_options.clear()
    reward_manager = object.__new__(RewardLoopManager)
    reward_manager.config = config
    reward_manager.reward_loop_workers_class = FakeRemoteActor
    reward_manager.reward_router_address = None
    reward_manager._init_reward_loop_workers()
    assert len(actor_options) == 2
    assert all(item["scheduling_strategy"].node_id == node_a for item in actor_options)
    assert all(item["scheduling_strategy"].soft is False for item in actor_options)

    load_balancer_options = []

    class FakeLoadBalancer:
        @classmethod
        def options(cls, **options):
            load_balancer_options.append(options)
            return cls

        @staticmethod
        def remote(**kwargs):
            return kwargs

    monkeypatch.setattr(llm_server, "GlobalRequestLoadBalancer", FakeLoadBalancer)
    server_manager = object.__new__(llm_server.LLMServerManager)
    server_manager.config = config
    server_manager.server_addresses = ["server"]
    server_manager.server_handles = [object()]
    asyncio.run(server_manager._init_global_load_balancer())
    assert load_balancer_options == [{"resources": {"ncbr_node_A": 1e-3}}]


def test_local_asset_staging_is_verified_read_only_and_idempotent(tmp_path):
    source_model = tmp_path / "source-model"
    source_model.mkdir()
    (source_model / "config.json").write_text('{"model_type":"qwen3"}\n', encoding="utf-8")
    metadata = source_model / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("revision\n", encoding="utf-8")
    source_data = tmp_path / "source-data"
    source_data.mkdir()
    for name in ("dapo_math_17k_train.parquet", "aime-2024-verl.parquet", "aime-2025-verl.parquet"):
        (source_data / name).write_bytes(name.encode())
    destination = tmp_path / "destination"
    args = Namespace(node="B", destination=destination, model_source=source_model, data_source=source_data)
    stage(args)
    stage(args)
    local_model = destination / "model" / "Qwen3-1.7B-Base"
    assert file_hashes(local_model) == file_hashes(source_model)
    assert not (destination / "node_B_writable_files.txt").read_text(encoding="utf-8")


def test_model_manifest_excludes_mutable_cache_but_attests_revision_metadata(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}\n', encoding="utf-8")
    metadata = model / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("fixed-revision\netag\n0\n", encoding="utf-8")
    manifest = files_manifest(model)
    provenance = revision_provenance(model, "fixed-revision")
    assert set(manifest) == {"config.json"}
    assert set(provenance) == {".cache/huggingface/download/config.json.metadata"}
    assert model_files(model) == manifest
    assert revision_metadata(model) == provenance
    (model / "unexpected.bin").write_bytes(b"new file")
    assert model_files(model) != manifest
    with pytest.raises(SystemExit, match="revision mismatch"):
        revision_provenance(model, "wrong-revision")


def test_late_stage_manifest_binds_selected_candidate_and_diagnostics(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"status": "PASS", "selected_candidate": "P1"}), encoding="utf-8")
    overhead = tmp_path / "overhead.json"
    overhead.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    artifacts = {
        "profile_selection": {"path": str(selection)},
        "diagnostics_overhead_gate": {"path": str(overhead)},
    }
    validate_stage_artifacts("acceptance", "P1", "on", artifacts)
    with pytest.raises(SystemExit, match="profile selection"):
        validate_stage_artifacts("acceptance", "P0", "on", artifacts)
    with pytest.raises(SystemExit, match="diagnostics=on"):
        validate_stage_artifacts("acceptance", "P1", "off", artifacts)


def test_cross_node_manifest_comparison_allows_local_paths_only(tmp_path):
    common = {
        "code_sha": "a" * 40,
        "arm": "baseline",
        "candidate": "P0",
        "stage": "gate0",
        "diagnostics": "on",
        "model": {
            "repo_id": "Qwen/Qwen3-1.7B-Base",
            "revision": "r",
            "model_type": "qwen3",
            "files": {"config.json": "h"},
        },
        "data": {
            "train": {"sha256": "t"},
            "AIME2024": {"sha256": "24"},
            "AIME2025": {"sha256": "25"},
        },
    }
    paths = []
    for node in ("A", "B"):
        manifest = json.loads(json.dumps(common))
        manifest["node"] = node
        manifest["model"]["local_path"] = f"/local/{node}/model"
        for name, record in manifest["data"].items():
            record["path"] = f"/local/{node}/{name}"
        path = tmp_path / f"{node}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            "tools/ncbr_profile/compare_node_manifests.py",
            "--node-a",
            str(paths[0]),
            "--node-b",
            str(paths[1]),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"


def test_overhead_gate_uses_maximum_normalized_stage_cost_and_fixed_replay_memory():
    assert overhead_fraction(100.0, 102.0) == pytest.approx(0.02)
    fixed = {
        "equivalence_pass": True,
        "actor_time_overhead_fraction": 0.01,
        "peak_allocated_overhead_fraction": 0.015,
        "peak_reserved_overhead_fraction": 0.01,
    }
    other_arm = {
        **fixed,
        "actor_time_overhead_fraction": 0.02,
        "peak_allocated_overhead_fraction": 0.018,
    }
    result = evaluate(
        [fixed, other_arm],
        {"actor": {"off": 2.0, "on": 2.04}, "reward": {"off": 1.0, "on": 1.025}},
    )
    assert result["max_time_overhead_fraction"] == pytest.approx(0.025)
    assert result["max_memory_overhead_fraction"] == pytest.approx(0.018)
    assert result["fixed_replay_receipt_count"] == 2
    assert result["status"] == "PASS"


def test_teardown_process_inventory_does_not_match_the_query_process_itself():
    snapshot = """\
 42 1 python verify_teardown.py raylet|gcs_server|vllm
 99 1 /usr/bin/raylet --node-manager-port=7112
100 1 harmless-worker
"""
    assert target_processes(snapshot, current_pid=42) == ["99 1 /usr/bin/raylet --node-manager-port=7112"]


def test_hard_prefix_panel_registration_uses_physical_jsonl_lines(tmp_path):
    source = tmp_path / "rollouts.jsonl"
    rows = [
        {
            "prompt_id": f"prompt-{index:02d}",
            "prompt_token_ids": [1, 2, 3],
            "response_token_ids": list(range(2048)),
            "trajectory_id": f"trajectory-{index:02d}",
            "finish_reason": "length",
            "response_text": "contains-unicode-separator-\u2028-within-one-json-record",
        }
        for index in range(20)
    ]
    with source.open("w", encoding="utf-8") as stream:
        for row in reversed(rows):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    panel = tmp_path / "panel.jsonl"
    manifest = tmp_path / "panel.manifest.json"
    subprocess.run(
        [
            sys.executable,
            "tools/ncbr_profile/register_hard_prefix_panel.py",
            "--input",
            str(source),
            "--output",
            str(panel),
            "--manifest",
            str(manifest),
            "--model-revision",
            "fixed-revision",
            "--seed",
            "42",
        ],
        check=True,
    )
    selected = [json.loads(line) for line in panel.open(encoding="utf-8")]
    assert len(selected) == 20
    assert selected[0]["prompt_id"] == "prompt-00"
    assert json.loads(manifest.read_text(encoding="utf-8"))["request_count"] == 20


def test_cross_node_calibration_uses_geometric_mean_and_requests_crossover_above_five_percent():
    left = {
        "normal_decode_tokens_per_second": 100.0,
        "reward_full_response_tokens_per_second": 100.0,
        "actor_valid_tokens_per_second": 100.0,
        "candidate_batches_per_second": 100.0,
        "gpu_utilization_mean": 90.0,
    }
    right = {**left, "normal_decode_tokens_per_second": 110.0, "gpu_utilization_mean": 92.0}
    result = compare(left, right)
    component = result["components"]["normal_decode_tokens_per_second"]
    assert component["geometric_mean_reference"] == pytest.approx(11000**0.5)
    assert component["node_throughput_factor"]["A"] < 1
    assert component["node_throughput_factor"]["B"] > 1
    assert result["crossover_required"]


def test_calibration_extractor_accepts_prefixed_actor_metrics_and_parent_intervals():
    intervals = [
        ("normal_rollout", 0.0, 10.0),
        ("short_reward", 10.0, 11.0),
        ("dynamic_sampling_filter", 11.0, 12.0),
        ("old_and_reference_log_prob", 12.0, 16.0),
        ("advantage_and_actor_update", 16.0, 21.0),
    ]
    profile = {
        "intervals": [
            {
                "interval_id": name,
                "name": name,
                "wall_start": start,
                "wall_end": end,
                "parent_id": None,
                "asynchronous": False,
                "metadata": {},
            }
            for name, start, end in intervals
        ],
        "step_metrics": {
            "train/generated_response_tokens": 100,
            "train/num_gen_batches": 2,
            "actor/actor_diagnostics/all/token_count": 50,
        },
    }
    result = extract_calibration(profile, {"gpu_utilization_percent": [80, 100]})
    assert result["normal_decode_tokens_per_second"] == pytest.approx(10)
    assert result["reward_full_response_tokens_per_second"] == pytest.approx(100)
    assert result["actor_valid_tokens_per_second"] == pytest.approx(50 / 9)
    assert result["candidate_batches_per_second"] == pytest.approx(1)
    assert result["actor_interval_source"] == "combined_parent_intervals"


def test_profile_analyzer_derives_workload_unit_costs_and_coverage(tmp_path):
    interval_names = (
        ("normal_rollout", 0.0, 10.0, {"normal_decode_tokens": 100}),
        ("short_reward", 10.0, 11.0, {}),
        ("dynamic_sampling_filter", 11.0, 12.0, {}),
        ("old_log_prob", 12.0, 14.0, {"actor_valid_tokens": 50}),
        ("reference_log_prob", 14.0, 16.0, {"actor_valid_tokens": 50}),
        ("advantage", 16.0, 17.0, {"actor_valid_tokens": 50}),
        ("actor_update", 17.0, 21.0, {"actor_valid_tokens": 50}),
    )
    profile = tmp_path / "profile.jsonl"
    with profile.open("w", encoding="utf-8") as stream:
        for step in range(1, 5):
            intervals = [
                {
                    "interval_id": f"{step}-{name}",
                    "name": name,
                    "wall_start": start,
                    "wall_end": end,
                    "parent_id": None,
                    "asynchronous": False,
                    "metadata": metadata,
                }
                for name, start, end, metadata in interval_names
            ]
            stream.write(
                json.dumps(
                    {
                        "intervals": intervals,
                        "trainer_timing_raw": {"step": 22.0},
                        "step_metrics": {
                            "train/generated_response_tokens": 100,
                            "train/num_gen_batches": 1,
                            "actor/actor_diagnostics/all/token_count": 50,
                            "boundary_return/continuation_request_count": 0,
                        },
                    }
                )
                + "\n"
            )
    output = tmp_path / "analysis.json"
    subprocess.run(
        [
            sys.executable,
            "tools/ncbr_profile/analyze_profile.py",
            "--input",
            str(profile),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["step_wall_cv"] == 0
    assert result["stable_window_unit_cost_medians"]["u_normal"] == pytest.approx(0.1)
    assert result["stable_window_unit_cost_medians"]["u_actor"] == pytest.approx(0.18)
    assert result["mechanism_coverage_insufficient"]


def _write_calibration_boundary(root, *, response_hash="same"):
    record = {
        "generation_batch_index": 1,
        "request_order": 0,
        "prompt_token_hash": "prompt",
        "prompt_token_count": 3,
        "rollout_index": 0,
        "response_token_ids_hash": response_hash,
        "response_length": 2,
        "finish_reason": "stop",
    }
    base = root / "run" / "rank_00000" / "step_000001" / "generation_batch_0001"
    for boundary in (0, 1):
        target = base / f"boundary_{boundary}_candidate"
        target.mkdir(parents=True)
        payload = json.dumps(record, sort_keys=True) + "\n"
        (target / "records.jsonl").write_text(payload, encoding="utf-8")
        manifest = {
            "status": "COMPLETE",
            "boundary": boundary,
            "global_step": 1,
            "generation_batch_index": 1,
            "effective_training_batch": False,
            "record_count": 1,
            "records_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_calibration_workload_comparison_requires_inputs_not_stochastic_outputs(tmp_path):
    node_a = tmp_path / "a"
    node_b = tmp_path / "b"
    _write_calibration_boundary(node_a)
    _write_calibration_boundary(node_b)
    assert compare_workloads(node_a, node_b)["status"] == "PASS"
    changed = tmp_path / "changed"
    _write_calibration_boundary(changed, response_hash="different")
    result = compare_workloads(node_a, changed)
    assert result["status"] == "PASS"
    assert not result["stochastic_response_reward_and_retained_outputs_match_observed"]


def test_calibration_workload_comparison_fails_on_prompt_input_change(tmp_path):
    node_a = tmp_path / "a"
    node_b = tmp_path / "b"
    _write_calibration_boundary(node_a)
    _write_calibration_boundary(node_b)
    records_path = next(node_b.rglob("boundary_0_candidate/records.jsonl"))
    record = json.loads(records_path.read_text(encoding="utf-8"))
    record["prompt_token_hash"] = "different-prompt"
    payload = json.dumps(record, sort_keys=True) + "\n"
    records_path.write_text(payload, encoding="utf-8")
    manifest_path = records_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert compare_workloads(node_a, node_b)["status"] == "FAIL"


def _profile_analysis(unit_scale):
    workloads = {
        "request_count": 10.0,
        "continuation_input_tokens": 30.0,
        "tail_decode_tokens": 20.0,
        "long_reward_rows": 10.0,
        "long_reward_full_response_tokens": 50.0,
        "normal_decode_tokens": 100.0,
        "normal_trajectories": 80.0,
        "actor_valid_tokens": 60.0,
        "candidate_batches": 2.0,
    }
    return {
        "records": [{"workloads": workloads} for _ in range(4)],
        "unstable": False,
        "stable_window_unit_cost_medians": {
            name: unit_scale
            for name in (
                "u_request",
                "u_cont_input",
                "u_tail_decode",
                "u_long_row",
                "u_long_token",
                "u_normal",
                "u_actor",
                "u_candidate",
            )
        },
    }


def test_profile_selector_applies_baseline_gate_then_documented_tiebreak(tmp_path):
    profiles = {}
    scales = {"P0": 1.0, "P1": 1.01, "P2": 1.3}
    for candidate, scale in scales.items():
        profiles[candidate] = {}
        for arm in ("baseline", "v1"):
            path = tmp_path / f"{candidate}_{arm}.json"
            path.write_text(json.dumps(_profile_analysis(scale)), encoding="utf-8")
            profiles[candidate][arm] = {"analysis": path.name, "node": "A", "safety_status": "PASS"}
    unit_factors = {
        node: {
            name: 1.0
            for name in (
                "u_request",
                "u_cont_input",
                "u_tail_decode",
                "u_long_row",
                "u_long_token",
                "u_normal",
                "u_actor",
                "u_candidate",
            )
        }
        for node in ("A", "B")
    }
    systems = {
        candidate: {
            "fixed_workload_peak_memory_bytes": memory,
            "retry_warning_count": 0,
            "tensor_model_parallel_size": 1 if candidate != "P2" else 2,
            "optimizer_offload": candidate != "P1",
            "ref_param_offload": candidate != "P1",
            "max_num_seqs": 384 if candidate == "P1" else 256,
            "gpu_memory_utilization": 0.55 if candidate == "P1" else 0.45,
        }
        for candidate, memory in {"P0": 200, "P1": 100, "P2": 50}.items()
    }
    result = select(
        {"profiles": profiles, "node_unit_cost_factors": unit_factors, "systems": systems},
        tmp_path,
    )
    assert "baseline_more_than_10_percent_slower" in result["excluded"]["P2"]
    assert result["score_tie_candidates_within_5_percentage_points"] == ["P0", "P1"]
    assert result["selected_candidate"] == "P1"


def test_profile_selector_excludes_a_safety_failure_without_requiring_analysis(tmp_path):
    profiles = {}
    for candidate in ("P0", "P1", "P2"):
        profiles[candidate] = {}
        for arm in ("baseline", "v1"):
            if candidate == "P2":
                profiles[candidate][arm] = {"node": "A", "safety_status": "FAIL"}
                continue
            path = tmp_path / f"{candidate}_{arm}.json"
            path.write_text(json.dumps(_profile_analysis(1.0)), encoding="utf-8")
            profiles[candidate][arm] = {"analysis": path.name, "node": "A", "safety_status": "PASS"}
    factors = {node: {name: 1.0 for name in UNIT_NAMES} for node in ("A", "B")}
    systems = {
        candidate: {
            "fixed_workload_peak_memory_bytes": 100,
            "retry_warning_count": 0,
            "tensor_model_parallel_size": 1,
            "optimizer_offload": True,
            "ref_param_offload": True,
            "max_num_seqs": 256,
            "gpu_memory_utilization": 0.45,
        }
        for candidate in ("P0", "P1", "P2")
    }
    result = select({"profiles": profiles, "node_unit_cost_factors": factors, "systems": systems}, tmp_path)
    assert result["selected_candidate"] in {"P0", "P1"}
    assert result["excluded"]["P2"] == [
        "baseline:analysis_unavailable",
        "baseline:safety",
        "v1:analysis_unavailable",
        "v1:safety",
    ]


def test_entropy_panel_and_aggregation_keep_benchmarks_and_estimators_separate():
    rows = [
        {
            "benchmark": benchmark,
            "prompt_id": f"{benchmark}-q0",
            "rollout_index": index,
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4],
        }
        for benchmark in ("AIME2024", "AIME2025")
        for index in (0, 8, 16, 24)
    ]
    selected = select_rows(rows)
    assert len(selected) == 8
    measured = [
        {
            **row,
            "categorical_entropy": [1.0, 3.0],
            "cap_hit": row["rollout_index"] == 24,
            "answer_transition": "wrong_to_correct" if row["rollout_index"] == 24 else "unchanged",
        }
        for row in selected
    ]
    result = aggregate(measured, bucket_size=1, prefix_length=1)
    assert set(result) == {"AIME2024", "AIME2025"}
    for benchmark in result:
        first_bucket = next(item for item in result[benchmark] if item["cohort"] == "0000_0001")
        assert first_bucket["token_weighted"] == 1.0
        assert first_bucket["sequence_balanced"] == 1.0
        assert first_bucket["trajectory_count"] == 4


def test_distributed_fixed_replay_requires_all_ranks_and_uses_worst_overhead(tmp_path):
    for rank, overhead in enumerate((0.01, 0.02)):
        receipt = {
            "rank": rank,
            "world_size": 2,
            "equivalence_pass": True,
            "max_diagnostic_reduction_calls_per_optimizer_step": 1,
            "actor_time_overhead_fraction": overhead,
            "peak_allocated_overhead_fraction": overhead / 2,
            "peak_reserved_overhead_fraction": overhead / 2,
        }
        (tmp_path / f"rank_{rank:05d}.json").write_text(json.dumps(receipt), encoding="utf-8")
    result = aggregate_replay(tmp_path, 2)
    assert result["status"] == "PASS"
    assert result["actor_time_overhead_fraction"] == pytest.approx(0.02)
    assert result["peak_allocated_overhead_fraction"] == pytest.approx(0.01)


def test_s300_estimate_includes_step0_thirty_validations_and_six_checkpoints():
    arm = {
        "training_step_seconds": {"early": 1, "moderate": 2, "stress": 3},
        "validation_step0_seconds": 10,
        "validation_periodic_seconds": 5,
        "checkpoint_seconds": 2,
        "checkpoint_bytes": 100,
        "uncertainty_fraction": {"early": 0.1, "moderate": 0.2, "stress": 0.3},
    }
    result = estimate({"arms": {"baseline": arm, "v1": arm}})
    moderate = result["scenarios"]["moderate"]
    assert moderate["arms"]["baseline"]["wall_clock_seconds"] == 600 + 10 + 150 + 12
    assert moderate["arms"]["baseline"]["checkpoint_disk_bytes"] == 600
    assert moderate["combined_gpu_hours"] == pytest.approx(2 * (772 * 8 / 3600))


def test_cumulative_axes_use_measured_work_and_keep_baseline_continuation_zero():
    records = [
        {
            "step_metrics": {
                "training/global_step": 2,
                "train/generated_prompt_groups": 4,
                "train/generated_response_tokens": 40,
                "actor/actor_diagnostics/all/token_count": 30,
            },
            "trainer_timing_raw": {"step": 20},
        },
        {
            "training/global_step": 1,
            "train/generated_prompt_groups": 3,
            "train/generated_response_tokens": 30,
            "actor_diagnostics/all/token_count": 20,
            "timing_s/step": 10,
        },
    ]
    rows = build_cumulative_axes(records, "baseline")
    assert [row["optimizer_step"] for row in rows] == [1, 2]
    assert rows[-1] == {
        "optimizer_step": 2,
        "candidate_prompts": 7.0,
        "normal_decode_tokens": 70.0,
        "continuation_input_tokens": 0.0,
        "continuation_tail_decode_tokens": 0.0,
        "actor_valid_tokens": 50.0,
        "wall_clock_seconds": 30.0,
        "gpu_hours": pytest.approx(30 * 8 / 3600),
    }
    records[0]["step_metrics"]["boundary_return/continuation_input_tokens"] = 1
    with pytest.raises(ValueError, match="exactly zero"):
        build_cumulative_axes(records, "baseline")


def test_s300_scheduler_assigns_baseline_first_and_never_uses_partial_nodes():
    assert choose_assignments(["B"], {}) == [("baseline", "B")]
    assert choose_assignments(["A", "B"], {}) == [("baseline", "A"), ("v1", "B")]
    assert choose_assignments(["A"], {"baseline": "B"}) == [("v1", "A")]
    assert choose_assignments([], {}) == []


def test_entropy_answer_transitions_join_on_benchmark_prompt_and_rollout():
    base = [
        {"benchmark": "AIME2024", "prompt_id": "q", "rollout_index": 0, "correctness": 0.0},
        {"benchmark": "AIME2025", "prompt_id": "q", "rollout_index": 0, "correctness": 1.0},
    ]
    checkpoint = [
        {"benchmark": "AIME2024", "prompt_id": "q", "rollout_index": 0, "correctness": 1.0},
        {"benchmark": "AIME2025", "prompt_id": "q", "rollout_index": 0, "correctness": 0.0},
    ]
    rows = annotate(base, checkpoint)
    assert [row["answer_transition"] for row in rows] == ["wrong_to_correct", "correct_to_wrong"]


def test_s300_gate_requires_every_receipt_and_keeps_step600_unauthorized(tmp_path):
    receipts = {}
    for name in S300_REQUIRED:
        path = tmp_path / f"{name}.json"
        payload = {"status": "PASS"}
        if name == "selection":
            payload["selected_candidate"] = "P1"
            payload["mechanism_required_candidates"] = []
        if name == "overhead":
            payload["fixed_replay_receipt_count"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")
        receipts[name] = path.name
    result = build_s300_gate(
        {
            "code_sha": "a" * 40,
            "selected_candidate": "P1",
            "mechanism_required": False,
            "receipts": receipts,
        },
        tmp_path,
    )
    assert result["s300_authorized"]
    assert not result["scope"]["step_600_authorized"]
    assert not result["scope"]["second_seed_authorized"]


def test_gate0_validates_every_v1_candidate_batch_event_sequence():
    records = [
        {
            "cycle": cycle,
            "target_cycles": 3,
            "candidate_batches": 1,
            "retained_uid_groups": 256,
            "retained_trajectories": 2048,
            "stopped_before": ["old_log_prob", "ref_log_prob", "advantage", "actor_update"],
            "status": "PASS",
            "arm": "v1",
        }
        for cycle in (1, 2, 3)
    ]
    complete_batch = "\n".join(
        (
            "boundary_return event=short_reward_complete",
            "boundary_return event=continuation_complete",
            "boundary_return event=long_reward_complete",
            "boundary_return event=filter metric=boundary_acc",
        )
    )
    assert validate_gate0(records, "v1", "\n".join([complete_batch] * 3))["status"] == "PASS"
    malformed = "\n".join([complete_batch, complete_batch, "boundary_return event=short_reward_complete"])
    assert validate_gate0(records, "v1", malformed)["status"] == "FAIL"


def test_continuation_complete_audit_event_has_one_owner():
    marker = "boundary_return event=continuation_complete"
    assert marker in inspect.getsource(run_boundary_continuations)
    assert marker not in inspect.getsource(
        RayDAPOBoundaryReturnTrainer._process_candidate_after_reward_before_filter
    )
