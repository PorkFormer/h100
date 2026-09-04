from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace

import pytest

from tools.ncbr_profile.compare_calibration import compare
from tools.ncbr_profile.create_stage_manifest import files_manifest, revision_provenance
from tools.ncbr_profile.evaluate_overhead import evaluate, overhead_fraction
from tools.ncbr_profile.stage_local_assets import file_hashes, stage
from tools.ncbr_profile.validate_stage_manifest import model_files, revision_metadata
from tools.ncbr_profile.verify_teardown import target_processes
from verl.experimental.natural_continuation_boundary_return.main_dapo_boundary_return import task_runner_options
from verl.single_controller.ray import base as ray_base


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
    result = evaluate(fixed, {"actor": {"off": 2.0, "on": 2.04}, "reward": {"off": 1.0, "on": 1.025}})
    assert result["max_time_overhead_fraction"] == pytest.approx(0.025)
    assert result["max_memory_overhead_fraction"] == pytest.approx(0.015)
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
                            "actor_diagnostics/all/token_count": 50,
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
