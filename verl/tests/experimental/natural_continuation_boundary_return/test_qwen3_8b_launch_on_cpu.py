from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tools.ncbr_profile.resolved_config_diff import DEFAULT_ALLOWED_PATHS, build_receipt
from tools.ncbr_profile.select_qwen3_8b_candidate import select
from tools.ncbr_profile.validate_baseline_completion import validate as validate_baseline_completion
from tools.ncbr_profile.validate_qwen3_8b_smoke import validate as validate_smoke


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "examples/natural_continuation_boundary_return/run_qwen3_8b_profile_fsdp.sh"


def _manifest(tmp_path: Path, node: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"manifest_{node}.json"
    path.write_text(
        json.dumps(
            {
                "model": {"local_path": "/workspace/models/Qwen3-8B-Base"},
                "data": {
                    "train": {"path": "/data/dapo_math_17k_train.parquet"},
                    "AIME2024": {"path": "/data/aime2024.parquet"},
                    "AIME2025": {"path": "/data/aime2025.parquet"},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _resolved_pair(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.yaml"
    v1 = tmp_path / "v1.yaml"
    OmegaConf.save(
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"boundary_return": {"mode": "off"}}}}),
        baseline,
    )
    OmegaConf.save(
        OmegaConf.create({"actor_rollout_ref": {"rollout": {"boundary_return": {"mode": "replace"}}}}),
        v1,
    )
    return baseline, v1


def _expand(tmp_path: Path, *, arm: str, candidate: str, stage: str) -> list[str]:
    manifest_a = _manifest(tmp_path, "A")
    manifest_b = _manifest(tmp_path, "B")
    environment = {
        **os.environ,
        "NCBR_ARM": arm,
        "NCBR_PROFILE_CANDIDATE": candidate,
        "NCBR_STAGE": stage,
        "NCBR_STAGE_MANIFEST": str(manifest_a),
        "NCBR_STAGE_MANIFEST_B": str(manifest_b),
        "NCBR_DIAGNOSTICS_MODE": "off",
        "RAY_ADDRESS": "10.8.191.127:6395",
        "NCBR_PRINT_COMMAND_ONLY": "1",
        "NCBR_TEST_ONLY_ALLOW_UNVALIDATED_PRINT": "1",
        "NCBR_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "NCBR_RUNTIME_ROOT": str(tmp_path / "runtime"),
    }
    if stage == "formal_s300":
        baseline, v1 = _resolved_pair(tmp_path)
        environment.update(
            {
                "NCBR_AUTHORIZE_S300": "AUTHORIZE_QWEN3_8B_S300",
                "NCBR_BASELINE_RESOLVED_CONFIG": str(baseline),
                "NCBR_V1_RESOLVED_CONFIG": str(v1),
                "NCBR_CONFIG_DIFF_RECEIPT": str(tmp_path / "config_diff.json"),
                "NCBR_WANDB_RUN_ID": f"test-{arm}",
            }
        )
        if arm == "v1":
            completion = tmp_path / "baseline_completion.json"
            completion.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "arm": "baseline",
                        "total_training_steps": 300,
                        "completed_step": 300,
                        "checkpoint_status": "PASS",
                        "teardown_status": "PASS",
                    }
                ),
                encoding="utf-8",
            )
            environment["NCBR_BASELINE_COMPLETION_RECEIPT"] = str(completion)
    result = subprocess.run([str(LAUNCHER)], env=environment, text=True, capture_output=True, check=True)
    return shlex.split(result.stdout)


def _overrides(command: list[str]) -> dict[str, str]:
    return dict(item.split("=", 1) for item in command if "=" in item)


@pytest.mark.parametrize(
    ("candidate", "tp", "logprob", "utilization", "seqs", "tokens"),
    [
        ("P0", "4", "1", "0.40", "128", "16384"),
        ("P1", "4", "2", "0.50", "256", "32768"),
        ("P_SAFE", "8", "1", "0.35", "64", "8192"),
    ],
)
def test_profile_candidate_command_expansion(tmp_path, candidate, tp, logprob, utilization, seqs, tokens):
    config = _overrides(_expand(tmp_path, arm="baseline", candidate=candidate, stage="profile"))
    assert config["actor_rollout_ref.rollout.tensor_model_parallel_size"] == tp
    assert config["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"] == logprob
    assert config["actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"] == logprob
    assert config["actor_rollout_ref.rollout.gpu_memory_utilization"] == utilization
    assert config["actor_rollout_ref.rollout.max_num_seqs"] == seqs
    assert config["actor_rollout_ref.rollout.max_num_batched_tokens"] == tokens
    assert config["actor_rollout_ref.rollout.load_format"] == "auto"
    assert config["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] == "true"
    assert config["actor_rollout_ref.ref.fsdp_config.param_offload"] == "true"
    assert config["trainer.n_gpus_per_node"] == "8"
    assert config["trainer.nnodes"] == "2"
    assert config["trainer.total_training_steps"] == "5"


def test_smoke_and_formal_commands_pin_recipe_and_only_change_boundary_mode(tmp_path):
    smoke = _overrides(_expand(tmp_path, arm="v1", candidate="P0", stage="smoke"))
    assert smoke["trainer.total_training_steps"] == "3"
    assert smoke["actor_rollout_ref.rollout.boundary_return.mode"] == "replace"
    baseline = _overrides(_expand(tmp_path / "b", arm="baseline", candidate="P0", stage="formal_s300"))
    v1 = _overrides(_expand(tmp_path / "v", arm="v1", candidate="P0", stage="formal_s300"))
    for config in (baseline, v1):
        assert config["trainer.total_training_steps"] == "300"
        assert config["trainer.test_freq"] == "10"
        assert config["trainer.save_freq"] == "50"
        assert config["data.train_batch_size"] == "256"
        assert config["+data.gen_batch_size"] == "768"
        assert config["actor_rollout_ref.actor.ppo_mini_batch_size"] == "16"
        assert config["actor_rollout_ref.rollout.n"] == "8"
        assert config["actor_rollout_ref.actor.optim.lr"] == "1e-6"
        assert config["data.max_response_length"] == "2048"
        assert config["actor_rollout_ref.rollout.boundary_return.long_response_length"] == "8192"
    assert baseline["actor_rollout_ref.rollout.boundary_return.mode"] == "off"
    assert v1["actor_rollout_ref.rollout.boundary_return.mode"] == "replace"


def test_formal_launcher_is_locked_without_new_approval(tmp_path):
    environment = {
        **os.environ,
        "NCBR_ARM": "baseline",
        "NCBR_PROFILE_CANDIDATE": "P0",
        "NCBR_STAGE": "formal_s300",
        "NCBR_STAGE_MANIFEST": str(_manifest(tmp_path, "A")),
        "NCBR_STAGE_MANIFEST_B": str(_manifest(tmp_path, "B")),
        "NCBR_DIAGNOSTICS_MODE": "off",
        "RAY_ADDRESS": "10.8.191.127:6395",
        "NCBR_PRINT_COMMAND_ONLY": "1",
        "NCBR_TEST_ONLY_ALLOW_UNVALIDATED_PRINT": "1",
    }
    result = subprocess.run([str(LAUNCHER)], env=environment, text=True, capture_output=True)
    assert result.returncode == 3
    assert "locked" in result.stderr


def _full_config() -> dict:
    return {
        "data": {"train_batch_size": 256, "max_response_length": 2048, "seed": 42},
        "actor_rollout_ref": {
            "model": {"path": "/workspace/models/Qwen3-8B-Base"},
            "actor": {"optim": {"lr": 1e-6}, "data_loader_seed": 42},
            "rollout": {
                "tensor_model_parallel_size": 4,
                "n": 8,
                "boundary_return": {"mode": "off", "long_response_length": 8192, "seed": 42},
            },
        },
        "trainer": {"test_freq": 10, "save_freq": 50, "experiment_name": "baseline"},
    }


def _write_pair(tmp_path: Path, left: dict, right: dict) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline, v1 = tmp_path / "baseline.yaml", tmp_path / "v1.yaml"
    OmegaConf.save(OmegaConf.create(left), baseline)
    OmegaConf.save(OmegaConf.create(right), v1)
    return baseline, v1


def test_resolved_config_gate_accepts_only_identity_paths_and_mode(tmp_path):
    left = _full_config()
    right = copy.deepcopy(left)
    right["actor_rollout_ref"]["rollout"]["boundary_return"]["mode"] = "replace"
    right["trainer"]["experiment_name"] = "v1"
    baseline, v1 = _write_pair(tmp_path, left, right)
    receipt = build_receipt(baseline, v1, DEFAULT_ALLOWED_PATHS)
    assert receipt["status"] == "PASS"
    assert {item["path"] for item in receipt["differences"]} == {
        "actor_rollout_ref.rollout.boundary_return.mode",
        "trainer.experiment_name",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("actor_rollout_ref", "model", "path"), "/wrong/model"),
        (("data", "seed"), 7),
        (("data", "train_batch_size"), 128),
        (("actor_rollout_ref", "actor", "optim", "lr"), 2e-6),
        (("data", "max_response_length"), 4096),
        (("actor_rollout_ref", "rollout", "tensor_model_parallel_size"), 8),
        (("trainer", "test_freq"), 20),
        (("trainer", "save_freq"), 100),
    ],
)
def test_resolved_config_gate_rejects_recipe_tampering(tmp_path, path, value):
    left = _full_config()
    right = copy.deepcopy(left)
    right["actor_rollout_ref"]["rollout"]["boundary_return"]["mode"] = "replace"
    cursor = right
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    baseline, v1 = _write_pair(tmp_path, left, right)
    receipt = build_receipt(baseline, v1, DEFAULT_ALLOWED_PATHS)
    assert receipt["status"] == "FAIL"
    assert ".".join(path) in {item["path"] for item in receipt["unexpected_differences"]}


def test_baseline_completion_gate_is_strict():
    valid = {
        "status": "PASS",
        "arm": "baseline",
        "total_training_steps": 300,
        "completed_step": 300,
        "checkpoint_status": "PASS",
        "teardown_status": "PASS",
    }
    validate_baseline_completion(valid)
    with pytest.raises(SystemExit, match="not sufficient"):
        validate_baseline_completion({**valid, "completed_step": 299})


def _profile(candidate: str, *, memory: float, seconds: float, oom: int = 0) -> dict:
    gpu_values = {str(index): memory for index in range(16)}
    return {
        "candidate": candidate,
        "steps": [
            {
                "step": step,
                "valid_optimizer_step": True,
                "total_seconds": seconds,
                "rollout_seconds": seconds / 2,
                "actor_seconds": seconds / 4,
                "normal_tokens_per_second": 1000,
                "candidate_batches": 3,
            }
            for step in range(1, 6)
        ],
        "peak_nvml_memory_gib": gpu_values,
        "peak_allocated_gib": gpu_values,
        "peak_reserved_gib": gpu_values,
        "gpu_sample_interval_seconds": 1,
        "gpu_utilization_distribution": {str(index): {"p50": 90} for index in range(16)},
        "oom_count": oom,
        "worker_loss_count": 0,
        "preemption_count": 0,
        "deadlock": False,
        "vllm_scheduling": {"preemptions": 0},
        "ray_worker_status": {"lost": 0},
    }


def test_profile_selection_uses_frozen_decision_tree():
    result = select(_profile("P0", memory=35, seconds=10), _profile("P1", memory=38, seconds=10.9), None)
    assert result["status"] == "PASS"
    assert result["selected_candidate"] == "P1"
    fallback = select(_profile("P0", memory=37, seconds=10), None, _profile("P_SAFE", memory=38, seconds=12))
    assert fallback["selected_candidate"] == "P_SAFE"
    blocked = select(_profile("P0", memory=37, seconds=10), None, _profile("P_SAFE", memory=39, seconds=12))
    assert blocked["status"] == "FAIL"


def test_smoke_gate_distinguishes_zero_coverage_from_system_failure():
    summary = {
        "completed_steps": [1, 2, 3],
        "continuation_request_count": 4,
        "oom_count": 0,
        "deadlock": False,
        "timeout_count": 0,
        "long_verifier_rows": 4,
        "expected_long_verifier_rows": 4,
        "prefix_penalty_drift_max": 0,
        "nan_count": 0,
        "inf_count": 0,
        "boundary_applied_count": 2,
        "actor_tail_token_count": 0,
        "teardown_status": "PASS",
    }
    assert validate_smoke(summary)["status"] == "PASS"
    zero = validate_smoke({**summary, "continuation_request_count": 0, "long_verifier_rows": 0, "expected_long_verifier_rows": 0})
    assert zero["status"] == "FAIL"
    assert zero["classification"] == "mechanism_coverage_insufficient"
