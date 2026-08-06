from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer
from verl.experimental.nondeterminism_diagnostics import (
    DiagnosticsConfig,
    NondeterminismDiagnostics,
    data_proto_semantic_hash,
)


def _batch(*, duplicate_request_identity: bool = False, explicit_seed: bool = False) -> DataProto:
    request_ids = ["request-0", "request-0" if duplicate_request_identity else "request-1"]
    rollout_indices = [0, 0 if duplicate_request_identity else 1]
    non_tensors: dict[str, np.ndarray] = {
        "uid": np.asarray(["prompt-group-0", "prompt-group-0"], dtype=object),
        "request_id": np.asarray(request_ids, dtype=object),
        "prompt_id": np.asarray(["prompt-0", "prompt-0"], dtype=object),
        "dataset_row_id": np.asarray([19, 19], dtype=object),
        "rollout_index": np.asarray(rollout_indices, dtype=object),
        "finish_reason": np.asarray(["stop", "length"], dtype=object),
    }
    if explicit_seed:
        non_tensors["sampling_seed"] = np.asarray([101, 102], dtype=object)
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[0, 11, 12], [0, 11, 12]], dtype=torch.long),
            "responses": torch.tensor([[21, 22, 0], [31, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=torch.long
            ),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long),
            "token_level_scores": torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        },
        non_tensors=non_tensors,
        meta_info={"global_steps": 3, "temperature": 1.0},
    )


def _enabled(tmp_path: Path) -> NondeterminismDiagnostics:
    return NondeterminismDiagnostics(
        DiagnosticsConfig(
            enabled=True,
            output_dir=tmp_path,
            run_id="unit-run",
            config_sha256="c" * 64,
            git_commit="a" * 40,
            rank=0,
            ray_actor_identity="ray-actor-0",
            vllm_engine_identity="engine-0",
            tp_group_identity="tp-0",
            dataloader_base_seed=42,
            sampler_generator_hash="s" * 64,
        )
    )


def _capture(writer: NondeterminismDiagnostics, boundary: int, batch: DataProto) -> None:
    writer.capture(
        boundary=boundary,
        batch=batch,
        global_step=3,
        generation_batch_index=1,
        rollout_n=2,
        filter_metric="seq_final_reward",
        completion_order=[1, 0],
        effective_training_batch=boundary == 3,
    )


def test_diagnostics_disabled_creates_no_files_or_fields_and_preserves_order(tmp_path):
    batch = _batch()
    before_hash = data_proto_semantic_hash(batch)
    before_tensor_order = tuple(batch.batch.keys())
    before_non_tensor_order = tuple(batch.non_tensor_batch.keys())
    writer = NondeterminismDiagnostics(DiagnosticsConfig(enabled=False))

    for boundary in range(4):
        _capture(writer, boundary, batch)

    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
    assert data_proto_semantic_hash(batch) == before_hash
    assert tuple(batch.batch.keys()) == before_tensor_order
    assert tuple(batch.non_tensor_batch.keys()) == before_non_tensor_order


def test_diagnostics_enabled_atomically_writes_all_four_boundaries(tmp_path):
    batch = _batch(explicit_seed=True)
    writer = _enabled(tmp_path)

    for boundary in range(4):
        _capture(writer, boundary, batch)

    manifests = sorted(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 4
    assert {json.loads(path.read_text())["boundary"] for path in manifests} == {0, 1, 2, 3}
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest["status"] == "COMPLETE"
        assert manifest["run_id"] == "unit-run"
        assert manifest["config_sha256"] == "c" * 64
        assert manifest["git_commit"] == "a" * 40
        assert (path.parent / "records.jsonl").is_file()
        assert not any(part.name.endswith(".tmp") for part in path.parent.parent.iterdir())


def test_diagnostic_write_failure_never_commits_partial_pass_artifact(tmp_path, monkeypatch):
    batch = _batch()
    writer = _enabled(tmp_path)

    def fail_commit(*args, **kwargs):
        raise OSError("injected diagnostic write failure")

    monkeypatch.setattr(writer, "_atomic_commit", fail_commit)
    with pytest.raises(OSError, match="injected diagnostic write failure"):
        _capture(writer, 0, batch)

    assert list(tmp_path.rglob("manifest.json")) == []
    assert list(tmp_path.rglob("records.jsonl")) == []


def test_duplicate_sample_identity_fails_audit_closed(tmp_path):
    writer = _enabled(tmp_path)
    batch = _batch(duplicate_request_identity=True)

    with pytest.raises(ValueError, match="duplicate sample identity"):
        _capture(writer, 0, batch)

    assert list(tmp_path.rglob("manifest.json")) == []


def test_instrumentation_does_not_modify_dataproto_order_or_semantic_hash(tmp_path):
    writer = _enabled(tmp_path)
    batch = _batch(explicit_seed=True)
    before_hash = data_proto_semantic_hash(batch)
    before_tensor_order = tuple(batch.batch.keys())
    before_non_tensor_order = tuple(batch.non_tensor_batch.keys())

    _capture(writer, 2, batch)

    assert data_proto_semantic_hash(batch) == before_hash
    assert tuple(batch.batch.keys()) == before_tensor_order
    assert tuple(batch.non_tensor_batch.keys()) == before_non_tensor_order


def test_boundary_zero_missing_explicit_seed_saves_null_without_inventing_seed(tmp_path):
    writer = _enabled(tmp_path)
    batch = _batch(explicit_seed=False)

    _capture(writer, 0, batch)

    records_path = next(tmp_path.rglob("records.jsonl"))
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert records
    assert all(record["assigned_sampling_seed"] is None for record in records)
    assert all(record["sampling_seed_status"] == "MISSING_EXPLICIT_REQUEST_SEED" for record in records)


def test_dapo_trainer_diagnostics_are_disabled_by_default():
    trainer = RayDAPOProbeCreditTrainer.__new__(RayDAPOProbeCreditTrainer)
    trainer.config = {"trainer": {}}

    writer = trainer._get_nondeterminism_diagnostics()

    assert writer.config.enabled is False


def test_dapo_fit_captures_four_boundaries_in_scientific_control_flow_order():
    source = inspect.getsource(RayDAPOProbeCreditTrainer.fit)
    positions = [
        re.search(rf"_capture_nondeterminism_boundary\(\s*{boundary},", source).start()
        for boundary in range(4)
    ]

    assert positions == sorted(positions)
    assert positions[0] < source.index("generate_sequences(gen_input)")
    assert source.index("union(gen_output)") < positions[1] < source.index("_score_batch_with_existing_reward_pipeline")
    assert source.index('candidate.batch["token_level_rewards"]') < positions[2]
    assert source.index("filter_dapo_generation_batch") < positions[3]


def test_boundary_zero_handles_agent_loop_non_tensor_batch_with_exact_prompt_override(tmp_path):
    batch = DataProto(
        batch=None,
        non_tensor_batch={
            "uid": np.asarray(["group-0", "group-0"], dtype=object),
            "raw_prompt": np.asarray(
                [
                    [{"role": "user", "content": "one"}],
                    [{"role": "user", "content": "one"}],
                ],
                dtype=object,
            ),
        },
        meta_info={"global_steps": 3},
    )
    writer = _enabled(tmp_path)

    writer.capture(
        boundary=0,
        batch=batch,
        global_step=3,
        generation_batch_index=1,
        rollout_n=2,
        prompt_token_ids_override=[[101, 102, 103], [101, 102, 103]],
    )

    records_path = next(tmp_path.rglob("records.jsonl"))
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert [record["prompt_token_count"] for record in records] == [3, 3]
    assert records[0]["prompt_token_hash"] == records[1]["prompt_token_hash"]
    assert records[0]["prompt_token_identity_status"] == "EXACT_TOKEN_IDS"
