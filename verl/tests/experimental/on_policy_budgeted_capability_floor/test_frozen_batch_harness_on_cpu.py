from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.frozen_batch_harness import (
    FrozenBatchHarness,
    FrozenBatchHarnessConfig,
)
from verl.experimental.on_policy_budgeted_capability_floor.frozen_batch_trainer import (
    RayDAPOFrozenBatchOBCFTrainer,
    FrozenBatchTrainerMixin,
)
from verl.experimental.nondeterminism_diagnostics import data_proto_semantic_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path, *, wrong_batch_hash: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2], [3, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            "advantages": torch.tensor([[1.0, -1.0], [0.5, 0.0]]),
            "returns": torch.tensor([[1.0, -1.0], [0.5, 0.0]]),
            "token_level_scores": torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        },
        non_tensors={
            "uid": np.asarray(["p0", "p1"], dtype=object),
            "prompt_id": np.asarray(["p0", "p1"], dtype=object),
        },
        meta_info={"temperature": 1.0},
    )
    batch_path = tmp_path / "frozen.dp"
    batch.save_to_disk(batch_path)
    batch_manifest = {
        "schema_version": "obcf-frozen-batch-v2",
        "file_sha256": "0" * 64 if wrong_batch_hash else _sha256(batch_path),
        "semantic_sha256": data_proto_semantic_hash(batch),
        "row_count": len(batch),
        "source_commit": "a" * 40,
        "source_config_sha256": "b" * 64,
        "source_checkpoint_fingerprint": "model-identity",
    }
    batch_manifest_path = tmp_path / "frozen.manifest.json"
    batch_manifest_path.write_text(json.dumps(batch_manifest))
    initial_state = {
        "schema_version": "obcf-frozen-initial-state-v2",
        "identities": {
            "actor": "model-identity",
            "optimizer": "optimizer-identity",
            "scheduler": "scheduler-identity",
        },
    }
    initial_state_path = tmp_path / "initial.manifest.json"
    initial_state_path.write_text(json.dumps(initial_state))
    return batch_path, batch_manifest_path, initial_state_path


def _config(tmp_path: Path, mode: str = "baseline", *, wrong_batch_hash: bool = False):
    batch_path, batch_manifest_path, initial_state_path = _write_inputs(
        tmp_path, wrong_batch_hash=wrong_batch_hash
    )
    return FrozenBatchHarnessConfig(
        mode=mode,
        frozen_batch_path=batch_path,
        frozen_batch_manifest_path=batch_manifest_path,
        initial_state_manifest_path=initial_state_path,
        output_dir=tmp_path / "output",
        run_id=f"{mode}-a",
        protocol_fingerprint="p" * 64,
        cache_fingerprint="c" * 64 if mode == "shadow" else None,
    )


def _identity():
    return {
        "actor": "model-identity",
        "optimizer": "optimizer-identity",
        "scheduler": "scheduler-identity",
    }


def _update_result(**overrides):
    result = {
        "actor_update_count": 1,
        "rollout_request_count": 0,
        "cache_loaded_count": 0,
        "prefix_verifier_calls": 0,
        "constraint_observation_count": 0,
        "lambda_update_count": 0,
        "lambda_before": 0.0,
        "lambda_after": 0.0,
        "metrics": {"actor/loss": 0.25},
        "shadow_diagnostics": None,
    }
    result.update(overrides)
    return result


def test_frozen_batch_hash_mismatch_refuses_to_run_before_update(tmp_path):
    calls = []
    harness = FrozenBatchHarness(
        _config(tmp_path, wrong_batch_hash=True),
        initial_state_identity=_identity,
        actor_update=lambda batch, mode: calls.append(mode) or _update_result(),
    )

    with pytest.raises(ValueError, match="frozen batch file hash mismatch"):
        harness.run()

    assert calls == []


def test_initial_checkpoint_optimizer_scheduler_mismatch_refuses_to_run(tmp_path):
    harness = FrozenBatchHarness(
        _config(tmp_path),
        initial_state_identity=lambda: _identity() | {"actor": "wrong-model"},
        actor_update=lambda batch, mode: _update_result(),
    )

    with pytest.raises(ValueError, match="initial state identity mismatch"):
        harness.run()


def test_mode_off_requires_all_obcf_side_effect_counters_zero(tmp_path):
    harness = FrozenBatchHarness(
        _config(tmp_path, "off"),
        initial_state_identity=_identity,
        actor_update=lambda batch, mode: _update_result(),
    )

    result = harness.run()

    assert result["decision"] == "PASS"
    assert result["mode"] == "off"
    assert result["cache_loaded_count"] == 0
    assert result["prefix_verifier_calls"] == 0
    assert result["constraint_observation_count"] == 0
    assert result["lambda_update_count"] == 0
    assert result["capability_batch_field_count"] == 0


def test_mode_shadow_records_prefix_diagnostics_without_changing_total_advantage(tmp_path):
    def shadow_update(batch, mode):
        assert mode == "shadow"
        batch.batch["capability_read_only"] = torch.ones(len(batch), 1)
        return _update_result(
            cache_loaded_count=1,
            prefix_verifier_calls=2,
            constraint_observation_count=1,
            shadow_diagnostics={"mixed_group_count": 1, "all_zero_group_count": 0},
        )

    harness = FrozenBatchHarness(
        _config(tmp_path, "shadow"),
        initial_state_identity=_identity,
        actor_update=shadow_update,
    )

    result = harness.run()

    assert result["decision"] == "PASS"
    assert result["shadow_diagnostics"]["mixed_group_count"] == 1
    assert result["total_advantage_exactly_unchanged"] is True
    assert result["lambda_before"] == result["lambda_after"] == 0.0


def test_harness_rejects_zero_or_multiple_actor_updates(tmp_path):
    for count in (0, 2):
        run_dir = tmp_path / f"count-{count}"
        harness = FrozenBatchHarness(
            _config(run_dir),
            initial_state_identity=_identity,
            actor_update=lambda batch, mode, count=count: _update_result(actor_update_count=count),
        )
        with pytest.raises(ValueError, match="exactly one actor update"):
            harness.run()


def test_harness_rejects_any_rollout_request(tmp_path):
    harness = FrozenBatchHarness(
        _config(tmp_path),
        initial_state_identity=_identity,
        actor_update=lambda batch, mode: _update_result(rollout_request_count=1),
    )

    with pytest.raises(ValueError, match="must not request rollout"):
        harness.run()


def test_harness_does_not_modify_frozen_input_artifact(tmp_path):
    config = _config(tmp_path)
    before = _sha256(Path(config.frozen_batch_path))

    harness = FrozenBatchHarness(
        config,
        initial_state_identity=_identity,
        actor_update=lambda batch, mode: (
            batch.batch.__setitem__("diagnostic_only", torch.ones(len(batch), 1)) or _update_result()
        ),
    )
    result = harness.run()

    assert _sha256(Path(config.frozen_batch_path)) == before
    assert result["frozen_input_file_unchanged"] is True


def test_invalid_mode_fails_closed_before_loading_or_update(tmp_path):
    calls = []
    config = _config(tmp_path)
    config = FrozenBatchHarnessConfig(**({**config.__dict__, "mode": "dual"}))
    harness = FrozenBatchHarness(
        config,
        initial_state_identity=lambda: calls.append("identity") or _identity(),
        actor_update=lambda batch, mode: calls.append("update") or _update_result(),
    )

    with pytest.raises(ValueError, match="invalid frozen-batch mode"):
        harness.run()

    assert calls == []


def test_real_frozen_trainer_adapter_has_no_rollout_or_dynamic_filter_call():
    source = inspect.getsource(FrozenBatchTrainerMixin)

    assert "generate_sequences" not in source
    assert "filter_dapo_generation_batch" not in source
    assert "select_complete_prompt_groups" not in source


def test_real_frozen_trainer_adapter_rejects_a_second_actor_update():
    trainer = FrozenBatchTrainerMixin.__new__(FrozenBatchTrainerMixin)
    trainer._frozen_actor_update_count = 1

    with pytest.raises(RuntimeError, match="more than one actor update"):
        trainer._update_actor(object())


def test_frozen_entrypoint_routes_baseline_and_obcf_without_training_loop_calls():
    entrypoint = (
        Path(__file__).resolve().parents[3]
        / "verl"
        / "experimental"
        / "on_policy_budgeted_capability_floor"
        / "main_dapo_frozen_batch.py"
    )
    source = entrypoint.read_text()

    assert "RayDAPOFrozenBatchBaselineTrainer" in source
    assert "RayDAPOFrozenBatchOBCFTrainer" in source
    assert 'mode == "baseline"' in source
    assert "trainer.fit()" in source
    assert "generate_sequences" not in source
    assert "filter_dapo_generation_batch" not in source


def test_frozen_entrypoint_does_not_materialize_scientific_datasets_that_it_never_iterates():
    entrypoint = (
        Path(__file__).resolve().parents[3]
        / "verl"
        / "experimental"
        / "on_policy_budgeted_capability_floor"
        / "main_dapo_frozen_batch.py"
    )
    source = entrypoint.read_text()

    assert "create_rl_dataset(" not in source
    assert "FrozenPlaceholderDataset" in source


def test_shadow_frozen_replay_loads_baseline_step_zero_without_requiring_obcf_resume_state():
    source = inspect.getsource(RayDAPOFrozenBatchOBCFTrainer._load_frozen_initial_checkpoint)

    assert "RayDAPOProbeCreditTrainer._load_checkpoint(self)" in source
    assert "reference_model_fingerprint" in source
    assert "shadow frozen replay requires zero dual state" in source
