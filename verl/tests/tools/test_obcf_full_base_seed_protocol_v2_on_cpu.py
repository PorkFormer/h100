from __future__ import annotations

import copy

import pytest

from tools.on_policy_budgeted_capability_floor.audit_full_base_generation import (
    SEED_PROTOCOL_VERSION,
    audit_generation_rows_v2,
    derive_sample_uid,
    enrich_generation_rows_v2,
    frozen_sampling_seed,
    response_token_hash,
    validate_generation_protocol_attestation_v2,
)


DATASET_FP = "d" * 64
CONFIG_FP = "c" * 64
TOKENIZER_FP = "t" * 64
TEMPLATE_FP = "h" * 64
MODEL_FP = "m" * 64


def _row(prompt_id: int, rollout_index: int, *, seed: int | None = None) -> dict:
    prompt_hash = f"prompt-{prompt_id}"
    response_tokens = [100 + prompt_id % 10, 200 + rollout_index]
    return {
        "model_id": "base",
        "prompt_id": prompt_id,
        "rollout_index": rollout_index,
        "sampling_seed": frozen_sampling_seed(42, prompt_id, rollout_index) if seed is None else seed,
        "prompt_hash": prompt_hash,
        "prompt_token_ids": [10, prompt_id],
        "prompt_token_count": 2,
        "response_token_ids": response_tokens,
        "response_token_count": len(response_tokens),
        "response_hash": response_token_hash(response_tokens),
        "config_fingerprint": CONFIG_FP,
        "prompt_shard": f"{prompt_id % 16}/16",
        "generation_error": None,
    }


def _audit(rows: list[dict]) -> dict:
    return audit_generation_rows_v2(
        rows=rows,
        master_seed=42,
        dataset_fingerprint=DATASET_FP,
        generation_config_fingerprint=CONFIG_FP,
    )


def _auditable_enriched_rows(rows: list[dict]) -> list[dict]:
    prompts = {
        row["prompt_id"]: {
            "prompt_id": row["prompt_id"],
            "prompt_hash": row["prompt_hash"],
            "prompt_token_ids": row["prompt_token_ids"],
        }
        for row in rows
    }
    return enrich_generation_rows_v2(
        rows=rows,
        prompts_by_id=prompts,
        master_seed=42,
        dataset_fingerprint=DATASET_FP,
        generation_config_fingerprint=CONFIG_FP,
        tokenizer_fingerprint=TOKENIZER_FP,
        chat_template_fingerprint=TEMPLATE_FP,
        model_fingerprint=MODEL_FP,
    )


def test_cross_prompt_scalar_seed_collision_passes_and_preserves_identities():
    # These are one of the six observed frozen-rule collisions in the full artifact.
    rows = [_row(3995, 1), _row(9614, 1)]
    assert rows[0]["sampling_seed"] == rows[1]["sampling_seed"] == 1025237892

    report = _audit(rows)

    assert report["passed"] is True
    assert report["cross_prompt_seed_collision_pair_count"] == 1
    assert report["cross_prompt_seed_collision_row_count"] == 2
    collision = report["cross_prompt_seed_collisions"][0]
    assert collision["sampling_seed"] == 1025237892
    assert collision["identities"] == [
        {"prompt_id": 3995, "rollout_index": 1},
        {"prompt_id": 9614, "rollout_index": 1},
    ]
    assert report["within_prompt_seed_collision_count"] == 0
    assert report["wrong_seed_rule_count"] == 0


def test_within_prompt_scalar_seed_collision_fails_closed():
    rows = [_row(7, 0), _row(7, 1)]
    rows[1]["sampling_seed"] = rows[0]["sampling_seed"]

    report = _audit(rows)

    assert report["passed"] is False
    assert report["within_prompt_seed_collision_count"] == 1
    assert report["wrong_seed_rule_count"] == 1


def test_duplicate_sample_identity_fails_closed():
    row = _row(7, 0)
    report = _audit([row, copy.deepcopy(row)])

    assert report["passed"] is False
    assert report["duplicate_sample_identity_count"] == 1


def test_duplicate_declared_sample_uid_fails_closed():
    rows = [_row(7, 0), _row(8, 0)]
    rows[0]["sample_uid"] = "f" * 64
    rows[1]["sample_uid"] = "f" * 64

    report = _audit(rows)

    assert report["passed"] is False
    assert report["duplicate_sample_uid_count"] == 1
    assert report["sample_uid_mismatch_count"] == 2


def test_wrong_frozen_seed_rule_fails_closed():
    row = _row(7, 0)
    row["sampling_seed"] += 1

    report = _audit([row])

    assert report["passed"] is False
    assert report["wrong_seed_rule_count"] == 1


def test_cross_prompt_collision_rows_are_not_deleted_or_rewritten():
    rows = [_row(3995, 1), _row(9614, 1)]
    before = copy.deepcopy(rows)

    report = _audit(rows)

    assert report["passed"] is True
    assert rows == before
    assert report["row_count"] == len(before)


def test_collision_result_is_bound_into_protocol_attestation_hash():
    colliding = _audit([_row(3995, 1), _row(9614, 1)])
    noncolliding = _audit([_row(3995, 1), _row(9614, 2)])

    assert colliding["seed_collision_attestation_sha256"] != noncolliding[
        "seed_collision_attestation_sha256"
    ]
    assert colliding["generation_protocol_fingerprint"] != noncolliding[
        "generation_protocol_fingerprint"
    ]


def test_old_protocol_artifact_cannot_be_validated_as_v2():
    old = {
        "schema_version": 1,
        "protocol_version": "obcf-full-base-generation-seed-protocol-v1",
        "passed": True,
    }
    with pytest.raises(ValueError, match="protocol_version"):
        validate_generation_protocol_attestation_v2(old)


def test_generation_provenance_fingerprints_are_fail_closed_and_attested():
    source = [_row(7, 0)]
    enriched = _auditable_enriched_rows(source)
    prompts = {
        7: {
            "prompt_hash": source[0]["prompt_hash"],
            "prompt_token_ids": source[0]["prompt_token_ids"],
        }
    }
    report = audit_generation_rows_v2(
        rows=enriched,
        master_seed=42,
        dataset_fingerprint=DATASET_FP,
        generation_config_fingerprint=CONFIG_FP,
        expected_prompts_by_id=prompts,
        tokenizer_fingerprint=TOKENIZER_FP,
        chat_template_fingerprint=TEMPLATE_FP,
        model_fingerprint=MODEL_FP,
    )
    assert report["passed"] is True
    validate_generation_protocol_attestation_v2(report)

    tampered_rows = copy.deepcopy(enriched)
    tampered_rows[0]["tokenizer_fingerprint"] = "wrong"
    tampered_report = audit_generation_rows_v2(
        rows=tampered_rows,
        master_seed=42,
        dataset_fingerprint=DATASET_FP,
        generation_config_fingerprint=CONFIG_FP,
        expected_prompts_by_id=prompts,
        tokenizer_fingerprint=TOKENIZER_FP,
        chat_template_fingerprint=TEMPLATE_FP,
        model_fingerprint=MODEL_FP,
    )
    assert tampered_report["passed"] is False
    assert tampered_report["tokenizer_fingerprint_mismatch_count"] == 1

    tampered_attestation = copy.deepcopy(report)
    tampered_attestation["model_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_generation_protocol_attestation_v2(tampered_attestation)


def test_enrichment_preserves_identity_response_ids_and_hashes_exactly():
    source = [_row(7, 0), _row(8, 0)]
    for row in source:
        row.pop("response_hash")
        row.pop("generation_error")
    before = copy.deepcopy(source)
    prompts = {
        row["prompt_id"]: {
            "prompt_id": row["prompt_id"],
            "prompt_hash": row["prompt_hash"],
            "prompt_token_ids": row["prompt_token_ids"],
            "raw_prompt": [{"role": "user", "content": f"problem-{row['prompt_id']}"}],
            "extra_info": {"source": "test"},
        }
        for row in source
    }

    enriched = enrich_generation_rows_v2(
        rows=source,
        prompts_by_id=prompts,
        master_seed=42,
        dataset_fingerprint=DATASET_FP,
        generation_config_fingerprint=CONFIG_FP,
        tokenizer_fingerprint=TOKENIZER_FP,
        chat_template_fingerprint=TEMPLATE_FP,
        model_fingerprint=MODEL_FP,
    )

    assert source == before
    assert len(enriched) == len(source)
    for old, new in zip(source, enriched, strict=True):
        assert (new["prompt_id"], new["rollout_index"]) == (
            old["prompt_id"],
            old["rollout_index"],
        )
        assert new["response_token_ids"] == old["response_token_ids"]
        assert new["response_hash"] == response_token_hash(old["response_token_ids"])
        assert new["sample_uid"] == derive_sample_uid(
            dataset_fingerprint=DATASET_FP,
            prompt_id=old["prompt_id"],
            rollout_index=old["rollout_index"],
            prompt_hash=old["prompt_hash"],
            generation_config_fingerprint=CONFIG_FP,
        )
        assert new["seed_protocol_version"] == SEED_PROTOCOL_VERSION
        assert new["generation_error"] is None
