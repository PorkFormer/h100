from __future__ import annotations

import torch

from verl.experimental.on_policy_budgeted_capability_floor.artifact_batch import (
    build_full_rollout_batch_from_artifacts,
)
from verl.experimental.on_policy_budgeted_capability_floor.prefix_batch import (
    build_exact_prefix_batch,
)


def _artifacts():
    prompts = [
        {
            "prompt_id": 0,
            "prompt_hash": "prompt-0",
            "prompt_token_ids": [10, 11],
            "raw_prompt": [{"role": "user", "content": "question"}],
            "data_source": "math",
            "ground_truth": "42",
            "extra_info": {"split": "audit"},
        }
    ]
    rollouts = [
        {
            "model_id": "base",
            "prompt_id": 0,
            "rollout_index": 1,
            "prompt_hash": "prompt-0",
            "response_hash": "response-1",
            "sampling_seed": 11,
            "prompt_token_ids": [10, 11],
            "response_token_ids": [30],
        },
        {
            "model_id": "base",
            "prompt_id": 0,
            "rollout_index": 0,
            "prompt_hash": "prompt-0",
            "response_hash": "response-0",
            "sampling_seed": 10,
            "prompt_token_ids": [10, 11],
            "response_token_ids": [20, 21, 22],
        },
    ]
    return prompts, rollouts


def test_builds_deterministically_sorted_full_dataproto_with_required_fields():
    prompts, rollouts = _artifacts()
    batch = build_full_rollout_batch_from_artifacts(
        prompt_rows=prompts,
        rollout_rows=rollouts,
        tokenizer=object(),
        pad_token_id=0,
    )

    assert batch.non_tensor_batch["rollout_index"].tolist() == [0, 1]
    assert set(batch.batch.keys()) == {
        "prompts",
        "responses",
        "input_ids",
        "attention_mask",
        "position_ids",
        "response_mask",
    }
    assert batch.batch["prompts"].tolist() == [[10, 11], [10, 11]]
    assert batch.batch["responses"].tolist() == [[20, 21, 22], [30, 0, 0]]
    assert batch.batch["response_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert torch.equal(
        batch.batch["input_ids"],
        torch.cat((batch.batch["prompts"], batch.batch["responses"]), dim=-1),
    )
    assert batch.non_tensor_batch["response_token_count"].tolist() == [3, 1]
    assert batch.non_tensor_batch["reward_model"].tolist() == [
        {"ground_truth": "42"},
        {"ground_truth": "42"},
    ]


def test_exact_token_truncation_adds_no_eos_or_suffix():
    prompts, rollouts = _artifacts()
    full = build_full_rollout_batch_from_artifacts(
        prompt_rows=prompts,
        rollout_rows=rollouts,
        tokenizer=type("Tokenizer", (), {"eos_token_id": 2})(),
        pad_token_id=0,
    )

    prefix = build_exact_prefix_batch(
        batch=full,
        rollout_indices=torch.tensor([0, 1]),
        reference_budget=2,
        pad_token_id=0,
    )

    assert prefix.batch["responses"].tolist() == [[20, 21], [30, 0]]
    assert prefix.batch["response_mask"].tolist() == [[1, 1], [1, 0]]
    assert 2 not in prefix.batch["responses"].tolist()[0]


def test_rollout_prompt_hash_seed_and_required_source_fields_fail_closed():
    prompts, rollouts = _artifacts()
    rollouts[0]["prompt_hash"] = "wrong"
    try:
        build_full_rollout_batch_from_artifacts(
            prompt_rows=prompts,
            rollout_rows=rollouts,
            tokenizer=object(),
            pad_token_id=0,
        )
    except ValueError as error:
        assert "prompt_hash" in str(error)
    else:
        raise AssertionError("prompt hash mismatch must fail")
