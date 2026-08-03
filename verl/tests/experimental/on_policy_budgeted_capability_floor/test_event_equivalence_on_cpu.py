from __future__ import annotations

import copy

import numpy as np
import pytest

from verl.experimental.on_policy_budgeted_capability_floor.event_equivalence import (
    compare_prefix_events,
    prefix_protocol_fingerprint,
)
from tools.on_policy_budgeted_capability_floor.validate_floor_event_equivalence import (
    extract_binary_acc_from_reward_result,
)


BUDGET = 2048


def _rows(rewards=(False, True)):
    return [
        {
            "model_id": "base",
            "prompt_id": 7,
            "rollout_index": index,
            "prompt_hash": "prompt-hash",
            "response_hash": f"response-{index}",
            "response_token_count": 10 + index,
            "sampling_seed": 100 + index,
            f"prefix_reward_{BUDGET}": reward,
            f"prefix_error_{BUDGET}": None,
        }
        for index, reward in enumerate(rewards)
    ]


def test_exact_match_passes_and_input_order_is_irrelevant():
    historical = _rows()
    recomputed = list(reversed(copy.deepcopy(historical)))

    report = compare_prefix_events(
        historical_rows=historical,
        recomputed_rows=recomputed,
        reference_budget=BUDGET,
    )

    assert report.row_count == 2
    assert report.exact_match_count == 2
    assert report.mismatch_count == 0
    assert report.passed is True


def test_directional_mismatches_fail():
    historical = _rows((True, False))
    recomputed = _rows((False, True))

    report = compare_prefix_events(
        historical_rows=historical,
        recomputed_rows=recomputed,
        reference_budget=BUDGET,
    )

    assert report.mismatch_count == 2
    assert report.historical_true_recomputed_false_count == 1
    assert report.historical_false_recomputed_true_count == 1
    assert report.passed is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate"),
        (lambda rows: rows.pop(), "identities"),
        (lambda rows: rows[0].update(response_hash="wrong"), "response_hash"),
        (lambda rows: rows[0].update(sampling_seed=999), "sampling_seed"),
    ],
)
def test_duplicate_missing_or_metadata_mismatch_fails_closed(mutation, match):
    historical = _rows()
    recomputed = copy.deepcopy(historical)
    mutation(recomputed)
    with pytest.raises(ValueError, match=match):
        compare_prefix_events(
            historical_rows=historical,
            recomputed_rows=recomputed,
            reference_budget=BUDGET,
        )


def test_historical_and_recomputed_errors_fail_attestation():
    historical = _rows()
    historical[0][f"prefix_error_{BUDGET}"] = "historical failure"
    recomputed = _rows()
    recomputed[1][f"prefix_error_{BUDGET}"] = "recomputed failure"
    recomputed[1][f"prefix_reward_{BUDGET}"] = None

    report = compare_prefix_events(
        historical_rows=historical,
        recomputed_rows=recomputed,
        reference_budget=BUDGET,
    )

    assert report.historical_error_count == 1
    assert report.recomputed_error_count == 1
    assert report.passed is False


def test_nonbinary_acc_fails_without_shaped_reward_fallback():
    assert extract_binary_acc_from_reward_result(
        {"reward_score": 99.0, "reward_extra_info": {"acc": 1}}
    ) is True
    assert extract_binary_acc_from_reward_result(
        {"reward_score": 99.0, "reward_extra_info": {"acc": np.float32(0)}}
    ) is False
    with pytest.raises(ValueError, match="binary acc"):
        extract_binary_acc_from_reward_result(
            {"reward_score": 1.0, "reward_extra_info": {"acc": 0.5}}
        )
    with pytest.raises(ValueError, match="acc"):
        extract_binary_acc_from_reward_result({"reward_score": 1.0})


def test_protocol_fingerprint_binds_budget_and_reward_pipeline():
    kwargs = dict(
        reference_budget=BUDGET,
        tokenizer_fingerprint="tok",
        chat_template_fingerprint="template",
        verifier_fingerprint="verifier-a",
    )
    fingerprint = prefix_protocol_fingerprint(**kwargs)
    assert fingerprint == prefix_protocol_fingerprint(**kwargs)
    assert fingerprint != prefix_protocol_fingerprint(**(kwargs | {"reference_budget": 1024}))
    assert fingerprint != prefix_protocol_fingerprint(
        **(kwargs | {"verifier_fingerprint": "verifier-b"})
    )
