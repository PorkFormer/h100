"""Exact offline/online response-prefix event identities and comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


PREFIX_PROTOCOL_VERSION = "exact-response-token-prefix-v1"
_MATCH_FIELDS = (
    "prompt_hash",
    "response_hash",
    "response_token_count",
    "sampling_seed",
)


def prefix_protocol_fingerprint(
    *,
    reference_budget: int,
    tokenizer_fingerprint: str,
    chat_template_fingerprint: str,
    verifier_fingerprint: str,
) -> str:
    """Bind exact-token prefix semantics to every scoring-sensitive identity."""
    if (
        not isinstance(reference_budget, int)
        or isinstance(reference_budget, bool)
        or reference_budget <= 0
    ):
        raise ValueError("reference_budget must be a positive integer")
    values = {
        "protocol_version": PREFIX_PROTOCOL_VERSION,
        "reference_budget": reference_budget,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "chat_template_fingerprint": chat_template_fingerprint,
        "verifier_fingerprint": verifier_fingerprint,
    }
    for name in (
        "tokenizer_fingerprint",
        "chat_template_fingerprint",
        "verifier_fingerprint",
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ValueError(f"{name} must be a nonempty string")
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(b"obcf-prefix-protocol-fingerprint-v1\0")
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class EventEquivalenceReport:
    row_count: int
    exact_match_count: int
    mismatch_count: int
    historical_false_recomputed_true_count: int
    historical_true_recomputed_false_count: int
    historical_error_count: int
    recomputed_error_count: int
    passed: bool


def _identity(row: Mapping[str, Any]) -> tuple[str, int, int]:
    try:
        model_id = row["model_id"]
        prompt_id = row["prompt_id"]
        rollout_index = row["rollout_index"]
    except KeyError as error:
        raise ValueError(f"event row is missing identity field {error.args[0]}") from error
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    for name, value in (("prompt_id", prompt_id), ("rollout_index", rollout_index)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    return model_id, prompt_id, rollout_index


def _index(
    rows: Iterable[Mapping[str, Any]],
    name: str,
) -> dict[tuple[str, int, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in rows:
        identity = _identity(row)
        if identity in indexed:
            raise ValueError(f"duplicate {name} event identity {identity}")
        indexed[identity] = row
    return indexed


def _error_present(row: Mapping[str, Any], field: str) -> bool:
    if field not in row:
        raise ValueError(f"event row is missing {field}")
    value = row[field]
    return value is not None and value != ""


def _binary_event(row: Mapping[str, Any], field: str, *, allow_error: bool) -> bool | None:
    if field not in row:
        raise ValueError(f"event row is missing {field}")
    value = row[field]
    if allow_error and value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean when scoring succeeds")
    return value


def compare_prefix_events(
    *,
    historical_rows: Iterable[Mapping[str, Any]],
    recomputed_rows: Iterable[Mapping[str, Any]],
    reference_budget: int,
) -> EventEquivalenceReport:
    """Compare exact binary prefix events after strict artifact identity validation."""
    if (
        not isinstance(reference_budget, int)
        or isinstance(reference_budget, bool)
        or reference_budget <= 0
    ):
        raise ValueError("reference_budget must be a positive integer")
    historical = _index(historical_rows, "historical")
    recomputed = _index(recomputed_rows, "recomputed")
    if historical.keys() != recomputed.keys():
        missing = sorted(historical.keys() - recomputed.keys())
        extra = sorted(recomputed.keys() - historical.keys())
        raise ValueError(
            f"historical and recomputed identities differ; missing={missing}, extra={extra}"
        )
    reward_field = f"prefix_reward_{reference_budget}"
    error_field = f"prefix_error_{reference_budget}"
    exact = false_true = true_false = historical_errors = recomputed_errors = 0
    for identity in sorted(historical):
        old = historical[identity]
        new = recomputed[identity]
        for field in _MATCH_FIELDS:
            if field not in old or field not in new:
                raise ValueError(f"event identity is missing required match field {field}")
            if old[field] != new[field]:
                raise ValueError(f"event identity {identity} has {field} mismatch")
        old_error = _error_present(old, error_field)
        new_error = _error_present(new, error_field)
        historical_errors += int(old_error)
        recomputed_errors += int(new_error)
        old_value = _binary_event(old, reward_field, allow_error=old_error)
        new_value = _binary_event(new, reward_field, allow_error=new_error)
        if old_error or new_error:
            continue
        if old_value == new_value:
            exact += 1
        elif old_value is False and new_value is True:
            false_true += 1
        elif old_value is True and new_value is False:
            true_false += 1
        else:
            raise AssertionError("binary event comparison reached an impossible state")
    mismatch_count = false_true + true_false
    row_count = len(historical)
    return EventEquivalenceReport(
        row_count=row_count,
        exact_match_count=exact,
        mismatch_count=mismatch_count,
        historical_false_recomputed_true_count=false_true,
        historical_true_recomputed_false_count=true_false,
        historical_error_count=historical_errors,
        recomputed_error_count=recomputed_errors,
        passed=(
            mismatch_count == 0
            and historical_errors == 0
            and recomputed_errors == 0
            and exact == row_count
        ),
    )
