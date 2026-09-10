"""Original response-cap detector from d23da0e (no forced-answer runtime)."""
from typing import Any, Sequence
import numpy as np

_LENGTH_REASONS = {"length", "max_length", "max_tokens", "token_limit"}
_EOS_REASONS = {"stop", "eos", "eos_token", "end_turn"}
_ABORT_REASONS = {"abort", "aborted", "cancelled", "canceled", "error"}

def _normalize_finish_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    value = getattr(reason, "value", reason)
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    normalized = str(value).strip().lower()
    return normalized or None

def detect_hit_response_cap(
    *,
    finish_reasons: Sequence[Any] | None,
    response_lengths: Sequence[int],
    max_response_length: int,
    response_token_ids: Sequence[Sequence[int]] | None = None,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Distinguish explicit length termination from natural EOS.

    Backend finish metadata takes priority. When it is unavailable or ambiguous,
    a full response without an EOS token is treated as a cap hit.
    """
    if max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    lengths = [int(length) for length in response_lengths]
    if finish_reasons is not None and len(finish_reasons) != len(lengths):
        raise ValueError("finish_reasons and response_lengths must have equal length")
    if response_token_ids is not None and len(response_token_ids) != len(lengths):
        raise ValueError("response_token_ids and response_lengths must have equal length")

    result: list[bool] = []
    for index, length in enumerate(lengths):
        reason = _normalize_finish_reason(finish_reasons[index]) if finish_reasons is not None else None
        if reason in _LENGTH_REASONS:
            result.append(True)
            continue
        if reason in _EOS_REASONS or reason in _ABORT_REASONS:
            result.append(False)
            continue

        has_eos = False
        if response_token_ids is not None and eos_token_id is not None:
            has_eos = int(eos_token_id) in [int(token_id) for token_id in response_token_ids[index][:length]]
        result.append(length >= max_response_length and not has_eos)
    return np.asarray(result, dtype=bool)
