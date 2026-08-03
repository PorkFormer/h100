"""Stable token and local model-weight identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_prompt_key(
    tokenizer_fingerprint: str,
    chat_template_fingerprint: str,
    prompt_token_ids: Iterable[int],
) -> str:
    """Hash rendered prompt tokens independently of mutable dataset row indices."""
    payload = {
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "chat_template_fingerprint": chat_template_fingerprint,
        "prompt_token_ids": [int(token) for token in prompt_token_ids],
    }
    return _sha256_bytes(_canonical_json(payload))


def tokenizer_fingerprints(tokenizer: Any) -> tuple[str, str]:
    """Return deterministic tokenizer-vocabulary and chat-template fingerprints."""
    tokenizer_fingerprint = _sha256_bytes(
        _canonical_json(
            {
                "vocab": tokenizer.get_vocab(),
                "special_tokens": tokenizer.special_tokens_map,
            }
        )
    )
    template_fingerprint = _sha256_bytes(_canonical_json(tokenizer.chat_template or ""))
    return tokenizer_fingerprint, template_fingerprint


def reference_model_fingerprint(path: str | Path) -> str:
    """Hash local Hugging Face weight bytes, independent of checkpoint path."""
    root = Path(path).expanduser()
    if root.is_file():
        weight_files = [root]
        relative_to = root.parent
    elif root.is_dir():
        patterns = (
            "model*.safetensors",
            "pytorch_model*.bin",
            "adapter_model*.safetensors",
            "adapter_model*.bin",
            "consolidated*.pth",
        )
        weight_files = sorted(
            {candidate for pattern in patterns for candidate in root.rglob(pattern)}
        )
        relative_to = root
    else:
        raise ValueError("strict model fingerprinting requires a local model path")
    if not weight_files:
        raise ValueError(f"no supported model weight files found under {root}")

    # Keep the established BSSF digest domain so existing caches remain valid.
    digest = hashlib.sha256()
    digest.update(b"bssf-reference-model-weights-v1\0")
    for weight_file in weight_files:
        relative = weight_file.relative_to(relative_to).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(weight_file.stat().st_size.to_bytes(8, "big"))
        with weight_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()
