"""Reference-compatible teacher-forcing sequence log probabilities."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def sequence_logprobs(
    model,
    examples: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    pad_token_id: int,
    temperature: float,
    device: torch.device | str,
) -> list[float]:
    """Score response tokens under a causal LM with FP32 log-softmax/summation."""
    if not examples:
        return []
    if temperature <= 0.0:
        raise ValueError("logprob temperature must be positive")
    normalized = [([int(x) for x in prompt], [int(x) for x in response]) for prompt, response in examples]
    if any(not prompt or not response for prompt, response in normalized):
        raise ValueError("teacher-forcing examples require nonempty prompt and response tokens")
    lengths = [len(prompt) + len(response) for prompt, response in normalized]
    width = max(lengths)
    input_ids = torch.full((len(normalized), width), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(normalized), width), dtype=torch.long, device=device)
    for row, (prompt, response) in enumerate(normalized):
        tokens = prompt + response
        input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention_mask[row, : len(tokens)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        log_probs = torch.log_softmax(logits.float() / float(temperature), dim=-1)
    output: list[float] = []
    for row, (prompt, response) in enumerate(normalized):
        positions = torch.arange(len(prompt) - 1, len(prompt) + len(response) - 1, device=device)
        targets = torch.tensor(response, dtype=torch.long, device=device)
        value = log_probs[row, positions, targets].sum(dtype=torch.float32)
        if not bool(torch.isfinite(value).item()):
            raise ValueError("teacher-forcing produced a non-finite sequence log probability")
        output.append(float(value.item()))
    return output


def load_reference_model(model_path: str, tokenizer_path: str | None = None, *, device: str = "auto"):
    """Load an offline-only Hugging Face causal LM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define a pad or EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, tokenizer, torch.device(device)
