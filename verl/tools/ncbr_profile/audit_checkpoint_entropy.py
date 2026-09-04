#!/usr/bin/env python3
"""Measure full-vocabulary categorical entropy on frozen token histories."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def categorical_entropy(logits):
    """Return entropy from logits without substituting sampled-token surprisal."""
    import torch

    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1)
    probabilities = torch.exp(values - log_z.unsqueeze(-1))
    return log_z - (probabilities * values).sum(dim=-1)


def aggregate(records: list[dict[str, Any]], bucket_size: int = 256, prefix_length: int = 2048) -> dict[str, Any]:
    if bucket_size <= 0 or prefix_length <= 0:
        raise ValueError("entropy bucket and prefix lengths must be positive")
    groups: dict[tuple[str, str, str, str], list[tuple[str, list[float]]]] = defaultdict(list)
    for record in records:
        benchmark = str(record["benchmark"])
        if benchmark not in {"AIME2024", "AIME2025"}:
            raise ValueError(f"unsupported or pooled benchmark label: {benchmark}")
        values = [float(value) for value in record["categorical_entropy"]]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("categorical entropy values must be finite and nonempty")
        trajectory = f"{record['prompt_id']}:{int(record['rollout_index'])}"
        cap = "cap" if bool(record.get("cap_hit", len(values) >= prefix_length)) else "non_cap"
        transition = str(record.get("answer_transition", "unavailable"))
        for start in range(0, len(values), bucket_size):
            bucket = f"{start:04d}_{start + bucket_size:04d}"
            groups[(benchmark, "bucket", bucket, "all")].append((trajectory, values[start : start + bucket_size]))
        groups[(benchmark, "segment", "prefix", "all")].append((trajectory, values[:prefix_length]))
        if len(values) > prefix_length:
            groups[(benchmark, "segment", "tail", "all")].append((trajectory, values[prefix_length:]))
        groups[(benchmark, "cohort", cap, "all")].append((trajectory, values))
        groups[(benchmark, "answer_transition", transition, "all")].append((trajectory, values))

    output: dict[str, Any] = {"AIME2024": [], "AIME2025": []}
    for (benchmark, dimension, cohort, _), sequences in sorted(groups.items()):
        nonempty = [(trajectory, values) for trajectory, values in sequences if values]
        flat = [value for _, values in nonempty for value in values]
        sequence_means = [sum(values) / len(values) for _, values in nonempty]
        output[benchmark].append(
            {
                "dimension": dimension,
                "cohort": cohort,
                "token_weighted": sum(flat) / len(flat),
                "sequence_balanced": sum(sequence_means) / len(sequence_means),
                "token_count": len(flat),
                "trajectory_count": len(nonempty),
            }
        )
    if any(not output[benchmark] for benchmark in output):
        raise ValueError("AIME2024 and AIME2025 entropy results must both be present")
    return output


def _measure_row(model, row: dict[str, Any], chunk_size: int) -> list[float]:
    import torch

    prompt = [int(token) for token in row["prompt_token_ids"]]
    response = [int(token) for token in row["response_token_ids"]]
    tokens = prompt + response
    if not prompt or not response:
        raise ValueError("entropy rows require nonempty prompt and response tokens")
    device = next(model.parameters()).device
    past = None
    entropies: list[float] = []
    absolute_start = 0
    with torch.inference_mode():
        for start in range(0, len(tokens), chunk_size):
            chunk = torch.tensor([tokens[start : start + chunk_size]], dtype=torch.long, device=device)
            output = model(input_ids=chunk, past_key_values=past, use_cache=True)
            past = output.past_key_values
            values = categorical_entropy(output.logits[0]).detach().cpu().tolist()
            for offset, value in enumerate(values):
                prediction_position = absolute_start + offset
                response_index = prediction_position - (len(prompt) - 1)
                if 0 <= response_index < len(response):
                    entropies.append(float(value))
            absolute_start += chunk.shape[-1]
    if len(entropies) != len(response):
        raise RuntimeError(f"entropy alignment mismatch: expected {len(response)}, got {len(entropies)}")
    return entropies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--panel-kind", choices=("on_policy", "shared_base_teacher_forced"), required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--prefix-length", type=int, default=2048)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("chunk size must be positive")
    if args.details.exists() or args.output.exists():
        raise SystemExit("refusing to overwrite checkpoint entropy artifacts")
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    rows = [json.loads(line) for line in args.panel.read_text(encoding="utf-8").splitlines() if line.strip()]
    measured = []
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("x", encoding="utf-8") as stream:
        for row in rows:
            record = {
                **row,
                "checkpoint": args.checkpoint_label,
                "panel_kind": args.panel_kind,
                "categorical_entropy": _measure_row(model, row, args.chunk_size),
            }
            measured.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
    result = {
        "schema_version": "qwen3-1p7b-checkpoint-categorical-entropy-v1",
        "checkpoint": args.checkpoint_label,
        "panel_kind": args.panel_kind,
        "estimator": "full_vocabulary_categorical_entropy",
        "sampled_token_surprisal_used": False,
        "benchmarks": aggregate(measured, args.bucket_size, args.prefix_length),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
