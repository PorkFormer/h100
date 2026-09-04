#!/usr/bin/env python3
"""Generate deterministic 32-way H8192 AIME rollouts for entropy panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _score(data_source, text, reward_model, extra_info):
    from verl.utils.reward_score import default_compute_score

    value = default_compute_score(
        data_source=data_source,
        solution_str=text,
        ground_truth=reward_model["ground_truth"],
        extra_info=extra_info,
    )
    if isinstance(value, dict):
        value = value.get("acc", value.get("score", value.get("reward")))
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--aime2024", type=Path, required=True)
    parser.add_argument("--aime2025", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite entropy rollout JSONL")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    prompts = []
    for benchmark, path in (("AIME2024", args.aime2024), ("AIME2025", args.aime2025)):
        unique = {}
        for row in pq.read_table(path).to_pylist():
            index = int(row["extra_info"]["index"])
            unique.setdefault(index, row)
        if len(unique) != 30:
            raise SystemExit(f"{benchmark} must contain exactly 30 unique questions, got {len(unique)}")
        for index, row in sorted(unique.items()):
            token_ids = tokenizer.apply_chat_template(
                row["prompt"],
                add_generation_prompt=True,
                tokenize=True,
            )
            if len(token_ids) > 1024:
                raise SystemExit(f"overlong entropy prompt after chat template: {benchmark}:{index}:{len(token_ids)}")
            prompts.append(
                {
                    "benchmark": benchmark,
                    "prompt_id": f"{benchmark}:{index}",
                    "prompt_token_ids": [int(token) for token in token_ids],
                    "data_source": row["data_source"],
                    "reward_model": row["reward_model"],
                    "extra_info": row["extra_info"],
                }
            )
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=9216,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_chunked_prefill=True,
        enforce_eager=False,
        seed=args.seed,
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        n=32,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=8192,
        ignore_eos=False,
        seed=args.seed,
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=row["prompt_token_ids"]) for row in prompts],
        sampling_params=sampling,
        use_tqdm=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for prompt, request_output in zip(prompts, outputs, strict=True):
            if len(request_output.outputs) != 32:
                raise RuntimeError(f"entropy rollout count mismatch for {prompt['prompt_id']}")
            for output in sorted(request_output.outputs, key=lambda item: item.index):
                tokens = [int(token) for token in output.token_ids]
                finish_reason = str(getattr(output, "finish_reason", "")).lower().split(".")[-1]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                correctness = _score(
                    prompt["data_source"],
                    text,
                    prompt["reward_model"],
                    prompt["extra_info"],
                )
                stream.write(
                    json.dumps(
                        {
                            "benchmark": prompt["benchmark"],
                            "prompt_id": prompt["prompt_id"],
                            "rollout_index": int(output.index),
                            "prompt_token_ids": prompt["prompt_token_ids"],
                            "response_token_ids": tokens,
                            "finish_reason": finish_reason,
                            "cap_hit": finish_reason == "length" and len(tokens) == 8192,
                            "correctness": correctness,
                            "checkpoint": args.checkpoint_label,
                            "seed": args.seed,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()


if __name__ == "__main__":
    main()
