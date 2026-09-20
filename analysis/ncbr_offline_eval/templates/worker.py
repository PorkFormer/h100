#!/usr/bin/env python3
"""One isolated TP=1 vLLM worker for one independently generated horizon."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


from common import ROOT as TASK_ROOT, output_root, read_rows, pending_requests, prompts as task_prompts
ROOT = TASK_ROOT / "formal"
VERL = Path("/workspace/rl/h100-natural-continuation-boundary-return-v1/verl")
TOKENIZER = "/workspace/models/Qwen3-4B-Base"


def stable_sampling_seed(master_seed: int, prompt_id: int, rollout_index: int) -> int:
    payload = f"offline-answer-timing-v1\0{master_seed}\0{prompt_id}\0{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def append_locked(path: Path, row: dict) -> None:
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(payload)
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--horizon", type=int, choices=(2048, 8192), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--attempt-dir", required=True)
    args = parser.parse_args()
    ROOT = Path(args.attempt_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("physical GPU isolation mismatch")

    manifest = json.loads((TASK_ROOT / "checkpoint_manifest.json").read_text())
    matches = [row for row in manifest["checkpoints"] if row["model_id"] == args.model_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown model: {args.model_id}")
    model = matches[0]
    prompts = task_prompts(args.smoke)
    raw_path = ROOT / f"generation/h{args.horizon}/raw_generations.jsonl"
    existing = set()
    if raw_path.exists():
        raise RuntimeError("attempt output already exists")
    requests = []
    for prompt_order, prompt in enumerate(prompts):
        for rollout_index in range(32):
            if (prompt_order * 32 + rollout_index) % args.shards != args.shard:
                continue
            requests.append((prompt, rollout_index,
                             stable_sampling_seed(42, int(prompt["prompt_id"]), rollout_index)))
    requests = pending_requests(requests, existing, args.model_id)
    if not requests:
        print(json.dumps({"gpu": args.gpu, "model_id": args.model_id, "generated": 0, "resumed": True}))
        return

    import vllm
    assert vllm.__version__ == "0.18.0"
    import subprocess
    visible_uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    assert visible_uuid == os.environ["EXPECTED_GPU_UUID"]
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(VERL))
    from verl.utils.reward_score.math_dapo import compute_score

    max_model_len = 4096 if args.horizon == 2048 else 9216
    load_started = time.monotonic()
    llm = LLM(model=model["checkpoint_path"], tokenizer=TOKENIZER, tensor_parallel_size=1,
              dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.72,
              max_num_seqs=8, max_num_batched_tokens=32768, enable_chunked_prefill=True,
              enable_prefix_caching=True, trust_remote_code=False, seed=42, disable_log_stats=True)
    load_seconds = time.monotonic() - load_started
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=False)
    generated = 0
    for start in range(0, len(requests), 8):
        batch = requests[start:start + 8]
        params = [SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
                                 repetition_penalty=1.0, max_tokens=args.horizon,
                                 ignore_eos=False, seed=seed) for _, _, seed in batch]
        batch_started = time.monotonic()
        outputs = llm.generate([{"prompt_token_ids": prompt["prompt_token_ids"]}
                                for prompt, _, _ in batch], params, use_tqdm=False)
        elapsed = time.monotonic() - batch_started
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM output count mismatch")
        for (prompt, rollout_index, seed), request_output in zip(batch, outputs, strict=True):
            if len(request_output.outputs) != 1:
                raise RuntimeError("completion count mismatch")
            completion = request_output.outputs[0]
            token_ids = [int(value) for value in completion.token_ids]
            response_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            scoring_error = None
            try:
                reward = compute_score(solution_str=response_text, ground_truth=prompt["ground_truth"])
                assert isinstance(reward, dict) and "acc" in reward and "score" in reward
            except Exception as exc:
                scoring_error = repr(exc)
                reward = {"acc": None, "score": None}
            benchmark = prompt["benchmark"]
            if (benchmark, prompt["data_source"]) not in (("AMC23", "amc23"), ("AIME2024", "aime2024"), ("AIME2025", "aime2025")):
                raise ValueError("unexpected benchmark")
            pair_key = f"{args.model_id}|{int(prompt['prompt_id'])}|{rollout_index}"
            row = {
                "trajectory_id": f"{pair_key}|h{args.horizon}", "pair_key": pair_key,
                "request_id": f"offline-{hashlib.sha256((pair_key + f'|h{args.horizon}').encode()).hexdigest()[:24]}",
                "horizon": args.horizon, "model_id": args.model_id, "arm": model["arm"],
                "step": int(model["step"]), "checkpoint_path": model["checkpoint_path"],
                "checkpoint_sha256": model["sha256"], "prompt_id": int(prompt["prompt_id"]),
                "prompt_hash": prompt["prompt_hash"], "benchmark": benchmark,
                "original_dataset_index": int(prompt["original_dataset_index"]),
                "data_source": prompt["data_source"], "ground_truth": prompt["ground_truth"],
                "extra_info_json": prompt["extra_info_json"], "rollout_index": rollout_index,
                "stable_seed": seed, "prompt_token_ids": [int(value) for value in prompt["prompt_token_ids"]],
                "response_token_ids": token_ids, "response_text": response_text,
                "engine_response_text": str(completion.text), "response_token_count": len(token_ids),
                "scoring_error": scoring_error, "gpu_uuid": visible_uuid,
                "reward_acc": None if scoring_error else bool(reward["acc"]), "reward_score": None if scoring_error else float(reward["score"]),
                "reward_pred": None if reward.get("pred") is None else str(reward["pred"]),
                "finish_reason": None if completion.finish_reason is None else str(completion.finish_reason),
                "stop_reason": None if completion.stop_reason is None else str(completion.stop_reason),
                "hit_cap": completion.finish_reason == "length" or len(token_ids) >= args.horizon,
                "gpu": args.gpu, "worker_shard": args.shard, "batch_generation_seconds": elapsed,
                "model_load_seconds": load_seconds, "written_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_locked(raw_path, row)
            generated += 1
        print(json.dumps({"model_id": args.model_id, "horizon": args.horizon,
                          "gpu": args.gpu, "completed": generated, "assigned": len(requests)}), flush=True)
    print(json.dumps({"model_id": args.model_id, "horizon": args.horizon,
                      "gpu": args.gpu, "generated": generated, "passed": True}))


if __name__ == "__main__":
    main()
