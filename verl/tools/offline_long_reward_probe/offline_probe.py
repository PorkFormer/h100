#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import datetime as dt
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MISSING = object()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if args.num_prompts is not None:
        config.setdefault("data", {})["num_prompts"] = args.num_prompts
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
    return config


def step_name(step: int) -> str:
    return f"step_{step:04d}"


def output_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_dir"]).expanduser()


def stage_dir(config: dict[str, Any], stage: str, step: int) -> Path:
    return output_dir(config) / stage / step_name(step)


def normalize_compression(compression: Any) -> str | None:
    if compression is None:
        return None
    compression = str(compression)
    if compression.lower() in {"", "none", "null", "false"}:
        return None
    return compression


def write_parquet(df: pd.DataFrame, path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = normalize_compression(config.get("storage", {}).get("compression"))
    df.to_parquet(path, index=False, compression=compression)


def resolve_checkpoint_path(config: dict[str, Any], step: int, cli_checkpoint_path: str | None = None) -> str:
    if cli_checkpoint_path:
        return cli_checkpoint_path

    explicit_paths = config.get("checkpoints", {}).get("explicit_paths", {}) or {}
    if step in explicit_paths:
        return str(explicit_paths[step])
    if str(step) in explicit_paths:
        return str(explicit_paths[str(step)])

    paths = config["paths"]
    template = paths["checkpoint_template"]
    return template.format(checkpoint_root=paths["checkpoint_root"], step=step)


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


def row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    raise TypeError(f"Unsupported row type: {type(row)!r}")


def get_dotted_value(row: Any, dotted_path: str) -> Any:
    current = row_to_dict(row)
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return MISSING
    return current


def extract_ground_truth(row: Any, extractors: list[str], prompt_id: int) -> Any:
    row_dict = row_to_dict(row)
    for extractor in extractors:
        value = get_dotted_value(row_dict, extractor)
        if value is not MISSING and not is_missing_value(value):
            return value
    columns = sorted(row_dict.keys())
    raise KeyError(f"Could not find ground truth for prompt_id={prompt_id}; columns={columns}; tried={extractors}")


def normalize_extra_info(value: Any) -> Any:
    if value is MISSING or is_missing_value(value):
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return value


def normalize_prompt_value(prompt: Any) -> Any:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    elif hasattr(prompt, "as_py"):
        prompt = prompt.as_py()
    elif hasattr(prompt, "to_pylist"):
        prompt = prompt.to_pylist()

    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
        except (TypeError, json.JSONDecodeError):
            return prompt
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
        return prompt

    return prompt


def render_prompt(row: Any, tokenizer: Any, prompt_key: str) -> str:
    row_dict = row_to_dict(row)
    if prompt_key not in row_dict:
        raise KeyError(f"Missing prompt key {prompt_key!r}; columns={sorted(row_dict.keys())}")

    prompt = normalize_prompt_value(row_dict[prompt_key])
    if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt, str):
        return prompt

    raise TypeError(
        f"Unsupported prompt schema for key {prompt_key!r}: type={type(prompt).__name__}; "
        f"columns={sorted(row_dict.keys())}"
    )


def tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return list(encoded["input_ids"])


def load_tokenizer(config: dict[str, Any], checkpoint_path: str) -> tuple[Any, str]:
    from transformers import AutoTokenizer

    paths = config["paths"]
    trust_remote_code = bool(config.get("rollout", {}).get("trust_remote_code", False))
    candidates: list[str] = []

    tokenizer_path = paths.get("tokenizer_path")
    if tokenizer_path:
        candidates.append(str(tokenizer_path))
    candidates.append(str(checkpoint_path))
    if paths.get("base_model_path"):
        candidates.append(str(paths["base_model_path"]))

    seen: set[str] = set()
    errors: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=trust_remote_code)
            return tokenizer, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Failed to load tokenizer from candidates:\n" + "\n".join(errors))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_path_for(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__file__", None)


def torch_runtime_metadata() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "torch_cuda_is_available": False,
            "torch_cuda_device_count": 0,
            "torch_cuda_device_names": [],
        }

    cuda = getattr(torch, "cuda", None)
    try:
        cuda_is_available = bool(cuda.is_available()) if cuda is not None else False
    except Exception:
        cuda_is_available = False

    try:
        cuda_device_count = int(cuda.device_count()) if cuda is not None else 0
    except Exception:
        cuda_device_count = 0

    cuda_device_names = []
    if cuda is not None:
        for device_index in range(cuda_device_count):
            try:
                cuda_device_names.append(cuda.get_device_name(device_index))
            except Exception as exc:
                cuda_device_names.append(f"<error: {exc}>")

    return {
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "torch_cuda_is_available": cuda_is_available,
        "torch_cuda_device_count": cuda_device_count,
        "torch_cuda_device_names": cuda_device_names,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    return value


def metadata(
    config: dict[str, Any],
    checkpoint_path: str,
    tokenizer_path: str | None,
    stage: str,
    step: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = {
        "stage": stage,
        "step": step,
        "config": config,
        "checkpoint_path": checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "python_version": sys.version,
        "vllm_version": package_version("vllm"),
        "transformers_version": package_version("transformers"),
        "verl_import_path": import_path_for("verl"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    data.update(torch_runtime_metadata())
    if extra:
        data.update(extra)
    return jsonable(data)


def write_metadata(
    config: dict[str, Any],
    checkpoint_path: str,
    tokenizer_path: str | None,
    stage: str,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    directory = stage_dir(config, stage, step)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata(config, checkpoint_path, tokenizer_path, stage, step, extra), f, indent=2, sort_keys=True)
        f.write("\n")


def load_reward_function(config: dict[str, Any]) -> Callable[..., Any]:
    reward_config = config.get("scoring", {}).get("reward_function", {})
    file_path = reward_config.get("file_path")
    function_name = reward_config.get("function_name")

    if file_path:
        if not function_name:
            raise ValueError("reward_function.function_name is required when file_path is set")
        spec = importlib.util.spec_from_file_location("offline_probe_reward_function", file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load reward function file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, function_name)

    import_path = reward_config.get("import_path", "verl.utils.reward_score.default_compute_score")
    module_name, attr_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def reward_to_float(value: Any) -> float:
    if isinstance(value, dict):
        if "score" in value:
            return float(value["score"])
        if "reward" in value:
            return float(value["reward"])
        raise KeyError(f"Reward dict must contain 'score' or 'reward': {value}")
    if isinstance(value, (int, float, bool)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    return float(value)


def call_reward_function(
    fn: Callable[..., Any],
    data_source: str,
    text: str,
    ground_truth: Any,
    extra_info: Any | None = None,
) -> float:
    extra_info = normalize_extra_info(extra_info)
    attempts = [
        lambda: fn(data_source=data_source, solution_str=text, ground_truth=ground_truth, extra_info=extra_info),
        lambda: fn(data_source, text, ground_truth, extra_info),
        lambda: fn(data_source, text, ground_truth),
        lambda: fn(text, ground_truth),
    ]
    errors: list[TypeError] = []
    for attempt in attempts:
        try:
            return reward_to_float(attempt())
        except TypeError as exc:
            errors.append(exc)
    raise errors[-1]


def parse_token_ids(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [int(token_id) for token_id in value]
    if isinstance(value, tuple):
        return [int(token_id) for token_id in value]
    if hasattr(value, "tolist"):
        return [int(token_id) for token_id in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        return parse_token_ids(parsed)
    raise TypeError(f"Unsupported token id value type: {type(value)!r}")


def comparison_lengths(prefix_lengths: list[int]) -> tuple[int, int]:
    lengths = [int(length) for length in prefix_lengths]
    if 2048 in lengths and 10240 in lengths:
        return 2048, 10240
    return min(lengths), max(lengths)


def validate_requested_prefixes(prefix_lengths: list[int]) -> None:
    lengths = {int(length) for length in prefix_lengths}
    missing = [length for length in (2048, 10240) if length not in lengths]
    if missing:
        raise ValueError(f"prefix_lengths must include 2048 and 10240; missing={missing}")


def rollout_stop_flags(response_token_len: int, finish_reason: Any, max_tokens: int) -> dict[str, bool]:
    hit_cap = finish_reason == "length" or response_token_len >= max_tokens
    return {
        "is_over_2048": response_token_len > 2048,
        "is_over_4096": response_token_len > 4096,
        "is_over_8192": response_token_len > 8192,
        "hit_10240_cap": hit_cap,
        "eos_or_stop_before_2048": response_token_len < 2048 and finish_reason == "stop",
    }


def stop_category(row: pd.Series) -> str:
    response_token_len = int(row["response_token_len"])
    finish_reason = row.get("finish_reason")
    hit_10240_cap = bool(row.get("hit_10240_cap", row.get("hit_max_tokens", False)))
    if hit_10240_cap:
        return "hit_10240_cap"
    if finish_reason == "stop" and response_token_len < 2048:
        return "eos_or_stop_before_2048"
    if finish_reason == "stop" and 2048 < response_token_len < 10240:
        return "stop_after_2048_before_10240"
    return "other_stop_reason"


def score_dataframe(
    df: pd.DataFrame,
    tokenizer: Any,
    reward_fn: Callable[..., Any],
    prefix_lengths: list[int],
    positive_threshold: float,
    decode_skip_special_tokens: bool,
) -> pd.DataFrame:
    prefix_lengths = [int(length) for length in prefix_lengths]
    first_len, full_len = comparison_lengths(prefix_lengths)
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        out = row.to_dict()
        token_ids = parse_token_ids(out.get("response_token_ids"))
        data_source = out.get("data_source")
        ground_truth = out.get("ground_truth")
        extra_info = normalize_extra_info(out.get("extra_info", {}))
        out["extra_info"] = extra_info
        first_positive_prefix_len = None
        prefix_texts: dict[int, str] = {}

        for prefix_len in prefix_lengths:
            prefix_ids = token_ids[:prefix_len]
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=decode_skip_special_tokens)
            prefix_texts[prefix_len] = prefix_text
            reward = call_reward_function(reward_fn, data_source, prefix_text, ground_truth, extra_info)
            out[f"reward_{prefix_len}"] = reward
            if first_positive_prefix_len is None and reward > positive_threshold:
                first_positive_prefix_len = prefix_len

        first_reward = float(out[f"reward_{first_len}"])
        full_reward = float(out[f"reward_{full_len}"])
        out["first_positive_prefix_len"] = first_positive_prefix_len
        out[f"reward_delta_{full_len}_minus_{first_len}"] = full_reward - first_reward
        out["reward_delta_10240_minus_2048"] = full_reward - first_reward
        out["false_negative_2048_to_10240"] = first_reward <= positive_threshold and full_reward > positive_threshold
        out["answer_corruption_2048_to_10240"] = first_reward > positive_threshold and full_reward <= positive_threshold
        out["response_prefix_2048"] = prefix_texts[first_len]
        out["response_full"] = out.get("response_text") or tokenizer.decode(
            token_ids, skip_special_tokens=decode_skip_special_tokens
        )
        rows.append(out)

    return pd.DataFrame(rows)


def state_from_num_positive(num_positive: int, num_samples: int) -> str:
    if num_positive <= 0:
        return "all_zero"
    if num_positive >= num_samples:
        return "all_positive"
    return "mixed"


def bool_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.astype(bool).mean())


def positive_rate(series: pd.Series, positive_threshold: float) -> float:
    if len(series) == 0:
        return 0.0
    return float((pd.to_numeric(series) > positive_threshold).mean())


def build_analysis_tables(
    scored: pd.DataFrame,
    prefix_lengths: list[int],
    positive_threshold: float,
    eps: float = 1e-8,
) -> dict[str, pd.DataFrame]:
    df = scored.copy()
    prefix_lengths = [int(length) for length in prefix_lengths]

    for column in ["reward_2048", "reward_10240", "reward_delta_10240_minus_2048", "response_token_len"]:
        df[column] = pd.to_numeric(df[column])
    df["hit_max_tokens"] = df.get("hit_max_tokens", False).astype(bool)
    if "is_over_2048" not in df:
        df["is_over_2048"] = df["response_token_len"] > 2048
    if "is_over_4096" not in df:
        df["is_over_4096"] = df["response_token_len"] > 4096
    if "is_over_8192" not in df:
        df["is_over_8192"] = df["response_token_len"] > 8192
    if "hit_10240_cap" not in df:
        df["hit_10240_cap"] = df["hit_max_tokens"]
    if "finish_reason" not in df:
        df["finish_reason"] = None
    if "stop_reason" not in df:
        df["stop_reason"] = None
    if "eos_or_stop_before_2048" not in df:
        df["eos_or_stop_before_2048"] = (df["response_token_len"] < 2048) & df["finish_reason"].eq("stop")
    df["stop_category"] = df.apply(stop_category, axis=1)
    false_negative_default = df["reward_2048"].le(positive_threshold) & df["reward_10240"].gt(positive_threshold)
    answer_corruption_default = df["reward_2048"].gt(positive_threshold) & df["reward_10240"].le(positive_threshold)
    df["false_negative_2048_to_10240"] = df.get(
        "false_negative_2048_to_10240", false_negative_default
    ).astype(bool)
    df["answer_corruption_2048_to_10240"] = df.get(
        "answer_corruption_2048_to_10240", answer_corruption_default
    ).astype(bool)

    step = int(df["step"].iloc[0]) if len(df) else -1
    over2048 = df["is_over_2048"]
    p_false_negative_given_over2048 = (
        bool_mean(df.loc[over2048, "false_negative_2048_to_10240"]) if bool(over2048.any()) else 0.0
    )
    trajectory_summary = pd.DataFrame(
        [
            {
                "step": step,
                "num_trajectories": int(len(df)),
                "num_prompts": int(df["prompt_id"].nunique()) if len(df) else 0,
                "mean_response_len": float(df["response_token_len"].mean()) if len(df) else 0.0,
                "median_response_len": float(df["response_token_len"].median()) if len(df) else 0.0,
                "p_len_gt_2048": bool_mean(df["response_token_len"] > 2048),
                "p_len_gt_4096": bool_mean(df["response_token_len"] > 4096),
                "p_len_gt_8192": bool_mean(df["response_token_len"] > 8192),
                "p_hit_10240_cap": bool_mean(df["hit_10240_cap"]),
                "p_eos_or_stop_before_2048": bool_mean(df["eos_or_stop_before_2048"]),
                "p_stop_after_2048_before_10240": bool_mean(
                    df["stop_category"].eq("stop_after_2048_before_10240")
                ),
                "p_other_stop_reason": bool_mean(df["stop_category"].eq("other_stop_reason")),
                "mean_reward_2048": float(df["reward_2048"].mean()) if len(df) else 0.0,
                "mean_reward_10240": float(df["reward_10240"].mean()) if len(df) else 0.0,
                "p_reward_2048_pos": positive_rate(df["reward_2048"], positive_threshold),
                "p_reward_10240_pos": positive_rate(df["reward_10240"], positive_threshold),
                "p_false_negative": bool_mean(df["false_negative_2048_to_10240"]),
                "p_false_negative_given_over2048": p_false_negative_given_over2048,
                "p_answer_corruption": bool_mean(df["answer_corruption_2048_to_10240"]),
                "mean_reward_delta_10240_minus_2048": float(df["reward_delta_10240_minus_2048"].mean())
                if len(df)
                else 0.0,
            }
        ]
    )

    curve_rows = []
    for prefix_len in prefix_lengths:
        column = f"reward_{prefix_len}"
        if column not in df:
            continue
        curve_rows.append(
            {
                "step": step,
                "prefix_len": prefix_len,
                "mean_reward": float(pd.to_numeric(df[column]).mean()) if len(df) else 0.0,
                "p_reward_pos": positive_rate(df[column], positive_threshold),
            }
        )
    prefix_reward_curve = pd.DataFrame(curve_rows)

    group_rows = []
    for prompt_id, group in df.groupby("prompt_id", sort=True):
        r2048 = pd.to_numeric(group["reward_2048"])
        r10240 = pd.to_numeric(group["reward_10240"])
        num_samples = int(len(group))
        num_pos_2048 = int((r2048 > positive_threshold).sum())
        num_pos_10240 = int((r10240 > positive_threshold).sum())
        group_rows.append(
            {
                "step": step,
                "prompt_id": prompt_id,
                "num_samples": num_samples,
                "num_pos_2048": num_pos_2048,
                "num_pos_10240": num_pos_10240,
                "mean_reward_2048": float(r2048.mean()),
                "mean_reward_10240": float(r10240.mean()),
                "std_reward_2048": float(r2048.std(ddof=0)),
                "std_reward_10240": float(r10240.std(ddof=0)),
                "state_2048": state_from_num_positive(num_pos_2048, num_samples),
                "state_10240": state_from_num_positive(num_pos_10240, num_samples),
            }
        )
    group_summary = pd.DataFrame(group_rows)

    if len(group_summary):
        transition = (
            group_summary.groupby(["state_2048", "state_10240"], sort=True)
            .size()
            .reset_index(name="count")
        )
        transition["step"] = step
        transition["ratio"] = transition["count"] / float(len(group_summary))
        group_transitions = transition[["step", "state_2048", "state_10240", "count", "ratio"]]
    else:
        group_transitions = pd.DataFrame(columns=["step", "state_2048", "state_10240", "count", "ratio"])

    adv_rows = []
    for _, group in df.groupby("prompt_id", sort=True):
        r2048 = pd.to_numeric(group["reward_2048"]).to_numpy(dtype=float)
        r10240 = pd.to_numeric(group["reward_10240"]).to_numpy(dtype=float)
        std2048 = float(r2048.std())
        std10240 = float(r10240.std())
        a2048 = [0.0 for _ in r2048] if std2048 < eps else ((r2048 - r2048.mean()) / (std2048 + eps)).tolist()
        a10240 = [0.0 for _ in r10240] if std10240 < eps else ((r10240 - r10240.mean()) / (std10240 + eps)).tolist()
        adv_rows.extend({"a2048": a, "a10240": b} for a, b in zip(a2048, a10240, strict=False))

    if adv_rows:
        adv_df = pd.DataFrame(adv_rows)
        neg_to_pos = (adv_df["a2048"] < 0) & (adv_df["a10240"] > 0)
        pos_to_neg = (adv_df["a2048"] > 0) & (adv_df["a10240"] < 0)
        zero_to_pos = (adv_df["a2048"] == 0) & (adv_df["a10240"] > 0)
        nonpos_to_pos = (adv_df["a2048"] <= 0) & (adv_df["a10240"] > 0)
        pos_to_nonpos = (adv_df["a2048"] > 0) & (adv_df["a10240"] <= 0)
        sign_flip = neg_to_pos | pos_to_neg
        advantage_summary = pd.DataFrame(
            [
                {
                    "step": step,
                    "adv_sign_flip_rate": float(sign_flip.mean()),
                    "adv_neg_to_pos_rate": float(neg_to_pos.mean()),
                    "adv_pos_to_neg_rate": float(pos_to_neg.mean()),
                    "adv_zero_to_pos_rate": float(zero_to_pos.mean()),
                    "adv_nonpos_to_pos_rate": float(nonpos_to_pos.mean()),
                    "adv_pos_to_nonpos_rate": float(pos_to_nonpos.mean()),
                    "zero_to_pos_rate": float(zero_to_pos.mean()),
                    "nonpos_to_pos_rate": float(nonpos_to_pos.mean()),
                    "pos_to_nonpos_rate": float(pos_to_nonpos.mean()),
                    "mean_abs_adv_delta": float((adv_df["a10240"] - adv_df["a2048"]).abs().mean()),
                }
            ]
        )
    else:
        advantage_summary = pd.DataFrame(
            [
                {
                    "step": step,
                    "adv_sign_flip_rate": 0.0,
                    "adv_neg_to_pos_rate": 0.0,
                    "adv_pos_to_neg_rate": 0.0,
                    "adv_zero_to_pos_rate": 0.0,
                    "adv_nonpos_to_pos_rate": 0.0,
                    "adv_pos_to_nonpos_rate": 0.0,
                    "zero_to_pos_rate": 0.0,
                    "nonpos_to_pos_rate": 0.0,
                    "pos_to_nonpos_rate": 0.0,
                    "mean_abs_adv_delta": 0.0,
                }
            ]
        )

    if len(df):
        stop_reason_summary = (
            df.assign(
                finish_reason=df["finish_reason"].fillna("<none>").astype(str),
                stop_reason=df["stop_reason"].fillna("<none>").astype(str),
            )
            .groupby(["finish_reason", "stop_reason", "stop_category"], sort=True)
            .size()
            .reset_index(name="count")
        )
        stop_reason_summary["step"] = step
        stop_reason_summary["ratio"] = stop_reason_summary["count"] / float(len(df))
        stop_reason_summary = stop_reason_summary[
            ["step", "finish_reason", "stop_reason", "stop_category", "count", "ratio"]
        ]
    else:
        stop_reason_summary = pd.DataFrame(
            columns=["step", "finish_reason", "stop_reason", "stop_category", "count", "ratio"]
        )

    return {
        "trajectory_summary": trajectory_summary,
        "prefix_reward_curve": prefix_reward_curve,
        "group_summary": group_summary,
        "group_transitions": group_transitions,
        "advantage_summary": advantage_summary,
        "stop_reason_summary": stop_reason_summary,
    }


def build_rollout_input_record(
    row_dict: dict[str, Any],
    original_index: Any,
    prompt_id: int,
    tokenizer: Any,
    prompt_key: str,
    data_source_key: str,
    max_prompt_length: int,
    extractors: list[str],
) -> dict[str, Any] | None:
    if data_source_key not in row_dict:
        raise KeyError(f"Missing data_source key {data_source_key!r}; columns={sorted(row_dict.keys())}")

    prompt_text = render_prompt(row_dict, tokenizer, prompt_key)
    prompt_token_len = len(tokenizer_encode(tokenizer, prompt_text))
    if prompt_token_len > max_prompt_length:
        return None

    return {
        "prompt_id": prompt_id,
        "original_index": int(original_index) if isinstance(original_index, int) else original_index,
        "data_source": row_dict[data_source_key],
        "ground_truth": extract_ground_truth(row_dict, extractors, prompt_id),
        "extra_info": normalize_extra_info(row_dict.get("extra_info", {})),
        "prompt_text": prompt_text,
        "prompt_token_len": prompt_token_len,
    }


def prepare_rollout_inputs(config: dict[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_config = config["data"]
    df = pd.read_parquet(config["paths"]["data_path"])
    prompt_start = int(data_config.get("prompt_start", 0))
    num_prompts = int(data_config.get("num_prompts", len(df)))
    end = None if num_prompts <= 0 else prompt_start + num_prompts

    prompt_key = data_config.get("prompt_key", "prompt")
    data_source_key = data_config.get("data_source_key", "data_source")
    max_prompt_length = int(data_config.get("max_prompt_length", 1024))
    extractors = list(data_config.get("ground_truth_extractors", []))
    filter_overlong_before_slice = bool(data_config.get("filter_overlong_before_slice", True))
    filtered_overlong = 0

    if filter_overlong_before_slice:
        filtered_rows: list[dict[str, Any]] = []
        for scan_id, (original_index, row) in enumerate(df.iterrows()):
            record = build_rollout_input_record(
                row.to_dict(),
                original_index,
                scan_id,
                tokenizer,
                prompt_key,
                data_source_key,
                max_prompt_length,
                extractors,
            )
            if record is None:
                filtered_overlong += 1
                continue
            filtered_rows.append(record)

        rows = filtered_rows[prompt_start:end]
        for prompt_id, row in enumerate(rows):
            row["prompt_id"] = prompt_id
            row["prompt_id_dense"] = prompt_id
    else:
        selected = df.iloc[prompt_start:end]
        rows = []
        for local_id, (original_index, row) in enumerate(selected.iterrows()):
            record = build_rollout_input_record(
                row.to_dict(),
                original_index,
                local_id,
                tokenizer,
                prompt_key,
                data_source_key,
                max_prompt_length,
                extractors,
            )
            if record is None:
                filtered_overlong += 1
                continue
            rows.append(record)
        for dense_id, row in enumerate(rows):
            row["prompt_id_dense"] = dense_id

    stats = {
        "dataset_rows": int(len(df)),
        "selected_rows": int(len(rows) if filter_overlong_before_slice else len(df.iloc[prompt_start:end])),
        "filtered_overlong_prompts": int(filtered_overlong),
        "kept_prompts": int(len(rows)),
        "filter_overlong_before_slice": filter_overlong_before_slice,
    }
    return rows, stats


def build_vllm_kwargs(checkpoint_path: str, tokenizer_path: str, rollout_config: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "model": checkpoint_path,
        "tokenizer": tokenizer_path,
        "trust_remote_code": bool(rollout_config.get("trust_remote_code", False)),
        "dtype": rollout_config.get("dtype", "bfloat16"),
        "tensor_parallel_size": int(rollout_config.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": float(rollout_config.get("gpu_memory_utilization", 0.72)),
        "max_model_len": int(rollout_config.get("max_model_len", 12288)),
        "enforce_eager": bool(rollout_config.get("enforce_eager", False)),
        "enable_prefix_caching": bool(rollout_config.get("enable_prefix_caching", True)),
    }
    if "max_num_seqs" in rollout_config and rollout_config["max_num_seqs"] is not None:
        kwargs["max_num_seqs"] = int(rollout_config["max_num_seqs"])
    if "max_num_batched_tokens" in rollout_config and rollout_config["max_num_batched_tokens"] is not None:
        kwargs["max_num_batched_tokens"] = int(rollout_config["max_num_batched_tokens"])
    return kwargs


def validate_rollout_storage_config(config: dict[str, Any]) -> None:
    if not bool(config.get("storage", {}).get("save_response_token_ids", True)):
        raise ValueError("score mode requires response_token_ids; do not disable save_response_token_ids")


def run_rollout(config: dict[str, Any], step: int, cli_checkpoint_path: str | None) -> None:
    checkpoint_path = resolve_checkpoint_path(config, step, cli_checkpoint_path)
    validate_rollout_storage_config(config)
    rollout_dir = stage_dir(config, "rollouts", step)
    output_path = rollout_dir / "rollouts.parquet"
    skip_existing = bool(config.get("storage", {}).get("skip_existing", False))
    if skip_existing and output_path.exists():
        write_metadata(config, checkpoint_path, None, "rollouts", step, {"skipped": True, "reason": "output exists"})
        print(f"Skipping rollout because output exists: {output_path}")
        return

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    tokenizer, tokenizer_path = load_tokenizer(config, checkpoint_path)
    prompts, prompt_stats = prepare_rollout_inputs(config, tokenizer)

    from vllm import LLM, SamplingParams

    rollout_config = config["rollout"]
    llm = LLM(**build_vllm_kwargs(checkpoint_path, tokenizer_path, rollout_config))
    sampling_params = SamplingParams(
        n=int(rollout_config.get("n", 8)),
        max_tokens=int(rollout_config.get("max_tokens", 10240)),
        temperature=float(rollout_config.get("temperature", 1.0)),
        top_p=float(rollout_config.get("top_p", 1.0)),
        top_k=int(rollout_config.get("top_k", -1)),
        ignore_eos=bool(rollout_config.get("ignore_eos", False)),
    )

    records: list[dict[str, Any]] = []
    batch_size = int(rollout_config.get("batch_size_prompts", 128))
    save_response_text = bool(config.get("storage", {}).get("save_response_text", True))
    save_response_token_ids = bool(config.get("storage", {}).get("save_response_token_ids", True))
    max_tokens = int(rollout_config.get("max_tokens", 10240))

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        outputs = llm.generate([item["prompt_text"] for item in batch], sampling_params)
        for item, request_output in zip(batch, outputs, strict=False):
            completions = getattr(request_output, "outputs", [])
            for sample_id, completion in enumerate(completions):
                token_ids = list(getattr(completion, "token_ids", []) or [])
                finish_reason = getattr(completion, "finish_reason", None)
                stop_reason = getattr(completion, "stop_reason", None)
                response_token_len = len(token_ids)
                stop_flags = rollout_stop_flags(response_token_len, finish_reason, max_tokens)
                records.append(
                    {
                        "step": step,
                        "prompt_id": item["prompt_id"],
                        "prompt_id_dense": item.get("prompt_id_dense", item["prompt_id"]),
                        "original_index": item["original_index"],
                        "sample_id": sample_id,
                        "data_source": item["data_source"],
                        "ground_truth": item["ground_truth"],
                        "extra_info": item["extra_info"],
                        "prompt_text": item["prompt_text"],
                        "prompt_token_len": item["prompt_token_len"],
                        "response_text": getattr(completion, "text", None) if save_response_text else None,
                        "response_token_ids": token_ids if save_response_token_ids else None,
                        "response_token_len": response_token_len,
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "hit_max_tokens": stop_flags["hit_10240_cap"],
                        **stop_flags,
                    }
                )
        print(f"Generated prompts {start + len(batch)}/{len(prompts)}")

    write_parquet(pd.DataFrame(records), output_path, config)
    write_metadata(
        config,
        checkpoint_path,
        tokenizer_path,
        "rollouts",
        step,
        {"skipped": False, "output_path": str(output_path), **prompt_stats, "num_trajectories": len(records)},
    )
    print(f"Wrote rollout parquet: {output_path}")


def run_score(config: dict[str, Any], step: int, cli_checkpoint_path: str | None) -> None:
    checkpoint_path = resolve_checkpoint_path(config, step, cli_checkpoint_path)
    scored_dir = stage_dir(config, "scored", step)
    output_path = scored_dir / "scored.parquet"
    skip_existing = bool(config.get("storage", {}).get("skip_existing", False))
    if skip_existing and output_path.exists():
        write_metadata(config, checkpoint_path, None, "scored", step, {"skipped": True, "reason": "output exists"})
        print(f"Skipping score because output exists: {output_path}")
        return

    prefix_lengths = [int(length) for length in config["scoring"]["prefix_lengths"]]
    validate_requested_prefixes(prefix_lengths)
    rollout_path = stage_dir(config, "rollouts", step) / "rollouts.parquet"
    df = pd.read_parquet(rollout_path)
    if "response_token_ids" not in df:
        raise KeyError("rollouts.parquet is missing response_token_ids; scoring requires token ids")

    tokenizer, tokenizer_path = load_tokenizer(config, checkpoint_path)
    reward_fn = load_reward_function(config)
    scored = score_dataframe(
        df,
        tokenizer,
        reward_fn,
        prefix_lengths,
        float(config["scoring"].get("positive_threshold", 0.0)),
        bool(config["scoring"].get("decode_skip_special_tokens", True)),
    )
    write_parquet(scored, output_path, config)
    write_metadata(
        config,
        checkpoint_path,
        tokenizer_path,
        "scored",
        step,
        {"skipped": False, "input_path": str(rollout_path), "output_path": str(output_path), "num_rows": len(scored)},
    )
    print(f"Wrote scored parquet: {output_path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def example_rows(df: pd.DataFrame, mask: pd.Series) -> list[dict[str, Any]]:
    columns = [
        "step",
        "prompt_id",
        "sample_id",
        "data_source",
        "ground_truth",
        "prompt_text",
        "response_prefix_2048",
        "response_full",
        "reward_2048",
        "reward_10240",
        "first_positive_prefix_len",
        "response_token_len",
    ]
    examples = df.loc[mask].sort_values(["prompt_id", "sample_id"]).head(100)
    rows = []
    for _, row in examples.iterrows():
        row_dict = row.to_dict()
        rows.append({column: row_dict.get(column) for column in columns})
    return rows


def run_analyze(config: dict[str, Any], step: int, cli_checkpoint_path: str | None) -> None:
    checkpoint_path = resolve_checkpoint_path(config, step, cli_checkpoint_path)
    analysis_dir = stage_dir(config, "analysis", step)
    summary_path = analysis_dir / "trajectory_summary.csv"
    skip_existing = bool(config.get("storage", {}).get("skip_existing", False))
    if skip_existing and summary_path.exists():
        write_metadata(config, checkpoint_path, None, "analysis", step, {"skipped": True, "reason": "output exists"})
        print(f"Skipping analysis because output exists: {summary_path}")
        return

    prefix_lengths = [int(length) for length in config["scoring"]["prefix_lengths"]]
    validate_requested_prefixes(prefix_lengths)
    scored_path = stage_dir(config, "scored", step) / "scored.parquet"
    scored = pd.read_parquet(scored_path)
    tables = build_analysis_tables(
        scored,
        prefix_lengths,
        float(config["scoring"].get("positive_threshold", 0.0)),
    )

    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(analysis_dir / f"{name}.csv", index=False)

    examples_dir = output_dir(config) / "examples" / step_name(step)
    write_jsonl(
        examples_dir / "false_negative_examples.jsonl",
        example_rows(scored, scored["false_negative_2048_to_10240"].astype(bool)),
    )
    write_jsonl(
        examples_dir / "answer_corruption_examples.jsonl",
        example_rows(scored, scored["answer_corruption_2048_to_10240"].astype(bool)),
    )
    write_metadata(
        config,
        checkpoint_path,
        None,
        "analysis",
        step,
        {"skipped": False, "input_path": str(scored_path), "output_dir": str(analysis_dir)},
    )
    print(f"Wrote analysis outputs: {analysis_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline fixed-checkpoint long-rollout reward probe")
    parser.add_argument("--config", required=True, help="Path to probe_config.yaml")
    parser.add_argument("--step", required=True, type=int, help="Checkpoint global step")
    parser.add_argument("--mode", required=True, choices=["rollout", "score", "analyze", "all"])
    parser.add_argument("--num-prompts", type=int, default=None, help="Override data.num_prompts")
    parser.add_argument("--output-dir", default=None, help="Override paths.output_dir")
    parser.add_argument("--checkpoint-path", default=None, help="Override checkpoint path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)

    if args.mode in {"rollout", "all"}:
        run_rollout(config, args.step, args.checkpoint_path)
    if args.mode in {"score", "all"}:
        run_score(config, args.step, args.checkpoint_path)
    if args.mode in {"analyze", "all"}:
        run_analyze(config, args.step, args.checkpoint_path)


if __name__ == "__main__":
    main()
