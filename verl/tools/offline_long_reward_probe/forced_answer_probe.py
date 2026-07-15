#!/usr/bin/env python3
"""Forced-answer counterfactual probes for offline long rollouts.

Generation consumes the token ids saved by ``offline_probe.py`` and appends a
short answer cue at selected response horizons.  Analysis is deliberately
CPU-only and has no import-time dependency on vLLM or transformers.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.offline_long_reward_probe import offline_probe  # noqa: E402

ERROR_RATE_THRESHOLD = 0.05
TRAJECTORY_KEYS = ["step", "prompt_id", "sample_id"]
REQUIRED_SOURCE_COLUMNS = [
    "step",
    "prompt_id",
    "sample_id",
    "data_source",
    "ground_truth",
    "extra_info",
    "prompt_text",
    "response_token_ids",
    "response_token_len",
    "reward_10240",
]
RAW_COLUMNS = [
    "step",
    "prompt_id",
    "sample_id",
    "original_index",
    "request_id",
    "probe_branch_id",
    "horizon",
    "kind",
    "cue",
    "cue_text",
    "status",
    "error_type",
    "error_message",
    "is_context_overflow",
    "is_carry_forward",
    "data_source",
    "ground_truth",
    "extra_info",
    "prompt_text",
    "prompt_token_ids",
    "source_response_text",
    "response_token_ids",
    "response_token_len",
    "cue_token_ids",
    "input_token_ids",
    "input_token_len",
    "completion_text",
    "completion_token_ids",
    "completion_token_len",
    "finish_reason",
    "stop_reason",
    "terminal_reward",
    "terminal_success",
    "probe_reward",
    "probe_success",
]
TAXONOMY_LABELS = [
    "delayed_solve_candidate",
    "early_recoverable",
    "terminal_only_candidate",
    "unstable_diagnostic",
    "early_recoverable_but_terminal_wrong",
]


def _canonical_ints(values: Any, name: str, *, allow_zero: bool) -> list[int]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"forced_answer.{name} must be a non-empty list")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"forced_answer.{name} values must be integers: {value!r}")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"forced_answer.{name} values must be integers: {value!r}") from exc
        if integer != value or integer < (0 if allow_zero else 1):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"forced_answer.{name} values must be {qualifier} integers: {value!r}")
        result.append(integer)
    return sorted(set(result))


def _config_value(section: dict[str, Any], name: str, default: Any = None) -> Any:
    if name in section:
        return section[name]
    for nested_name in ("generation", "analysis"):
        nested = section.get(nested_name, {})
        if isinstance(nested, dict) and name in nested:
            return nested[name]
    return default


def normalize_forced_answer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the forced-answer configuration."""

    config = copy.deepcopy(config)
    section = config.get("forced_answer")
    if not isinstance(section, dict):
        raise ValueError("Config must contain a forced_answer mapping")

    horizons = _canonical_ints(
        _config_value(section, "horizons", _config_value(section, "fixed_horizons")),
        "horizons",
        allow_zero=True,
    )
    tail_offsets = _canonical_ints(
        _config_value(section, "tail_offsets", _config_value(section, "preterminal_offsets")),
        "tail_offsets",
        allow_zero=False,
    )

    raw_cues = _config_value(section, "cues")
    if not isinstance(raw_cues, list) or not raw_cues:
        raise ValueError("forced_answer.cues must be a non-empty list")
    cues: list[dict[str, str]] = []
    for index, cue in enumerate(raw_cues):
        if isinstance(cue, str):
            name, text = f"cue_{index}", cue
        elif isinstance(cue, dict):
            name = str(cue.get("name", "")).strip()
            text = str(cue.get("text", ""))
        else:
            raise ValueError(f"forced_answer.cues[{index}] must be a string or mapping")
        if not name or not text.strip():
            raise ValueError(f"forced_answer.cues[{index}] must have a non-empty name and text")
        cues.append({"name": name, "text": text})
    names = [cue["name"] for cue in cues]
    texts = [cue["text"] for cue in cues]
    if len(names) != len(set(names)):
        raise ValueError("forced_answer cue names must be unique")
    if len(texts) != len(set(texts)):
        raise ValueError("forced_answer cue texts must be unique")

    n = int(_config_value(section, "n", 1))
    max_tokens = int(_config_value(section, "max_tokens", 256))
    batch_size = int(_config_value(section, "batch_size_requests", 64))
    if n <= 0:
        raise ValueError("forced_answer.n must be positive")
    if max_tokens <= 0:
        raise ValueError("forced_answer.max_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("forced_answer.batch_size_requests must be positive")

    temperature = float(_config_value(section, "temperature", 0.0))
    top_p = float(_config_value(section, "top_p", 1.0))
    stability_threshold = float(_config_value(section, "stability_threshold", 0.5))
    if temperature < 0:
        raise ValueError("forced_answer.temperature must be non-negative")
    if not 0.0 <= top_p <= 1.0:
        raise ValueError("forced_answer.top_p must be in [0, 1]")
    if not 0.0 <= stability_threshold <= 1.0:
        raise ValueError("forced_answer.stability_threshold must be in [0, 1]")

    max_trajectories = _config_value(section, "max_trajectories")
    if max_trajectories is not None:
        max_trajectories = int(max_trajectories)
        if max_trajectories <= 0:
            raise ValueError("forced_answer.max_trajectories must be positive when set")

    canonical = dict(section)
    canonical.update(
        {
            "horizons": horizons,
            "tail_offsets": tail_offsets,
            "cues": cues,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(_config_value(section, "top_k", -1)),
            "ignore_eos": bool(_config_value(section, "ignore_eos", False)),
            "seed": int(_config_value(section, "seed", 42)),
            "batch_size_requests": batch_size,
            "stability_threshold": stability_threshold,
            "eps": float(_config_value(section, "eps", 1e-8)),
            "max_trajectories": max_trajectories,
        }
    )
    if canonical["eps"] <= 0:
        raise ValueError("forced_answer.eps must be positive")
    config["forced_answer"] = canonical
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = copy.deepcopy(config)
    overrides: dict[str, Any] = {}
    if args.output_dir is not None:
        config.setdefault("paths", {})["output_dir"] = args.output_dir
        overrides["output_dir"] = args.output_dir
    if args.source_scored is not None:
        config.setdefault("forced_answer", {})["source_scored"] = args.source_scored
        overrides["source_scored"] = args.source_scored
    if args.max_trajectories is not None:
        config.setdefault("forced_answer", {})["max_trajectories"] = args.max_trajectories
        overrides["max_trajectories"] = args.max_trajectories
    if args.probe_n is not None:
        config.setdefault("forced_answer", {})["n"] = args.probe_n
        overrides["probe_n"] = args.probe_n
    if args.checkpoint_path is not None:
        config.setdefault("forced_answer", {})["checkpoint_path"] = args.checkpoint_path
        overrides["checkpoint_path"] = args.checkpoint_path
    return normalize_forced_answer_config(config), overrides


def forced_root(config: dict[str, Any]) -> Path:
    return offline_probe.output_dir(config) / "forced_answer"


def raw_dir(config: dict[str, Any], step: int) -> Path:
    return forced_root(config) / "raw" / offline_probe.step_name(step)


def analysis_dir(config: dict[str, Any], step: int) -> Path:
    return forced_root(config) / "analysis" / offline_probe.step_name(step)


def examples_dir(config: dict[str, Any], step: int) -> Path:
    return forced_root(config) / "examples" / offline_probe.step_name(step)


def source_scored_path(config: dict[str, Any], step: int) -> Path:
    explicit = config["forced_answer"].get("source_scored")
    if explicit:
        return Path(str(explicit)).expanduser()
    return offline_probe.stage_dir(config, "scored", step) / "scored.parquet"


def resolve_checkpoint(config: dict[str, Any], step: int, cli_path: str | None = None) -> str:
    forced_path = cli_path or config.get("forced_answer", {}).get("checkpoint_path")
    return offline_probe.resolve_checkpoint_path(config, step, forced_path)


def _write_metadata(
    directory: Path,
    config: dict[str, Any],
    checkpoint_path: str,
    tokenizer_path: str | None,
    stage: str,
    step: int,
    extra: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = offline_probe.metadata(config, checkpoint_path, tokenizer_path, stage, step, extra)
    with open(directory / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_source_dataframe(source: pd.DataFrame, step: int) -> None:
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in source.columns]
    if missing:
        raise KeyError(f"source scored parquet is missing required columns: {missing}")
    if source.empty:
        return
    source_steps = pd.to_numeric(source["step"], errors="coerce")
    invalid_steps = sorted({str(value) for value in source.loc[source_steps.ne(step) | source_steps.isna(), "step"]})
    if invalid_steps:
        raise ValueError(f"CLI step {step} does not match source step values: {invalid_steps}")
    duplicates = source.duplicated(["step", "prompt_id", "sample_id"], keep=False)
    if bool(duplicates.any()):
        keys = source.loc[duplicates, ["step", "prompt_id", "sample_id"]].head(10).to_dict("records")
        raise ValueError(f"source contains duplicate trajectory keys: {keys}")


def validate_raw_dataframe(raw: pd.DataFrame, step: int) -> None:
    required = ["step", "prompt_id", "sample_id", "horizon", "kind", "cue", "status", "probe_success"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise KeyError(f"forced-answer raw parquet is missing required columns: {missing}")
    if not raw.empty:
        steps = pd.to_numeric(raw["step"], errors="coerce")
        invalid = sorted({str(value) for value in raw.loc[steps.ne(step) | steps.isna(), "step"]})
        if invalid:
            raise ValueError(f"CLI step {step} does not match raw step values: {invalid}")


def build_probe_points(response_len: int, horizons: Iterable[int], tail_offsets: Iterable[int]) -> list[dict[str, Any]]:
    """Return de-duplicated fixed/preterminal points; fixed points win ties."""

    response_len = int(response_len)
    fixed = sorted(set(int(value) for value in horizons))
    points: dict[int, dict[str, Any]] = {}
    for horizon in fixed:
        if horizon < response_len:
            points[horizon] = {"horizon": horizon, "kind": "fixed", "is_carry_forward": False}
    for offset in sorted(set(int(value) for value in tail_offsets)):
        horizon = response_len - offset
        if 0 <= horizon < response_len and horizon not in points:
            points[horizon] = {"horizon": horizon, "kind": "preterminal", "is_carry_forward": False}
    result = [points[horizon] for horizon in sorted(points)]
    if any(horizon >= response_len for horizon in fixed):
        result.append({"horizon": response_len, "kind": "terminal", "is_carry_forward": True})
    return result


def build_probe_token_ids(
    prompt_token_ids: Iterable[int],
    response_token_ids: Iterable[int],
    horizon: int,
    cue_token_ids: Iterable[int],
) -> list[int]:
    return [
        *[int(token_id) for token_id in prompt_token_ids],
        *[int(token_id) for token_id in list(response_token_ids)[: int(horizon)]],
        *[int(token_id) for token_id in cue_token_ids],
    ]


def _source_response_text(row: dict[str, Any]) -> Any:
    value = row.get("response_text")
    if offline_probe.is_missing_value(value):
        value = row.get("response_full")
    return None if offline_probe.is_missing_value(value) else value


def _terminal_reward(row: dict[str, Any]) -> float:
    value = pd.to_numeric(pd.Series([row.get("reward_10240")]), errors="coerce").iloc[0]
    if pd.isna(value):
        raise ValueError(
            f"source terminal reward is missing for prompt_id={row.get('prompt_id')}, sample_id={row.get('sample_id')}"
        )
    return float(value)


def _base_raw_record(
    row: dict[str, Any],
    config: dict[str, Any],
    prompt_ids: list[int],
    response_ids: list[int],
    terminal_reward: float,
) -> dict[str, Any]:
    forced = config["forced_answer"]
    storage = config.get("storage", {})
    save_text = bool(forced.get("save_source_text", storage.get("save_response_text", True)))
    save_ids = bool(forced.get("save_source_token_ids", storage.get("save_response_token_ids", True)))
    threshold = float(config.get("scoring", {}).get("positive_threshold", 0.0))
    return {
        "step": int(row["step"]),
        "prompt_id": row["prompt_id"],
        "sample_id": row["sample_id"],
        "original_index": row.get("original_index"),
        "data_source": row["data_source"],
        "ground_truth": row["ground_truth"],
        "extra_info": offline_probe.normalize_extra_info(row.get("extra_info", {})),
        "prompt_text": row.get("prompt_text") if save_text else None,
        "prompt_token_ids": prompt_ids if save_ids else None,
        "source_response_text": _source_response_text(row) if save_text else None,
        "response_token_ids": response_ids if save_ids else None,
        "response_token_len": int(row["response_token_len"]),
        "terminal_reward": terminal_reward,
        "terminal_success": float(terminal_reward > threshold),
    }


def _error_record(request: dict[str, Any], error_type: str, message: str, *, branch_id: int = -1) -> dict[str, Any]:
    return {
        **request["record"],
        "probe_branch_id": branch_id,
        "status": "context_overflow" if error_type == "context_overflow" else "generation_error",
        "error_type": error_type,
        "error_message": message,
        "is_context_overflow": error_type == "context_overflow",
        "completion_text": None,
        "completion_token_ids": None,
        "completion_token_len": None,
        "finish_reason": None,
        "stop_reason": None,
        "probe_reward": None,
        "probe_success": None,
    }


def _records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    for column in RAW_COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame[RAW_COLUMNS]


def _sampling_params(forced: dict[str, Any], sampling_params_class: Any) -> Any:
    return sampling_params_class(
        n=int(forced["n"]),
        max_tokens=int(forced["max_tokens"]),
        temperature=float(forced["temperature"]),
        top_p=float(forced["top_p"]),
        top_k=int(forced["top_k"]),
        ignore_eos=bool(forced["ignore_eos"]),
        seed=int(forced["seed"]),
    )


def _generate_batches(llm: Any, requests: list[dict[str, Any]], sampling_params: Any, batch_size: int):
    """Yield ``(request, output, exception)`` and isolate failed batches."""

    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        try:
            outputs = list(llm.generate([request["tokens_prompt"] for request in batch], sampling_params))
        except Exception:
            for request in batch:
                try:
                    output = list(llm.generate([request["tokens_prompt"]], sampling_params))
                    if output:
                        yield request, output[0], None
                    else:
                        yield request, None, RuntimeError("vLLM returned no RequestOutput")
                except Exception as exc:
                    yield request, None, exc
            continue
        for index, request in enumerate(batch):
            if index < len(outputs):
                yield request, outputs[index], None
            else:
                yield request, None, RuntimeError("vLLM omitted RequestOutput for request")


def run_generate(
    config: dict[str, Any],
    step: int,
    cli_checkpoint_path: str | None = None,
    force: bool = False,
    cli_overrides: dict[str, Any] | None = None,
) -> None:
    config = normalize_forced_answer_config(config)
    forced = config["forced_answer"]
    checkpoint_path = resolve_checkpoint(config, step, cli_checkpoint_path)
    source_path = source_scored_path(config, step)
    directory = raw_dir(config, step)
    output_path = directory / "raw.parquet"
    metadata_common = {
        "source_scored": str(source_path),
        "cli_overrides": cli_overrides or {},
        "force": force,
        "error_rate_threshold": ERROR_RATE_THRESHOLD,
        "primary_cue": forced["cues"][0]["name"],
    }
    if offline_probe.should_skip_existing_output(config, output_path, force):
        _write_metadata(
            directory,
            config,
            checkpoint_path,
            None,
            "forced_answer_generate",
            step,
            {**metadata_common, "skipped": True, "reason": "output exists"},
        )
        print(f"Skipping forced-answer generation because output exists: {output_path}")
        return

    source = pd.read_parquet(source_path)
    validate_source_dataframe(source, step)
    input_rows = len(source)
    if forced.get("max_trajectories") is not None:
        source = source.iloc[: int(forced["max_trajectories"])].copy()
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    tokenizer, tokenizer_path = offline_probe.load_tokenizer(config, checkpoint_path)
    reward_fn = offline_probe.load_reward_function(config)

    # vLLM remains generation-only.  In particular, importing this module and
    # calling run_analyze never reaches these imports.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rollout_config = config["rollout"]
    llm = LLM(**offline_probe.build_vllm_kwargs(checkpoint_path, tokenizer_path, rollout_config))
    sampling_params = _sampling_params(forced, SamplingParams)
    max_model_len = int(rollout_config.get("max_model_len", 12288))
    positive_threshold = float(config.get("scoring", {}).get("positive_threshold", 0.0))
    storage = config.get("storage", {})
    save_completion_text = bool(forced.get("save_completion_text", storage.get("save_response_text", True)))
    save_completion_ids = bool(forced.get("save_completion_token_ids", storage.get("save_response_token_ids", True)))

    cue_ids = {cue["name"]: offline_probe.tokenizer_encode(tokenizer, cue["text"]) for cue in forced["cues"]}
    encode_cache: dict[str, list[int]] = {}
    fallback_trajectories = 0
    records: list[dict[str, Any]] = []
    requests_by_cue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    request_id = 0
    context_overflow_requests = 0

    for _, series in source.iterrows():
        row = series.to_dict()
        response_ids = offline_probe.parse_token_ids(row["response_token_ids"])
        prompt_cell = row.get("prompt_token_ids", None)
        if offline_probe.is_missing_value(prompt_cell):
            prompt_text = str(row["prompt_text"])
            if prompt_text not in encode_cache:
                encode_cache[prompt_text] = offline_probe.tokenizer_encode(tokenizer, prompt_text)
            prompt_ids = list(encode_cache[prompt_text])
            fallback_trajectories += 1
        else:
            prompt_ids = offline_probe.parse_token_ids(prompt_cell)
        terminal_reward = _terminal_reward(row)
        base = _base_raw_record(row, config, prompt_ids, response_ids, terminal_reward)
        points = build_probe_points(int(row["response_token_len"]), forced["horizons"], forced["tail_offsets"])

        for point in points:
            if point["is_carry_forward"]:
                records.append(
                    {
                        **base,
                        "request_id": None,
                        "probe_branch_id": -1,
                        "horizon": point["horizon"],
                        "kind": "terminal",
                        "cue": "<terminal>",
                        "cue_text": None,
                        "status": "generated",
                        "error_type": None,
                        "error_message": None,
                        "is_context_overflow": False,
                        "is_carry_forward": True,
                        "cue_token_ids": None,
                        "input_token_ids": None,
                        "input_token_len": None,
                        "completion_text": None,
                        "completion_token_ids": None,
                        "completion_token_len": 0,
                        "finish_reason": "carry_forward",
                        "stop_reason": None,
                        "probe_reward": base["terminal_reward"],
                        "probe_success": base["terminal_success"],
                    }
                )
                continue
            for cue in forced["cues"]:
                input_ids = build_probe_token_ids(prompt_ids, response_ids, point["horizon"], cue_ids[cue["name"]])
                record = {
                    **base,
                    "request_id": request_id,
                    "horizon": point["horizon"],
                    "kind": point["kind"],
                    "cue": cue["name"],
                    "cue_text": cue["text"] if save_completion_text else None,
                    "is_carry_forward": False,
                    "cue_token_ids": cue_ids[cue["name"]] if save_completion_ids else None,
                    "input_token_ids": input_ids if save_completion_ids else None,
                    "input_token_len": len(input_ids),
                }
                request = {"record": record, "tokens_prompt": TokensPrompt(prompt_token_ids=input_ids)}
                if len(input_ids) + int(forced["max_tokens"]) > max_model_len:
                    records.append(
                        _error_record(
                            request,
                            "context_overflow",
                            f"input_len={len(input_ids)} + max_tokens={forced['max_tokens']} "
                            f"exceeds max_model_len={max_model_len}",
                        )
                    )
                    context_overflow_requests += 1
                else:
                    requests_by_cue[cue["name"]].append(request)
                request_id += 1

    generation_error_requests = 0
    scoring_errors = 0
    extra_completions = 0
    generated_requests = 0
    for cue in forced["cues"]:
        cue_requests = requests_by_cue[cue["name"]]
        for request, request_output, generation_error in _generate_batches(
            llm, cue_requests, sampling_params, int(forced["batch_size_requests"])
        ):
            generated_requests += 1
            if generation_error is not None or request_output is None:
                generation_error_requests += 1
                message = str(generation_error or "missing RequestOutput")
                records.append(_error_record(request, "generation_error", message))
                continue
            completions = list(getattr(request_output, "outputs", []) or [])
            if len(completions) < int(forced["n"]):
                generation_error_requests += 1
            if len(completions) > int(forced["n"]):
                extra_completions += len(completions) - int(forced["n"])
            for branch_id in range(int(forced["n"])):
                if branch_id >= len(completions):
                    records.append(
                        _error_record(request, "missing_completion", "vLLM omitted completion", branch_id=branch_id)
                    )
                    continue
                completion = completions[branch_id]
                text = getattr(completion, "text", "")
                token_ids = list(getattr(completion, "token_ids", []) or [])
                record = {
                    **request["record"],
                    "probe_branch_id": branch_id,
                    "status": "generated",
                    "error_type": None,
                    "error_message": None,
                    "is_context_overflow": False,
                    "completion_text": text if save_completion_text else None,
                    "completion_token_ids": token_ids if save_completion_ids else None,
                    "completion_token_len": len(token_ids),
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    "probe_reward": None,
                    "probe_success": None,
                }
                try:
                    # The verifier receives the short completion and nothing else.
                    reward = offline_probe.call_reward_function(
                        reward_fn,
                        record["data_source"],
                        text,
                        record["ground_truth"],
                        record["extra_info"],
                    )
                    record["probe_reward"] = reward
                    record["probe_success"] = float(reward > positive_threshold)
                except Exception as exc:
                    scoring_errors += 1
                    record["status"] = "scoring_error"
                    record["error_type"] = "scoring_error"
                    record["error_message"] = str(exc)
                records.append(record)

    raw = _records_frame(records)
    offline_probe.write_parquet(raw, output_path, config)
    denominator = generated_requests * int(forced["n"])
    numerator = generation_error_requests * int(forced["n"]) + scoring_errors
    error_rate = float(numerator / denominator) if denominator else 0.0
    _write_metadata(
        directory,
        config,
        checkpoint_path,
        tokenizer_path,
        "forced_answer_generate",
        step,
        {
            **metadata_common,
            "skipped": False,
            "output_path": str(output_path),
            "input_trajectories": input_rows,
            "selected_trajectories": len(source),
            "prompt_token_fallback_trajectories": fallback_trajectories,
            "prompt_token_fallback_unique_texts": len(encode_cache),
            "total_tokenized_requests": request_id,
            "non_overflow_requests": generated_requests,
            "context_overflow_requests": context_overflow_requests,
            "generation_error_requests": generation_error_requests,
            "scoring_errors": scoring_errors,
            "extra_completions_ignored": extra_completions,
            "equivalent_branch_opportunities": denominator,
            "equivalent_branch_errors": numerator,
            "stage_error_rate": error_rate,
            "num_raw_rows": len(raw),
        },
    )
    print(f"Wrote forced-answer raw parquet: {output_path}")
    if error_rate > ERROR_RATE_THRESHOLD:
        raise RuntimeError(
            f"forced-answer stage error rate {error_rate:.2%} exceeds fixed {ERROR_RATE_THRESHOLD:.0%} threshold"
        )


def _expand_carry_forward(raw: pd.DataFrame, horizons: list[int], cue_names: list[str]) -> pd.DataFrame:
    carry_flags = (
        raw["is_carry_forward"].fillna(False).astype(bool)
        if "is_carry_forward" in raw
        else pd.Series(False, index=raw.index)
    )
    ordinary = raw.loc[~carry_flags].copy()
    expanded = [ordinary]
    carry = raw.loc[carry_flags]
    for _, row in carry.iterrows():
        response_len = int(row["response_token_len"])
        for horizon in horizons:
            if horizon < response_len:
                continue
            for cue in cue_names:
                out = row.to_dict()
                out.update({"horizon": horizon, "kind": "fixed", "cue": cue})
                expanded.append(pd.DataFrame([out]))
    return pd.concat(expanded, ignore_index=True) if expanded else raw.iloc[:0].copy()


def _cluster_se(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1) / math.sqrt(len(numeric)))


def aggregate_probe_values(
    raw: pd.DataFrame,
    horizons: Iterable[int],
    cues: Iterable[str | dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate generated binary outcomes into V(h) curves."""

    fixed = sorted(set(int(value) for value in horizons))
    cue_names = [str(cue["name"] if isinstance(cue, dict) else cue) for cue in cues]
    expanded = _expand_carry_forward(raw, fixed, cue_names)
    rows: list[dict[str, Any]] = []
    group_columns = ["step", "horizon", "kind", "cue"]
    for keys, group in expanded.groupby(group_columns, sort=True, dropna=False):
        valid = group.loc[group["status"].eq("generated") & group["probe_success"].notna()].copy()
        opportunity = group.groupby(TRAJECTORY_KEYS, sort=False, dropna=False)["status"].agg(list)
        overflow_rate = float(opportunity.map(lambda statuses: "context_overflow" in statuses).mean())
        error_rate = float(
            opportunity.map(
                lambda statuses: any(status in {"generation_error", "scoring_error"} for status in statuses)
            ).mean()
        )
        if valid.empty:
            v_mean = reward_mean = reward_std = math.nan
            se = 0.0
            num_prompts = num_trajectories = 0
        else:
            valid["probe_success"] = pd.to_numeric(valid["probe_success"])
            valid["probe_reward"] = pd.to_numeric(valid["probe_reward"], errors="coerce")
            trajectory_values = (
                valid.groupby(TRAJECTORY_KEYS, sort=False, dropna=False)["probe_success"].mean().reset_index()
            )
            prompt_values = trajectory_values.groupby("prompt_id", sort=False)["probe_success"].mean()
            v_mean = float(valid["probe_success"].mean())
            reward_mean = float(valid["probe_reward"].mean())
            reward_std = float(valid["probe_reward"].std(ddof=0))
            se = _cluster_se(prompt_values)
            num_prompts = int(prompt_values.size)
            num_trajectories = int(len(trajectory_values))
        rows.append(
            {
                "step": keys[0],
                "horizon": int(keys[1]),
                "kind": keys[2],
                "cue": keys[3],
                "value": v_mean,
                "v_mean": v_mean,
                "v_se_prompt_cluster": se,
                "probe_reward_mean": reward_mean,
                "probe_reward_std": reward_std,
                "num_valid_branches": int(len(valid)),
                "num_valid_trajectories": num_trajectories,
                "num_prompts": num_prompts,
                "num_trajectory_opportunities": int(len(opportunity)),
                "trajectory_overflow_rate": overflow_rate,
                "trajectory_error_rate": error_rate,
            }
        )
    return pd.DataFrame(rows)


def _trajectory_probe_values(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    forced = config["forced_answer"]
    expanded = _expand_carry_forward(raw, forced["horizons"], [cue["name"] for cue in forced["cues"]])
    rows: list[dict[str, Any]] = []
    group_columns = [*TRAJECTORY_KEYS, "horizon", "kind", "cue"]
    expected_n = int(forced["n"])
    for keys, group in expanded.groupby(group_columns, sort=False, dropna=False):
        carry = bool(group.get("is_carry_forward", False).fillna(False).astype(bool).any())
        valid = group.loc[group["status"].eq("generated") & group["probe_success"].notna()]
        has_problem = bool(group["status"].isin(["context_overflow", "generation_error", "scoring_error"]).any())
        complete = not has_problem and (carry and len(valid) == 1 or not carry and len(valid) == expected_n)
        rows.append(
            {
                "step": keys[0],
                "prompt_id": keys[1],
                "sample_id": keys[2],
                "horizon": int(keys[3]),
                "kind": keys[4],
                "cue": keys[5],
                "value": float(pd.to_numeric(valid["probe_success"]).mean()) if complete else math.nan,
                "reward_mean": float(pd.to_numeric(valid["probe_reward"], errors="coerce").mean())
                if complete
                else math.nan,
                "complete": complete,
                "had_overflow": bool(group["status"].eq("context_overflow").any()),
                "had_error": bool(group["status"].isin(["generation_error", "scoring_error"]).any()),
            }
        )
    return pd.DataFrame(rows)


def _nullable(value: bool | None) -> Any:
    return pd.NA if value is None else bool(value)


def _crossed_twice(states: list[bool]) -> bool:
    return sum(left != right for left, right in zip(states, states[1:], strict=False)) >= 2


def build_taxonomy(
    source: pd.DataFrame,
    trajectory_values: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    forced = config["forced_answer"]
    threshold = float(forced["stability_threshold"])
    positive_threshold = float(config.get("scoring", {}).get("positive_threshold", 0.0))
    primary_cue = forced["cues"][0]["name"]
    primary = trajectory_values.loc[trajectory_values["cue"].eq(primary_cue)]
    value_groups = {key: group for key, group in primary.groupby(TRAJECTORY_KEYS, sort=False, dropna=False)}
    rows: list[dict[str, Any]] = []

    for _, source_row in source.iterrows():
        source_dict = source_row.to_dict()
        key = (source_dict["step"], source_dict["prompt_id"], source_dict["sample_id"])
        group = value_groups.get(key, primary.iloc[:0])
        values = {
            (int(row["horizon"]), row["kind"]): float(row["value"])
            for _, row in group.iterrows()
            if not pd.isna(row["value"])
        }
        fixed_map = {horizon: values.get((horizon, "fixed")) for horizon in forced["horizons"]}
        response_len = int(source_dict["response_token_len"])
        terminal_reward = pd.to_numeric(pd.Series([source_dict["reward_10240"]]), errors="coerce").iloc[0]
        terminal_correct = None if pd.isna(terminal_reward) else bool(terminal_reward > positive_threshold)

        v2048 = fixed_map.get(2048)
        early = None if v2048 is None else v2048 >= threshold

        if terminal_correct is False:
            delayed: bool | None = False
        elif terminal_correct is None or v2048 is None:
            delayed = None
        elif v2048 >= threshold:
            delayed = False
        else:
            later_horizons = [h for h in forced["horizons"] if h > 2048]
            later_values = [fixed_map.get(h) for h in later_horizons]
            if not later_values or any(value is None for value in later_values):
                delayed = None
            else:
                stable_indices = [index for index, value in enumerate(later_values) if value >= threshold]
                delayed = bool(stable_indices) and all(
                    value >= threshold for value in later_values[stable_indices[0] :]
                )

        fixed_before = [h for h in forced["horizons"] if h < response_len]
        tail_targets = sorted(
            {
                response_len - int(offset)
                for offset in forced["tail_offsets"]
                if 0 <= response_len - int(offset) < response_len
            }
        )
        if terminal_correct is False:
            terminal_only: bool | None = False
        elif terminal_correct is None or not fixed_before or not tail_targets:
            terminal_only = None
        else:
            last_fixed = fixed_map.get(max(fixed_before))
            last_tail_horizon = max(tail_targets)
            last_tail = values.get((last_tail_horizon, "preterminal"), values.get((last_tail_horizon, "fixed")))
            terminal_only = (
                None if last_fixed is None or last_tail is None else last_fixed < threshold and last_tail < threshold
            )

        curve_values = [fixed_map.get(horizon) for horizon in forced["horizons"]]
        if len(curve_values) < 3 or any(value is None for value in curve_values):
            unstable: bool | None = None
        else:
            states = [value >= threshold for value in curve_values]
            deltas = [right - left for left, right in zip(curve_values, curve_values[1:], strict=False)]
            decline_then_recover = any(
                delta < 0 and any(later > 0 for later in deltas[index + 1 :]) for index, delta in enumerate(deltas)
            )
            unstable = _crossed_twice(states) or decline_then_recover

        if terminal_correct is True:
            early_wrong: bool | None = False
        elif terminal_correct is None or not fixed_before:
            early_wrong = None
        else:
            before_values = [fixed_map.get(horizon) for horizon in fixed_before]
            if any(value is None for value in before_values):
                early_wrong = None
            else:
                early_wrong = any(value >= threshold for value in before_values)

        stable_horizon = None
        for index, horizon in enumerate(forced["horizons"]):
            tail = curve_values[index:]
            if tail and all(value is not None and value >= threshold for value in tail):
                stable_horizon = horizon
                break
        rows.append(
            {
                "step": source_dict["step"],
                "prompt_id": source_dict["prompt_id"],
                "sample_id": source_dict["sample_id"],
                "terminal_reward": None if pd.isna(terminal_reward) else float(terminal_reward),
                "terminal_correct": _nullable(terminal_correct),
                "stable_horizon": stable_horizon,
                "delayed_solve_candidate": _nullable(delayed),
                "early_recoverable": _nullable(early),
                "terminal_only_candidate": _nullable(terminal_only),
                "unstable_diagnostic": _nullable(unstable),
                "early_recoverable_but_terminal_wrong": _nullable(early_wrong),
            }
        )
    result = pd.DataFrame(rows)
    for label in ["terminal_correct", *TAXONOMY_LABELS]:
        result[label] = result[label].astype("boolean")
    return result


def taxonomy_summary(taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    step = int(taxonomy["step"].iloc[0]) if len(taxonomy) else -1
    for label in TAXONOMY_LABELS:
        values = taxonomy[label].astype("boolean")
        true_count = int(values.eq(True).sum())
        false_count = int(values.eq(False).sum())
        unknown_count = int(values.isna().sum())
        known_count = true_count + false_count
        rows.append(
            {
                "step": step,
                "label": label,
                "true_count": true_count,
                "false_count": false_count,
                "unknown_count": unknown_count,
                "known_count": known_count,
                "known_fraction": float(known_count / len(values)) if len(values) else math.nan,
                "known_case_rate": float(true_count / known_count) if known_count else math.nan,
            }
        )
    return pd.DataFrame(rows)


def response_length_bucket(length: Any) -> str:
    length = int(length)
    if length <= 2048:
        return "<=2048"
    if length <= 4096:
        return "2049-4096"
    if length <= 8192:
        return "4097-8192"
    if length <= 10240:
        return "8193-10240"
    return ">10240"


def _standardize_group(group: pd.DataFrame, value_column: str, prefix: str, eps: float) -> pd.DataFrame:
    group = group.copy()
    values = pd.to_numeric(group[value_column], errors="coerce")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    degenerate = not math.isfinite(std) or std < eps
    group[f"{prefix}_group_mean"] = mean
    group[f"{prefix}_group_std"] = std
    group[f"{prefix}_degenerate"] = degenerate
    group[f"{prefix}_advantage"] = 0.0 if degenerate else (values - mean) / std
    return group


def build_counterfactual_advantage(
    source: pd.DataFrame,
    trajectory_values: pd.DataFrame,
    taxonomy: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    forced = config["forced_answer"]
    primary_cue = forced["cues"][0]["name"]
    fixed = trajectory_values.loc[trajectory_values["cue"].eq(primary_cue) & trajectory_values["kind"].eq("fixed")]
    value_lookup = {
        (row["step"], row["prompt_id"], row["sample_id"], int(row["horizon"])): row["value"]
        for _, row in fixed.iterrows()
    }
    source_lookup = {(row["step"], row["prompt_id"], row["sample_id"]): row.to_dict() for _, row in source.iterrows()}
    taxonomy_lookup = {
        (row["step"], row["prompt_id"], row["sample_id"]): row.to_dict() for _, row in taxonomy.iterrows()
    }
    rows = []
    horizons = forced["horizons"]
    for key, source_row in source_lookup.items():
        response_len = int(source_row["response_token_len"])
        for start, end in zip(horizons, horizons[1:], strict=False):
            if start >= response_len:
                continue
            left = value_lookup.get((*key, start), math.nan)
            right = value_lookup.get((*key, end), math.nan)
            if pd.isna(left) or pd.isna(right):
                continue
            tax = taxonomy_lookup[key]
            stable_horizon = tax.get("stable_horizon")
            has_stable_horizon = stable_horizon is not None and not pd.isna(stable_horizon)
            out = {
                "step": key[0],
                "prompt_id": key[1],
                "sample_id": key[2],
                "original_index": source_row.get("original_index"),
                "data_source": source_row.get("data_source"),
                "cue": primary_cue,
                "horizon_start": start,
                "horizon_end": end,
                "v_start": float(left),
                "v_end": float(right),
                "hc_reward": float(right - left),
                "terminal_reward": float(source_row["reward_10240"]),
                "response_token_len": response_len,
                "response_length_bucket": response_length_bucket(response_len),
                "stable_horizon": stable_horizon if has_stable_horizon else None,
                "after_stable": has_stable_horizon and start >= int(stable_horizon),
            }
            for label in TAXONOMY_LABELS:
                out[label] = tax[label]
            rows.append(out)
    if not rows:
        return pd.DataFrame(
            columns=[
                "step",
                "prompt_id",
                "sample_id",
                "horizon_start",
                "horizon_end",
                "hc_reward",
                "hc_advantage",
                "terminal_reward",
                "terminal_advantage",
            ]
        )
    frame = pd.DataFrame(rows)
    eps = float(forced["eps"])
    hc_groups = frame.groupby(["prompt_id", "horizon_start", "horizon_end"], sort=False)["hc_reward"]
    frame["hc_group_mean"] = hc_groups.transform("mean")
    frame["hc_group_std"] = hc_groups.transform(lambda values: values.std(ddof=0))
    frame["hc_degenerate"] = frame["hc_group_std"].lt(eps) | frame["hc_group_std"].isna()
    frame["hc_advantage"] = (frame["hc_reward"] - frame["hc_group_mean"]) / frame["hc_group_std"]
    frame.loc[frame["hc_degenerate"], "hc_advantage"] = 0.0

    terminal_values = source[["prompt_id", "sample_id", "reward_10240"]].copy()
    terminal_values["reward_10240"] = pd.to_numeric(terminal_values["reward_10240"], errors="coerce")
    terminal_stats = terminal_values.groupby("prompt_id", sort=False)["reward_10240"].agg(
        terminal_group_mean="mean", terminal_group_std=lambda values: values.std(ddof=0)
    )
    frame["terminal_group_mean"] = frame["prompt_id"].map(terminal_stats["terminal_group_mean"])
    frame["terminal_group_std"] = frame["prompt_id"].map(terminal_stats["terminal_group_std"])
    frame["terminal_degenerate"] = frame["terminal_group_std"].lt(eps) | frame["terminal_group_std"].isna()
    frame["terminal_advantage"] = (frame["terminal_reward"] - frame["terminal_group_mean"]) / frame[
        "terminal_group_std"
    ]
    frame.loc[frame["terminal_degenerate"], "terminal_advantage"] = 0.0
    frame["hc_sign"] = frame["hc_advantage"].map(lambda value: -1 if value < 0 else 1 if value > 0 else 0)
    frame["terminal_sign"] = frame["terminal_advantage"].map(lambda value: -1 if value < 0 else 1 if value > 0 else 0)
    frame["sign_disagreement"] = frame["hc_sign"].ne(frame["terminal_sign"])
    return frame


def _rate_and_cluster_se(
    frame: pd.DataFrame, numerator: pd.Series, denominator: pd.Series | None = None
) -> tuple[float, float]:
    denominator = pd.Series(True, index=frame.index) if denominator is None else denominator.fillna(False).astype(bool)
    selected = frame.loc[denominator].copy()
    if selected.empty:
        return math.nan, math.nan
    selected["_indicator"] = numerator.loc[selected.index].fillna(False).astype(bool).astype(float)
    prompt_values = selected.groupby("prompt_id", sort=False)["_indicator"].mean()
    return float(selected["_indicator"].mean()), _cluster_se(prompt_values)


def _mean_and_cluster_se(frame: pd.DataFrame, column: str, mask: pd.Series) -> tuple[float, float]:
    selected = frame.loc[mask].copy()
    if selected.empty:
        return math.nan, math.nan
    prompt_values = selected.groupby("prompt_id", sort=False)[column].mean()
    return float(pd.to_numeric(selected[column]).mean()), _cluster_se(prompt_values)


def build_advantage_summary(advantage: pd.DataFrame, step: int) -> pd.DataFrame:
    bucket_names = ["<=2048", "2049-4096", "4097-8192", "8193-10240", ">10240"]
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", advantage)]
    bucket_series = advantage.get("response_length_bucket", pd.Series(index=advantage.index, dtype=object))
    scopes.extend(
        ("response_length_bucket", bucket, advantage.loc[bucket_series.eq(bucket)]) for bucket in bucket_names
    )
    rows = []
    for scope, bucket, frame in scopes:
        row: dict[str, Any] = {
            "step": step,
            "scope": scope,
            "response_length_bucket": bucket,
            "num_segments": int(len(frame)),
            "num_prompts": int(frame["prompt_id"].nunique()) if len(frame) else 0,
        }
        metrics = {
            "hc_degenerate_rate": frame.get("hc_degenerate", pd.Series(False, index=frame.index)),
            "terminal_degenerate_rate": frame.get("terminal_degenerate", pd.Series(False, index=frame.index)),
            "sign_disagreement_rate": frame.get("sign_disagreement", pd.Series(False, index=frame.index)),
            "terminal_positive_to_hc_nonpositive_rate": (
                frame.get("terminal_advantage", pd.Series(index=frame.index, dtype=float)).gt(0)
                & frame.get("hc_advantage", pd.Series(index=frame.index, dtype=float)).le(0)
            ),
            "terminal_nonpositive_to_hc_positive_rate": (
                frame.get("terminal_advantage", pd.Series(index=frame.index, dtype=float)).le(0)
                & frame.get("hc_advantage", pd.Series(index=frame.index, dtype=float)).gt(0)
            ),
        }
        for name, numerator in metrics.items():
            rate, se = _rate_and_cluster_se(frame, numerator)
            row[name] = rate
            row[f"{name}_prompt_cluster_se"] = se
        row["adv_sign_disagreement_rate"] = row["sign_disagreement_rate"]
        row["adv_sign_disagreement_rate_prompt_cluster_se"] = row["sign_disagreement_rate_prompt_cluster_se"]
        row["adv_pos_to_nonpos_rate"] = row["terminal_positive_to_hc_nonpositive_rate"]
        row["adv_pos_to_nonpos_rate_prompt_cluster_se"] = row[
            "terminal_positive_to_hc_nonpositive_rate_prompt_cluster_se"
        ]
        row["adv_nonpos_to_pos_rate"] = row["terminal_nonpositive_to_hc_positive_rate"]
        row["adv_nonpos_to_pos_rate_prompt_cluster_se"] = row[
            "terminal_nonpositive_to_hc_positive_rate_prompt_cluster_se"
        ]

        stable_mean, stable_se = _mean_and_cluster_se(
            frame,
            "hc_advantage",
            frame.get("after_stable", pd.Series(False, index=frame.index)).fillna(False).astype(bool),
        )
        row["mean_hc_advantage_after_stable"] = stable_mean
        row["mean_hc_advantage_after_stable_prompt_cluster_se"] = stable_se

        for label in TAXONOMY_LABELS:
            label_mask = frame.get(label, pd.Series(pd.NA, index=frame.index, dtype="boolean")).eq(True).fillna(False)
            positive_credit = label_mask & frame.get(
                "terminal_advantage", pd.Series(index=frame.index, dtype=float)
            ).gt(0)
            rate, se = _rate_and_cluster_se(
                frame,
                frame.get("hc_advantage", pd.Series(index=frame.index, dtype=float)).le(0),
                positive_credit,
            )
            row[f"{label}_remove_positive_credit_rate"] = rate
            row[f"{label}_remove_positive_credit_rate_prompt_cluster_se"] = se

        any_taxonomy = pd.Series(False, index=frame.index)
        for label in TAXONOMY_LABELS:
            any_taxonomy |= (
                frame.get(label, pd.Series(pd.NA, index=frame.index, dtype="boolean")).eq(True).fillna(False)
            )
        terminal_positive = frame.get("terminal_advantage", pd.Series(index=frame.index, dtype=float)).gt(0)
        removed, removed_se = _rate_and_cluster_se(
            frame,
            frame.get("hc_advantage", pd.Series(index=frame.index, dtype=float)).le(0),
            any_taxonomy & terminal_positive,
        )
        row["taxonomy_remove_positive_credit_rate"] = removed
        row["taxonomy_remove_positive_credit_rate_prompt_cluster_se"] = removed_se

        delayed = (
            frame.get("delayed_solve_candidate", pd.Series(pd.NA, index=frame.index, dtype="boolean"))
            .eq(True)
            .fillna(False)
        )
        hc_positive = frame.get("hc_advantage", pd.Series(index=frame.index, dtype=float)).gt(0)
        retained, retained_se = _rate_and_cluster_se(frame, hc_positive, delayed & terminal_positive)
        newly, newly_se = _rate_and_cluster_se(frame, hc_positive, delayed & ~terminal_positive)
        row["delayed_solve_positive_credit_retained_rate"] = retained
        row["delayed_solve_positive_credit_retained_rate_prompt_cluster_se"] = retained_se
        row["delayed_solve_newly_positive_rate"] = newly
        row["delayed_solve_newly_positive_rate_prompt_cluster_se"] = newly_se
        rows.append(row)
    return pd.DataFrame(rows)


def _json_value(value: Any) -> Any:
    if value is pd.NA or offline_probe.is_missing_value(value):
        return None
    return offline_probe.jsonable(value)


def _review_rows(
    source: pd.DataFrame,
    raw: pd.DataFrame,
    values: pd.DataFrame,
    taxonomy: pd.DataFrame,
    config: dict[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    selected = taxonomy.loc[taxonomy[label].eq(True).fillna(False)].head(100)
    source_lookup = {(row["step"], row["prompt_id"], row["sample_id"]): row.to_dict() for _, row in source.iterrows()}
    primary_cue = config["forced_answer"]["cues"][0]["name"]
    records = []
    for _, tax_row in selected.iterrows():
        key = (tax_row["step"], tax_row["prompt_id"], tax_row["sample_id"])
        source_row = source_lookup[key]
        trajectory_raw = raw.loc[raw["step"].eq(key[0]) & raw["prompt_id"].eq(key[1]) & raw["sample_id"].eq(key[2])]
        trajectory_values = values.loc[
            values["step"].eq(key[0])
            & values["prompt_id"].eq(key[1])
            & values["sample_id"].eq(key[2])
            & values["cue"].eq(primary_cue)
        ]
        value_curve = {
            f"{row['kind']}:{int(row['horizon'])}": _json_value(row["value"])
            for _, row in trajectory_values.sort_values(["horizon", "kind"]).iterrows()
        }
        completions = []
        for _, row in trajectory_raw.loc[trajectory_raw["status"].isin(["generated", "scoring_error"])].iterrows():
            if bool(row.get("is_carry_forward", False)):
                continue
            completions.append(
                {
                    "horizon": int(row["horizon"]),
                    "kind": row["kind"],
                    "cue": row["cue"],
                    "branch_id": int(row["probe_branch_id"]),
                    "text": _json_value(row.get("completion_text")),
                    "token_ids": _json_value(row.get("completion_token_ids")),
                    "reward": _json_value(row.get("probe_reward")),
                    "success": _json_value(row.get("probe_success")),
                    "status": row["status"],
                }
            )
        prompt_token_ids = source_row.get("prompt_token_ids")
        if offline_probe.is_missing_value(prompt_token_ids) and not trajectory_raw.empty:
            saved_prompt_ids = (
                trajectory_raw["prompt_token_ids"].dropna() if "prompt_token_ids" in trajectory_raw else []
            )
            if len(saved_prompt_ids):
                prompt_token_ids = saved_prompt_ids.iloc[0]
        records.append(
            {
                "step": key[0],
                "prompt_id": key[1],
                "sample_id": key[2],
                "data_source": _json_value(source_row.get("data_source")),
                "ground_truth": _json_value(source_row.get("ground_truth")),
                "prompt_text": _json_value(source_row.get("prompt_text")),
                "prompt_token_ids": _json_value(prompt_token_ids),
                "response_text": _json_value(_source_response_text(source_row)),
                "response_token_ids": _json_value(source_row.get("response_token_ids")),
                "response_token_len": int(source_row["response_token_len"]),
                "terminal_reward": _json_value(source_row.get("reward_10240")),
                "probe_values": value_curve,
                "short_completions": completions,
                "taxonomy": {name: _json_value(tax_row[name]) for name in TAXONOMY_LABELS},
            }
        )
    return records


def run_analyze(
    config: dict[str, Any],
    step: int,
    cli_checkpoint_path: str | None = None,
    force: bool = False,
    cli_overrides: dict[str, Any] | None = None,
) -> None:
    """Analyze raw probes without importing vLLM/transformers or loading a model."""

    config = normalize_forced_answer_config(config)
    forced = config["forced_answer"]
    checkpoint_path = resolve_checkpoint(config, step, cli_checkpoint_path)
    source_path = source_scored_path(config, step)
    input_path = raw_dir(config, step) / "raw.parquet"
    directory = analysis_dir(config, step)
    curve_path = directory / "probe_curve.csv"
    metadata_common = {
        "source_scored": str(source_path),
        "input_raw": str(input_path),
        "cli_overrides": cli_overrides or {},
        "force": force,
        "primary_cue": forced["cues"][0]["name"],
    }
    if offline_probe.should_skip_existing_output(config, curve_path, force):
        _write_metadata(
            directory,
            config,
            checkpoint_path,
            None,
            "forced_answer_analyze",
            step,
            {**metadata_common, "skipped": True, "reason": "output exists"},
        )
        print(f"Skipping forced-answer analysis because output exists: {curve_path}")
        return

    raw = pd.read_parquet(input_path)
    source = pd.read_parquet(source_path)
    validate_raw_dataframe(raw, step)
    validate_source_dataframe(source, step)
    if forced.get("max_trajectories") is not None:
        source = source.iloc[: int(forced["max_trajectories"])].copy()
        selected_keys = pd.MultiIndex.from_frame(source[TRAJECTORY_KEYS])
        raw_keys = pd.MultiIndex.from_frame(raw[TRAJECTORY_KEYS])
        raw = raw.loc[raw_keys.isin(selected_keys)].copy()

    curve = aggregate_probe_values(raw, forced["horizons"], forced["cues"])
    trajectory_values = _trajectory_probe_values(raw, config)
    taxonomy = build_taxonomy(source, trajectory_values, config)
    tax_summary = taxonomy_summary(taxonomy)
    advantage = build_counterfactual_advantage(source, trajectory_values, taxonomy, config)
    advantage_summary = build_advantage_summary(advantage, step)

    directory.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_path, index=False)
    trajectory_values.to_parquet(directory / "trajectory_probe_values.parquet", index=False)
    taxonomy.to_parquet(directory / "taxonomy.parquet", index=False)
    tax_summary.to_csv(directory / "taxonomy_summary.csv", index=False)
    offline_probe.write_parquet(advantage, directory / "counterfactual_advantage.parquet", config)
    advantage_summary.to_csv(directory / "advantage_summary.csv", index=False)

    review_directory = examples_dir(config, step)
    for label in TAXONOMY_LABELS:
        offline_probe.write_jsonl(
            review_directory / f"{label}.jsonl",
            _review_rows(source, raw, trajectory_values, taxonomy, config, label),
        )
    missing_response_text = int(sum(_source_response_text(row.to_dict()) is None for _, row in source.iterrows()))
    _write_metadata(
        directory,
        config,
        checkpoint_path,
        None,
        "forced_answer_analyze",
        step,
        {
            **metadata_common,
            "skipped": False,
            "output_dir": str(directory),
            "examples_dir": str(review_directory),
            "num_source_trajectories": len(source),
            "num_raw_rows": len(raw),
            "num_curve_rows": len(curve),
            "num_advantage_rows": len(advantage),
            "source_rows_missing_response_text": missing_response_text,
        },
    )
    print(f"Wrote forced-answer analysis outputs: {directory}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline forced-answer prefix probe")
    parser.add_argument("--config", required=True, help="Path to probe_config.yaml")
    parser.add_argument("--step", required=True, type=int, help="Checkpoint global step")
    parser.add_argument("--mode", required=True, choices=["generate", "analyze", "all"])
    parser.add_argument("--checkpoint-path", default=None, help="Override fixed checkpoint path")
    parser.add_argument("--source-scored", default=None, help="Override source scored.parquet")
    parser.add_argument("--output-dir", default=None, help="Override paths.output_dir")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Use the first N source rows")
    parser.add_argument("--probe-n", type=int, default=None, help="Override forced_answer.n")
    parser.add_argument("--force", action="store_true", help="Overwrite existing stage outputs")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config, overrides = apply_cli_overrides(offline_probe.load_config(args.config), args)
    if args.mode in {"generate", "all"}:
        run_generate(config, args.step, args.checkpoint_path, args.force, overrides)
    if args.mode in {"analyze", "all"}:
        run_analyze(config, args.step, args.checkpoint_path, args.force, overrides)


if __name__ == "__main__":
    main()
