import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "offline_long_reward_probe" / "offline_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("offline_probe", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyTokenizer:
    def __init__(self):
        self.chat_template_calls = []

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        return {"input_ids": text.split()}

    def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=True):
        self.chat_template_calls.append(
            {"prompt": prompt, "tokenize": tokenize, "add_generation_prompt": add_generation_prompt}
        )
        return "chat rendered"

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


class PromptWithToList:
    def tolist(self):
        return [{"role": "user", "content": "from tolist"}]


class PromptWithAsPy:
    def as_py(self):
        return [{"role": "user", "content": "from as_py"}]


class UnsupportedPrompt:
    pass


def test_render_prompt_uses_chat_template_for_message_lists():
    probe = load_probe_module()
    tokenizer = DummyTokenizer()
    row = {"prompt": [{"role": "user", "content": "solve"}]}

    assert probe.render_prompt(row, tokenizer, "prompt") == "chat rendered"
    assert tokenizer.chat_template_calls == [
        {
            "prompt": [{"role": "user", "content": "solve"}],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]


def test_render_prompt_uses_chat_template_for_tolist_message_lists():
    probe = load_probe_module()
    tokenizer = DummyTokenizer()

    assert probe.render_prompt({"prompt": PromptWithToList()}, tokenizer, "prompt") == "chat rendered"
    assert tokenizer.chat_template_calls[0]["prompt"] == [{"role": "user", "content": "from tolist"}]


def test_render_prompt_uses_chat_template_for_as_py_message_lists():
    probe = load_probe_module()
    tokenizer = DummyTokenizer()

    assert probe.render_prompt({"prompt": PromptWithAsPy()}, tokenizer, "prompt") == "chat rendered"
    assert tokenizer.chat_template_calls[0]["prompt"] == [{"role": "user", "content": "from as_py"}]


def test_render_prompt_parses_json_encoded_message_lists():
    probe = load_probe_module()
    tokenizer = DummyTokenizer()
    prompt = '[{"role": "user", "content": "from json"}]'

    assert probe.render_prompt({"prompt": prompt}, tokenizer, "prompt") == "chat rendered"
    assert tokenizer.chat_template_calls[0]["prompt"] == [{"role": "user", "content": "from json"}]


def test_render_prompt_keeps_plain_strings_without_json_message_schema():
    probe = load_probe_module()
    tokenizer = DummyTokenizer()

    assert probe.render_prompt({"prompt": "solve this"}, tokenizer, "prompt") == "solve this"
    assert tokenizer.chat_template_calls == []


def test_render_prompt_rejects_unsupported_prompt_schema():
    probe = load_probe_module()
    tokenizer = DummyTokenizer()

    with pytest.raises(TypeError, match="Unsupported prompt schema"):
        probe.render_prompt({"prompt": UnsupportedPrompt()}, tokenizer, "prompt")


def test_extract_ground_truth_uses_first_matching_dotted_path():
    probe = load_probe_module()
    row = {
        "prompt": "p",
        "reward_model": {"ground_truth": "42"},
        "extra_info": {"answer": "wrong"},
    }

    assert (
        probe.extract_ground_truth(row, ["missing.path", "reward_model.ground_truth", "extra_info.answer"], 7) == "42"
    )


def test_extract_ground_truth_reports_columns_and_prompt_id_when_missing():
    probe = load_probe_module()

    with pytest.raises(KeyError) as exc_info:
        probe.extract_ground_truth({"prompt": "p", "answer": None}, ["reward_model.ground_truth"], 3)

    message = str(exc_info.value)
    assert "prompt_id=3" in message
    assert "columns=['answer', 'prompt']" in message


def test_call_reward_function_supports_kwargs_dicts_and_score_extraction():
    probe = load_probe_module()
    extra_info = {"difficulty": "hard"}

    def reward_fn(*, data_source, solution_str, ground_truth, extra_info):
        assert data_source == "math"
        assert solution_str == "solution"
        assert ground_truth == "gt"
        assert extra_info == {"difficulty": "hard"}
        return {"score": 1.5}

    assert probe.call_reward_function(reward_fn, "math", "solution", "gt", extra_info) == 1.5


def test_prepare_rollout_inputs_preserves_original_extra_info(monkeypatch):
    probe = load_probe_module()
    tokenizer = DummyTokenizer()
    df = pd.DataFrame(
        [
            {
                "prompt": "solve",
                "data_source": "math",
                "reward_model": {"ground_truth": "42"},
                "extra_info": {"answer": "42", "difficulty": "hard"},
            }
        ]
    )
    monkeypatch.setattr(probe.pd, "read_parquet", lambda _: df)

    rows, stats = probe.prepare_rollout_inputs(
        {
            "paths": {"data_path": "ignored.parquet"},
            "data": {
                "prompt_key": "prompt",
                "data_source_key": "data_source",
                "max_prompt_length": 1024,
                "ground_truth_extractors": ["reward_model.ground_truth"],
                "num_prompts": 1,
                "prompt_start": 0,
            },
        },
        tokenizer,
    )

    assert stats["kept_prompts"] == 1
    assert rows[0]["extra_info"] == {"answer": "42", "difficulty": "hard"}


def test_prepare_rollout_inputs_filters_overlong_before_slice_by_default(monkeypatch):
    probe = load_probe_module()
    tokenizer = DummyTokenizer()
    df = pd.DataFrame(
        [
            {"prompt": "too long", "data_source": "math", "reward_model": {"ground_truth": "skip"}},
            {"prompt": "ok", "data_source": "math", "reward_model": {"ground_truth": "keep-1"}},
            {"prompt": "fine", "data_source": "math", "reward_model": {"ground_truth": "keep-2"}},
        ]
    )
    monkeypatch.setattr(probe.pd, "read_parquet", lambda _: df)

    rows, stats = probe.prepare_rollout_inputs(
        {
            "paths": {"data_path": "ignored.parquet"},
            "data": {
                "prompt_key": "prompt",
                "data_source_key": "data_source",
                "max_prompt_length": 1,
                "ground_truth_extractors": ["reward_model.ground_truth"],
                "num_prompts": 2,
                "prompt_start": 0,
            },
        },
        tokenizer,
    )

    assert [row["original_index"] for row in rows] == [1, 2]
    assert [row["prompt_id"] for row in rows] == [0, 1]
    assert [row["prompt_id_dense"] for row in rows] == [0, 1]
    assert [row["ground_truth"] for row in rows] == ["keep-1", "keep-2"]
    assert stats["filtered_overlong_prompts"] == 1
    assert stats["selected_rows"] == 2
    assert stats["kept_prompts"] == 2


def test_prepare_rollout_inputs_can_filter_overlong_after_slice(monkeypatch):
    probe = load_probe_module()
    tokenizer = DummyTokenizer()
    df = pd.DataFrame(
        [
            {"prompt": "too long", "data_source": "math", "reward_model": {"ground_truth": "skip"}},
            {"prompt": "ok", "data_source": "math", "reward_model": {"ground_truth": "keep-1"}},
            {"prompt": "fine", "data_source": "math", "reward_model": {"ground_truth": "not-selected"}},
        ]
    )
    monkeypatch.setattr(probe.pd, "read_parquet", lambda _: df)

    rows, stats = probe.prepare_rollout_inputs(
        {
            "paths": {"data_path": "ignored.parquet"},
            "data": {
                "prompt_key": "prompt",
                "data_source_key": "data_source",
                "max_prompt_length": 1,
                "ground_truth_extractors": ["reward_model.ground_truth"],
                "num_prompts": 2,
                "prompt_start": 0,
                "filter_overlong_before_slice": False,
            },
        },
        tokenizer,
    )

    assert [row["original_index"] for row in rows] == [1]
    assert [row["prompt_id"] for row in rows] == [1]
    assert [row["prompt_id_dense"] for row in rows] == [0]
    assert [row["ground_truth"] for row in rows] == ["keep-1"]
    assert stats["filtered_overlong_prompts"] == 1
    assert stats["selected_rows"] == 2
    assert stats["kept_prompts"] == 1


def test_rollout_stop_flags_classify_response_lengths_and_stops():
    probe = load_probe_module()

    before_2048 = probe.rollout_stop_flags(response_token_len=1000, finish_reason="stop", max_tokens=10240)
    assert before_2048 == {
        "is_over_2048": False,
        "is_over_4096": False,
        "is_over_8192": False,
        "hit_10240_cap": False,
        "eos_or_stop_before_2048": True,
    }

    after_2048_stop = probe.rollout_stop_flags(response_token_len=3000, finish_reason="stop", max_tokens=10240)
    assert after_2048_stop["is_over_2048"] is True
    assert after_2048_stop["hit_10240_cap"] is False
    assert after_2048_stop["eos_or_stop_before_2048"] is False

    capped = probe.rollout_stop_flags(response_token_len=10240, finish_reason="length", max_tokens=10240)
    assert capped["is_over_8192"] is True
    assert capped["hit_10240_cap"] is True


def test_build_vllm_kwargs_includes_optional_capacity_controls():
    probe = load_probe_module()

    kwargs = probe.build_vllm_kwargs(
        checkpoint_path="/ckpt",
        tokenizer_path="/tok",
        rollout_config={
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.72,
            "max_model_len": 12288,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "max_num_seqs": "64",
            "max_num_batched_tokens": "32768",
        },
    )

    assert kwargs["model"] == "/ckpt"
    assert kwargs["tokenizer"] == "/tok"
    assert kwargs["max_num_seqs"] == 64
    assert kwargs["max_num_batched_tokens"] == 32768


def test_build_vllm_kwargs_omits_optional_capacity_controls_when_unset():
    probe = load_probe_module()

    kwargs = probe.build_vllm_kwargs(
        checkpoint_path="/ckpt",
        tokenizer_path="/tok",
        rollout_config={},
    )

    assert "max_num_seqs" not in kwargs
    assert "max_num_batched_tokens" not in kwargs


def test_validate_rollout_storage_config_rejects_disabled_token_ids():
    probe = load_probe_module()

    with pytest.raises(ValueError, match="response_token_ids"):
        probe.validate_rollout_storage_config({"storage": {"save_response_token_ids": False}})


def test_score_dataframe_truncates_by_token_ids_and_computes_flags():
    probe = load_probe_module()
    tokenizer = DummyTokenizer()
    df = pd.DataFrame(
        [
            {
                "step": 400,
                "prompt_id": 0,
                "sample_id": 0,
                "data_source": "math",
                "ground_truth": "1 2 3",
                "extra_info": {"target": "1 2 3"},
                "response_token_ids": [1, 2, 3],
            },
            {
                "step": 400,
                "prompt_id": 0,
                "sample_id": 1,
                "data_source": "math",
                "ground_truth": "1",
                "extra_info": {"target": "1"},
                "response_token_ids": [1, 2, 3],
            },
        ]
    )

    def reward_fn(*, data_source, solution_str, ground_truth, extra_info):
        assert data_source == "math"
        assert extra_info == {"target": ground_truth}
        return 1.0 if solution_str.strip() == ground_truth else -1.0

    scored = probe.score_dataframe(
        df,
        tokenizer,
        reward_fn,
        prefix_lengths=[1, 3],
        positive_threshold=0.0,
        decode_skip_special_tokens=True,
    )

    assert scored.loc[0, "reward_1"] == -1.0
    assert scored.loc[0, "reward_3"] == 1.0
    assert scored.loc[0, "first_positive_prefix_len"] == 3
    assert scored.loc[0, "reward_delta_3_minus_1"] == 2.0
    assert scored.loc[0, "reward_delta_10240_minus_2048"] == 2.0
    assert bool(scored.loc[0, "false_negative_2048_to_10240"]) is True
    assert bool(scored.loc[0, "answer_corruption_2048_to_10240"]) is False
    assert bool(scored.loc[1, "answer_corruption_2048_to_10240"]) is True


def test_torch_runtime_metadata_records_cuda_details(monkeypatch):
    probe = load_probe_module()

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            return f"GPU-{index}"

    fake_torch = types.SimpleNamespace(
        __version__="2.test",
        version=types.SimpleNamespace(cuda="12.4"),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = probe.torch_runtime_metadata()

    assert info["torch_version"] == "2.test"
    assert info["torch_cuda_version"] == "12.4"
    assert info["torch_cuda_is_available"] is True
    assert info["torch_cuda_device_count"] == 2
    assert info["torch_cuda_device_names"] == ["GPU-0", "GPU-1"]


def test_build_analysis_tables_computes_group_states_and_advantage_changes():
    probe = load_probe_module()
    scored = pd.DataFrame(
        [
            {
                "step": 400,
                "prompt_id": 0,
                "sample_id": 0,
                "response_token_len": 3000,
                "hit_max_tokens": False,
                "finish_reason": "stop",
                "stop_reason": None,
                "reward_2048": -1.0,
                "reward_10240": 1.0,
                "reward_delta_10240_minus_2048": 2.0,
                "false_negative_2048_to_10240": True,
                "answer_corruption_2048_to_10240": False,
            },
            {
                "step": 400,
                "prompt_id": 0,
                "sample_id": 1,
                "response_token_len": 1000,
                "hit_max_tokens": False,
                "finish_reason": "stop",
                "stop_reason": None,
                "reward_2048": 1.0,
                "reward_10240": -1.0,
                "reward_delta_10240_minus_2048": -2.0,
                "false_negative_2048_to_10240": False,
                "answer_corruption_2048_to_10240": True,
            },
            {
                "step": 400,
                "prompt_id": 1,
                "sample_id": 0,
                "response_token_len": 10240,
                "hit_max_tokens": True,
                "finish_reason": "length",
                "stop_reason": None,
                "reward_2048": 1.0,
                "reward_10240": -1.0,
                "reward_delta_10240_minus_2048": -2.0,
                "false_negative_2048_to_10240": False,
                "answer_corruption_2048_to_10240": True,
            },
            {
                "step": 400,
                "prompt_id": 1,
                "sample_id": 1,
                "response_token_len": 9000,
                "hit_max_tokens": False,
                "finish_reason": "abort",
                "stop_reason": "tool_stop",
                "reward_2048": 1.0,
                "reward_10240": -1.0,
                "reward_delta_10240_minus_2048": -2.0,
                "false_negative_2048_to_10240": False,
                "answer_corruption_2048_to_10240": True,
            },
        ]
    )

    tables = probe.build_analysis_tables(scored, prefix_lengths=[2048, 10240], positive_threshold=0.0, eps=1e-8)

    trajectory = tables["trajectory_summary"].iloc[0]
    assert trajectory["num_trajectories"] == 4
    assert trajectory["num_prompts"] == 2
    assert trajectory["p_false_negative"] == 0.25
    assert trajectory["p_answer_corruption"] == 0.75

    group_summary = tables["group_summary"].sort_values("prompt_id").reset_index(drop=True)
    assert group_summary.loc[0, "state_2048"] == "mixed"
    assert group_summary.loc[0, "state_10240"] == "mixed"
    assert group_summary.loc[1, "state_2048"] == "all_positive"
    assert group_summary.loc[1, "state_10240"] == "all_zero"

    transitions = {
        (row.state_2048, row.state_10240): row.count for row in tables["group_transitions"].itertuples(index=False)
    }
    assert transitions[("mixed", "mixed")] == 1
    assert transitions[("all_positive", "all_zero")] == 1

    advantage = tables["advantage_summary"].iloc[0]
    assert advantage["adv_neg_to_pos_rate"] == 0.25
    assert advantage["adv_pos_to_neg_rate"] == 0.25

    stop_reason_summary = tables["stop_reason_summary"]
    category_counts = {
        row.stop_category: row.count for row in stop_reason_summary.itertuples(index=False)
    }
    assert category_counts["stop_after_2048_before_10240"] == 1
    assert category_counts["eos_or_stop_before_2048"] == 1
    assert category_counts["hit_10240_cap"] == 1
    assert category_counts["other_stop_reason"] == 1


def test_build_analysis_tables_counts_zero_to_positive_advantage():
    probe = load_probe_module()
    scored = pd.DataFrame(
        [
            {
                "step": 400,
                "prompt_id": 0,
                "sample_id": 0,
                "response_token_len": 3000,
                "hit_max_tokens": False,
                "finish_reason": "stop",
                "stop_reason": None,
                "reward_2048": 0.0,
                "reward_10240": -1.0,
                "reward_delta_10240_minus_2048": -1.0,
            },
            {
                "step": 400,
                "prompt_id": 0,
                "sample_id": 1,
                "response_token_len": 3000,
                "hit_max_tokens": False,
                "finish_reason": "stop",
                "stop_reason": None,
                "reward_2048": 0.0,
                "reward_10240": 1.0,
                "reward_delta_10240_minus_2048": 1.0,
            },
        ]
    )

    tables = probe.build_analysis_tables(scored, prefix_lengths=[2048, 10240], positive_threshold=0.0, eps=1e-8)

    advantage = tables["advantage_summary"].iloc[0]
    assert advantage["adv_zero_to_pos_rate"] == 0.5
    assert advantage["adv_nonpos_to_pos_rate"] == 0.5
    assert advantage["adv_pos_to_nonpos_rate"] == 0.0
