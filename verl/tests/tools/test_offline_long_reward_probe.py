import importlib.util
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

    def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=True):
        self.chat_template_calls.append(
            {"prompt": prompt, "tokenize": tokenize, "add_generation_prompt": add_generation_prompt}
        )
        return "chat rendered"

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


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

    def reward_fn(*, data_source, solution_str, ground_truth, extra_info):
        assert data_source == "math"
        assert solution_str == "solution"
        assert ground_truth == "gt"
        assert extra_info == {}
        return {"score": 1.5}

    assert probe.call_reward_function(reward_fn, "math", "solution", "gt") == 1.5


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
                "response_token_ids": [1, 2, 3],
            },
            {
                "step": 400,
                "prompt_id": 0,
                "sample_id": 1,
                "data_source": "math",
                "ground_truth": "1",
                "response_token_ids": [1, 2, 3],
            },
        ]
    )

    def reward_fn(text, ground_truth):
        return 1.0 if text.strip() == ground_truth else -1.0

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
