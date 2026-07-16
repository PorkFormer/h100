import argparse
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "offline_long_reward_probe" / "forced_answer_probe.py"
CONFIG_PATH = MODULE_PATH.with_name("probe_config.yaml")
VIE_CONFIG_PATH = MODULE_PATH.with_name("probe_config_vie.yaml")
MATH_DAPO_PATH = MODULE_PATH.parents[2] / "verl" / "utils" / "reward_score" / "math_dapo.py"
REWARD_SCORE_INIT = MATH_DAPO_PATH.with_name("__init__.py")


def load_probe_module(name="forced_answer_probe_test"):
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_config(tmp_path, **forced_overrides):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    forced = {
        "horizons": [0, 2, 4],
        "tail_offsets": [1],
        "cues": [{"name": "primary", "text": " cue"}],
        "n": 2,
        "max_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": False,
        "seed": 7,
        "batch_size_requests": 8,
        "stability_threshold": 0.5,
        "eps": 1e-8,
    }
    forced.update(forced_overrides)
    return {
        "paths": {
            "output_dir": str(tmp_path / "out"),
            "checkpoint_root": str(tmp_path),
            "checkpoint_template": str(checkpoint),
        },
        "rollout": {"max_model_len": 32, "enable_prefix_caching": True},
        "scoring": {
            "positive_threshold": 0.0,
            "reward_function": {"import_path": "unused.reward"},
        },
        "storage": {
            "skip_existing": True,
            "save_response_text": True,
            "save_response_token_ids": True,
            "compression": None,
        },
        "forced_answer": forced,
    }


def source_frame(rows=None):
    rows = rows or [
        {
            "step": 4,
            "prompt_id": 0,
            "sample_id": 0,
            "original_index": 9,
            "data_source": "math",
            "ground_truth": "42",
            "extra_info": {"difficulty": "easy"},
            "prompt_text": "question",
            "prompt_token_ids": [10, 11],
            "response_text": "reasoning",
            "response_token_ids": [20, 21, 22],
            "response_token_len": 3,
            "reward_10240": 1.0,
        }
    ]
    return pd.DataFrame(rows)


class DummyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, add_special_tokens=False, return_attention_mask=False):
        self.calls.append(text)
        return {"input_ids": [100 + index for index, _ in enumerate(text.split())]}


class FakeCompletion:
    def __init__(self, text="42", token_ids=None):
        self.text = text
        self.token_ids = token_ids or [90]
        self.finish_reason = "stop"
        self.stop_reason = None


class FakeRequestOutput:
    def __init__(self, outputs=None):
        self.outputs = outputs if outputs is not None else [FakeCompletion(), FakeCompletion()]


def install_fake_vllm(monkeypatch, llm_class):
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TokensPrompt(dict):
        def __init__(self, *, prompt_token_ids):
            super().__init__(prompt_token_ids=prompt_token_ids)

    package = types.ModuleType("vllm")
    package.LLM = llm_class
    package.SamplingParams = SamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = TokensPrompt
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


def test_config_validation_normalizes_horizons_and_checks_cues(tmp_path):
    probe = load_probe_module()
    config = make_config(tmp_path, horizons=[4, 0, 2, 2], tail_offsets=[2, 1, 2])
    normalized = probe.normalize_forced_answer_config(config)
    assert normalized["forced_answer"]["horizons"] == [0, 2, 4]
    assert normalized["forced_answer"]["tail_offsets"] == [1, 2]
    bad = make_config(
        tmp_path,
        cues=[{"name": "same", "text": "a"}, {"name": "same", "text": "b"}],
    )
    with pytest.raises(ValueError, match="unique"):
        probe.normalize_forced_answer_config(bad)


def test_default_cue_and_real_math_dapo_reward_adapter(monkeypatch):
    probe = load_probe_module()
    config = probe.offline_probe.load_config(CONFIG_PATH)
    repo_root = MODULE_PATH.parents[2]
    verl_package = types.ModuleType("verl")
    verl_package.__path__ = [str(repo_root / "verl")]
    utils_package = types.ModuleType("verl.utils")
    utils_package.__path__ = [str(repo_root / "verl" / "utils")]
    monkeypatch.setitem(sys.modules, "verl", verl_package)
    monkeypatch.setitem(sys.modules, "verl.utils", utils_package)

    import_utils_path = repo_root / "verl" / "utils" / "import_utils.py"
    import_utils_spec = importlib.util.spec_from_file_location("verl.utils.import_utils", import_utils_path)
    import_utils = importlib.util.module_from_spec(import_utils_spec)
    monkeypatch.setitem(sys.modules, "verl.utils.import_utils", import_utils)
    import_utils_spec.loader.exec_module(import_utils)

    package_name = "verl.utils.reward_score"
    spec = importlib.util.spec_from_file_location(
        package_name,
        REWARD_SCORE_INIT,
        submodule_search_locations=[str(REWARD_SCORE_INIT.parent)],
    )
    reward_score = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, reward_score)
    spec.loader.exec_module(reward_score)
    reward_fn = probe.offline_probe.load_reward_function(config)

    scores = [
        probe.offline_probe.call_reward_function(reward_fn, "math", completion, "42", {})
        for completion in ["Answer: 42", r"Answer: \boxed{42}", "42"]
    ]

    assert scores == [1.0, 1.0, -1.0]
    assert Path(sys.modules[f"{package_name}.math_dapo"].__file__) == MATH_DAPO_PATH
    assert config["forced_answer"]["cues"][0] == {
        "name": "final_answer",
        "text": "\nProvide only the final answer in this exact format: Answer: <final answer>",
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n": 0}, "n must be positive"),
        ({"max_tokens": 0}, "max_tokens must be positive"),
        ({"batch_size_requests": 0}, "batch_size_requests must be positive"),
        ({"stability_threshold": 1.1}, "stability_threshold"),
        ({"tail_offsets": [0]}, "positive integers"),
    ],
)
def test_config_validation_rejects_invalid_ranges(tmp_path, override, message):
    probe = load_probe_module()
    with pytest.raises(ValueError, match=message):
        probe.normalize_forced_answer_config(make_config(tmp_path, **override))


def test_source_validation_reports_all_missing_columns():
    probe = load_probe_module()
    with pytest.raises(KeyError) as exc_info:
        probe.validate_source_dataframe(pd.DataFrame({"step": [4]}), 4)
    message = str(exc_info.value)
    assert "prompt_id" in message
    assert "response_token_ids" in message
    assert "reward_10240" in message


def test_source_validation_rejects_cli_step_mismatch():
    probe = load_probe_module()
    with pytest.raises(ValueError, match="CLI step 5"):
        probe.validate_source_dataframe(source_frame(), 5)


def test_source_validation_reports_response_token_length_and_trajectory_key():
    probe = load_probe_module()
    source = source_frame()
    source.loc[0, "response_token_len"] = 2

    with pytest.raises(ValueError) as exc_info:
        probe.validate_source_dataframe(source, 4)

    message = str(exc_info.value)
    assert "step=4, prompt_id=0, sample_id=0" in message
    assert "response_token_len=2" in message
    assert "parsed_response_token_ids=3" in message


def test_select_source_by_first_unique_prompts_keeps_all_rollouts_and_order(tmp_path):
    probe = load_probe_module()
    rows = []
    for prompt_id in [2, 0, 1]:
        for sample_id in range(8):
            row = source_frame().iloc[0].to_dict()
            row.update({"prompt_id": prompt_id, "sample_id": sample_id})
            rows.append(row)
    source = source_frame(rows)
    args = probe.build_parser().parse_args(
        ["--config", str(CONFIG_PATH), "--step", "4", "--mode", "analyze", "--max-prompts", "2"]
    )
    config, overrides = probe.apply_cli_overrides(make_config(tmp_path), args)

    selected, stats = probe.select_source_dataframe(source, config["forced_answer"])

    assert list(selected.index) == list(range(16))
    assert selected["prompt_id"].tolist() == [2] * 8 + [0] * 8
    assert stats == {
        "source_trajectories": 24,
        "source_prompts": 3,
        "selected_trajectories": 16,
        "selected_prompts": 2,
    }
    assert overrides["max_prompts"] == 2


def test_prompt_and_trajectory_limits_are_mutually_exclusive(tmp_path):
    probe = load_probe_module()
    with pytest.raises(ValueError, match="mutually exclusive"):
        probe.normalize_forced_answer_config(make_config(tmp_path, max_prompts=2, max_trajectories=4))

    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(
            [
                "--config",
                str(CONFIG_PATH),
                "--step",
                "4",
                "--mode",
                "analyze",
                "--max-prompts",
                "2",
                "--max-trajectories",
                "4",
            ]
        )


def test_build_probe_points_deduplicates_fixed_and_adds_one_terminal():
    probe = load_probe_module()
    points = probe.build_probe_points(3, [0, 2, 3, 4], [1, 2])
    assert points == [
        {"horizon": 0, "kind": "fixed", "is_carry_forward": False},
        {"horizon": 1, "kind": "preterminal", "is_carry_forward": False},
        {"horizon": 2, "kind": "fixed", "is_carry_forward": False},
        {"horizon": 3, "kind": "terminal", "is_carry_forward": True},
    ]


def test_probe_token_ids_are_concatenated_without_retokenization():
    probe = load_probe_module()
    assert probe.build_probe_token_ids([1, 2], [3, 4, 5], 2, [6, 7]) == [1, 2, 3, 4, 6, 7]


def test_aggregate_excludes_errors_and_overflow_from_value_denominator():
    probe = load_probe_module()
    raw = pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": 2,
                "kind": "fixed",
                "cue": "a",
                "status": "generated",
                "probe_success": 1.0,
                "probe_reward": 2.0,
            },
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 1,
                "horizon": 2,
                "kind": "fixed",
                "cue": "a",
                "status": "scoring_error",
                "probe_success": None,
                "probe_reward": None,
            },
            {
                "step": 4,
                "prompt_id": 1,
                "sample_id": 0,
                "horizon": 2,
                "kind": "fixed",
                "cue": "a",
                "status": "context_overflow",
                "probe_success": None,
                "probe_reward": None,
            },
        ]
    )
    row = probe.aggregate_probe_values(raw, [2], ["a"]).iloc[0]
    assert row["value"] == 1.0
    assert row["num_valid_branches"] == 1
    assert row["num_trajectory_opportunities"] == 3
    assert row["trajectory_error_rate"] == pytest.approx(1 / 3)
    assert row["trajectory_overflow_rate"] == pytest.approx(1 / 3)


def test_carry_forward_populates_fixed_grid_for_every_cue():
    probe = load_probe_module()
    raw = pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": 3,
                "kind": "terminal",
                "cue": "<terminal>",
                "status": "generated",
                "probe_success": 1.0,
                "probe_reward": 1.0,
                "response_token_len": 3,
                "is_carry_forward": True,
            }
        ]
    )
    curve = probe.aggregate_probe_values(raw, [2, 4, 8], ["a", "b"])
    assert set(zip(curve["horizon"], curve["cue"], strict=True)) == {(4, "a"), (4, "b"), (8, "a"), (8, "b")}
    assert curve["value"].eq(1.0).all()


def test_prompt_cluster_standard_error_averages_rollouts_first():
    probe = load_probe_module()
    raw = pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": prompt,
                "sample_id": sample,
                "horizon": 2,
                "kind": "fixed",
                "cue": "a",
                "status": "generated",
                "probe_success": value,
                "probe_reward": value,
            }
            for prompt, values in [(0, [0.0, 1.0]), (1, [1.0])]
            for sample, value in enumerate(values)
        ]
    )
    row = probe.aggregate_probe_values(raw, [2], ["a"]).iloc[0]
    assert row["value"] == 0.75
    assert row["v_se_prompt_cluster"] == pytest.approx(0.25)


def test_curve_uses_equal_branch_trajectory_prompt_weighting():
    probe = load_probe_module()
    generated = [
        {
            "step": 4,
            "prompt_id": 0,
            "sample_id": 0,
            "horizon": 4,
            "kind": "fixed",
            "cue": "a",
            "status": "generated",
            "probe_success": 0.0,
            "probe_reward": 0.0,
            "response_token_len": 8,
            "is_carry_forward": False,
        }
        for _ in range(8)
    ]
    carry = {
        "step": 4,
        "prompt_id": 0,
        "sample_id": 1,
        "horizon": 3,
        "kind": "terminal",
        "cue": "<terminal>",
        "status": "generated",
        "probe_success": 1.0,
        "probe_reward": 1.0,
        "response_token_len": 3,
        "is_carry_forward": True,
    }

    row = probe.aggregate_probe_values(pd.DataFrame([*generated, carry]), [4], ["a"]).iloc[0]

    assert row["value"] == 0.5
    assert row["probe_reward_mean"] == 0.5
    assert row["probe_reward_std"] == 0.0
    assert row["num_valid_branches"] == 9
    assert row["num_valid_trajectories"] == 2
    assert row["num_prompts"] == 1


def test_trajectory_values_record_generated_and_terminal_carry_provenance(tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[2, 4], n=1))
    raw = pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": 2,
                "kind": "fixed",
                "cue": "primary",
                "status": "generated",
                "probe_success": 0.0,
                "probe_reward": -1.0,
                "response_token_len": 3,
                "is_carry_forward": False,
            },
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": 3,
                "kind": "terminal",
                "cue": "<terminal>",
                "status": "generated",
                "probe_success": 1.0,
                "probe_reward": 1.0,
                "response_token_len": 3,
                "is_carry_forward": True,
            },
        ]
    )

    values = probe._trajectory_probe_values(raw, config).set_index("horizon")

    assert values.loc[2, "value_source"] == "generated"
    assert values.loc[4, "value_source"] == "terminal_carry"


def _taxonomy_inputs(probe, tmp_path):
    config = probe.normalize_forced_answer_config(
        make_config(tmp_path, horizons=[0, 2048, 4096, 8192], tail_offsets=[1], n=1)
    )
    source = source_frame(
        [
            {**source_frame().iloc[0].to_dict(), "prompt_id": 0, "response_token_len": 9000, "reward_10240": 1.0},
            {**source_frame().iloc[0].to_dict(), "prompt_id": 1, "response_token_len": 9000, "reward_10240": -1.0},
        ]
    )
    rows = []
    curves = {0: [0.0, 0.0, 1.0, 1.0], 1: [0.0, 1.0, 1.0, 1.0]}
    for prompt_id, curve in curves.items():
        for horizon, value in zip(config["forced_answer"]["horizons"], curve, strict=True):
            rows.append(
                {
                    "step": 4,
                    "prompt_id": prompt_id,
                    "sample_id": 0,
                    "horizon": horizon,
                    "kind": "fixed",
                    "cue": "primary",
                    "value": value,
                    "value_source": "generated",
                    "complete": True,
                }
            )
        rows.append(
            {
                "step": 4,
                "prompt_id": prompt_id,
                "sample_id": 0,
                "horizon": 8999,
                "kind": "preterminal",
                "cue": "primary",
                "value": 0.0,
                "value_source": "generated",
                "complete": True,
            }
        )
    return config, source, pd.DataFrame(rows)


def test_taxonomy_delayed_solve_and_terminal_wrong(tmp_path):
    probe = load_probe_module()
    config, source, values = _taxonomy_inputs(probe, tmp_path)
    taxonomy = probe.build_taxonomy(source, values, config).set_index("prompt_id")
    assert bool(taxonomy.loc[0, "delayed_solve_candidate"]) is True
    assert bool(taxonomy.loc[1, "early_recoverable_but_terminal_wrong"]) is True
    assert bool(taxonomy.loc[0, "terminal_only_candidate"]) is False


def test_taxonomy_marks_missing_key_points_unknown(tmp_path):
    probe = load_probe_module()
    config, source, values = _taxonomy_inputs(probe, tmp_path)
    values = values.loc[~((values["prompt_id"] == 0) & (values["horizon"] == 2048))]
    taxonomy = probe.build_taxonomy(source.iloc[:1], values, config).iloc[0]
    assert pd.isna(taxonomy["early_recoverable"])
    assert pd.isna(taxonomy["delayed_solve_candidate"])
    assert pd.isna(taxonomy["terminal_only_candidate"])


def test_taxonomy_terminal_carry_cannot_create_stable_crossing(tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(
        make_config(tmp_path, horizons=[0, 2048, 4096], tail_offsets=[1], n=1)
    )
    row = source_frame().iloc[0].to_dict()
    row.update({"response_token_len": 3000, "reward_10240": 1.0})
    source = source_frame([row])
    values = pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": horizon,
                "kind": "fixed",
                "cue": "primary",
                "value": value,
                "value_source": value_source,
                "complete": True,
            }
            for horizon, value, value_source in [
                (0, 0.0, "generated"),
                (2048, 0.0, "generated"),
                (4096, 1.0, "terminal_carry"),
            ]
        ]
    )

    taxonomy = probe.build_taxonomy(source, values, config).iloc[0]

    assert pd.isna(taxonomy["stable_horizon"])
    assert bool(taxonomy["delayed_solve_candidate"]) is False


def test_taxonomy_terminal_only_and_delayed_solve_are_mutually_exclusive(tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(
        make_config(tmp_path, horizons=[0, 2048, 4096], tail_offsets=[1], n=1)
    )
    base = source_frame().iloc[0].to_dict()
    source = source_frame(
        [
            {**base, "prompt_id": 0, "response_token_len": 5000, "reward_10240": 1.0},
            {**base, "prompt_id": 1, "response_token_len": 5000, "reward_10240": 1.0},
        ]
    )
    rows = []
    for prompt_id, curve, tail in [(0, [0.0, 0.0, 1.0], 1.0), (1, [0.0, 0.0, 0.0], 0.0)]:
        for horizon, value in zip([0, 2048, 4096], curve, strict=True):
            rows.append(
                {
                    "step": 4,
                    "prompt_id": prompt_id,
                    "sample_id": 0,
                    "horizon": horizon,
                    "kind": "fixed",
                    "cue": "primary",
                    "value": value,
                    "value_source": "generated",
                    "complete": True,
                }
            )
        rows.append(
            {
                "step": 4,
                "prompt_id": prompt_id,
                "sample_id": 0,
                "horizon": 4999,
                "kind": "preterminal",
                "cue": "primary",
                "value": tail,
                "value_source": "generated",
                "complete": True,
            }
        )

    taxonomy = probe.build_taxonomy(source, pd.DataFrame(rows), config).set_index("prompt_id")

    assert bool(taxonomy.loc[0, "delayed_solve_candidate"]) is True
    assert bool(taxonomy.loc[0, "terminal_only_candidate"]) is False
    assert bool(taxonomy.loc[1, "delayed_solve_candidate"]) is False
    assert bool(taxonomy.loc[1, "terminal_only_candidate"]) is True


def test_taxonomy_summary_counts_nullable_booleans(tmp_path):
    probe = load_probe_module()
    config, source, values = _taxonomy_inputs(probe, tmp_path)
    taxonomy = probe.build_taxonomy(source, values, config)
    taxonomy.loc[0, "unstable_diagnostic"] = pd.NA
    row = probe.taxonomy_summary(taxonomy).set_index("label").loc["unstable_diagnostic"]
    assert row["true_count"] + row["false_count"] + row["unknown_count"] == 2
    assert row["unknown_count"] == 1


def test_hc_advantage_uses_population_std_and_zero_sign_disagrees(tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0, 2], n=1))
    source = source_frame(
        [
            {**source_frame().iloc[0].to_dict(), "sample_id": 0, "response_token_len": 3, "reward_10240": -1.0},
            {**source_frame().iloc[0].to_dict(), "sample_id": 1, "response_token_len": 3, "reward_10240": 1.0},
        ]
    )
    values = pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": sample,
                "horizon": horizon,
                "kind": "fixed",
                "cue": "primary",
                "value": value,
            }
            for sample, curve in [(0, [0.0, 0.0]), (1, [0.0, 1.0])]
            for horizon, value in zip([0, 2], curve, strict=True)
        ]
    )
    taxonomy = pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": sample,
                "stable_horizon": None,
                **{label: False for label in probe.TAXONOMY_LABELS},
            }
            for sample in [0, 1]
        ]
    )
    advantage = probe.build_counterfactual_advantage(source, values, taxonomy, config).sort_values("sample_id")
    assert advantage["hc_advantage"].tolist() == [-1.0, 1.0]
    assert advantage["terminal_advantage"].tolist() == [-1.0, 1.0]
    # Degenerate HC groups are assigned exactly zero; zero versus nonzero is a disagreement.
    values.loc[values["horizon"] == 2, "value"] = 0.0
    degenerate = probe.build_counterfactual_advantage(source, values, taxonomy, config)
    assert degenerate["hc_degenerate"].all()
    assert degenerate["hc_sign"].eq(0).all()
    assert degenerate["sign_disagreement"].all()


def test_generate_uses_exact_tokens_and_completion_only_verifier(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path))
    source_path = probe.source_scored_path(config, 4)
    source_path.parent.mkdir(parents=True)
    source_frame().to_parquet(source_path, index=False)
    tokenizer = DummyTokenizer()
    verifier_inputs = []

    class LLM:
        instances = []

        def __init__(self, **kwargs):
            self.calls = []
            self.__class__.instances.append(self)

        def generate(self, prompts, params):
            self.calls.extend(prompts)
            return [FakeRequestOutput() for _ in prompts]

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (tokenizer, "/tok"))
    monkeypatch.setattr(
        probe.offline_probe,
        "load_reward_function",
        lambda config: lambda **kwargs: verifier_inputs.append(kwargs["solution_str"]) or 1.0,
    )
    probe.run_generate(config, 4, force=True)

    prompt_ids = [call["prompt_token_ids"] for call in LLM.instances[0].calls]
    assert [10, 11, 20, 21, 100] in prompt_ids
    assert verifier_inputs and set(verifier_inputs) == {"42"}
    raw = pd.read_parquet(probe.raw_dir(config, 4) / "raw.parquet")
    assert len(raw.loc[raw["is_carry_forward"]]) == 1


def test_generate_fallback_encoding_is_cached_by_prompt(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0], n=1))
    rows = []
    for sample_id in [0, 1]:
        row = source_frame().iloc[0].to_dict()
        row.update({"sample_id": sample_id, "prompt_token_ids": None})
        rows.append(row)
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame(rows).to_parquet(path, index=False)
    tokenizer = DummyTokenizer()

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            return [FakeRequestOutput([FakeCompletion()]) for _ in prompts]

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (tokenizer, "/tok"))
    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: lambda **kwargs: 1.0)
    probe.run_generate(config, 4, force=True)
    metadata = json.loads((probe.raw_dir(config, 4) / "metadata.json").read_text())
    assert metadata["prompt_token_fallback_trajectories"] == 2
    assert metadata["prompt_token_fallback_unique_texts"] == 1
    assert tokenizer.calls.count("question") == 1


def test_generate_multiple_cues_are_separate_batches(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(
        make_config(
            tmp_path,
            horizons=[0],
            n=1,
            cues=[{"name": "a", "text": " one"}, {"name": "b", "text": " two"}],
        )
    )
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame().to_parquet(path, index=False)

    class LLM:
        calls = []

        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            self.__class__.calls.append(prompts)
            return [FakeRequestOutput([FakeCompletion()]) for _ in prompts]

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (DummyTokenizer(), "/tok"))
    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: lambda **kwargs: 1.0)
    probe.run_generate(config, 4, force=True)
    assert len(LLM.calls) == 2
    raw = pd.read_parquet(probe.raw_dir(config, 4) / "raw.parquet")
    assert set(raw["cue"]) == {"a", "b"}


def test_failed_batch_retries_each_request_to_isolate_error():
    probe = load_probe_module()
    requests = [
        {"tokens_prompt": {"prompt_token_ids": [1]}},
        {"tokens_prompt": {"prompt_token_ids": [2]}},
    ]

    class LLM:
        def generate(self, prompts, params):
            if len(prompts) > 1:
                raise RuntimeError("batch failed")
            if prompts[0]["prompt_token_ids"] == [2]:
                raise RuntimeError("bad request")
            return [FakeRequestOutput()]

    results = list(probe._generate_batches(LLM(), requests, object(), batch_size=2))
    assert results[0][1] is not None and results[0][2] is None
    assert results[1][1] is None and "bad request" in str(results[1][2])


def test_partial_completion_counts_only_missing_branches(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0], n=4))
    rows = []
    for sample_id in range(3):
        row = source_frame().iloc[0].to_dict()
        row["sample_id"] = sample_id
        rows.append(row)
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame(rows).to_parquet(path, index=False)

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            outputs = [FakeRequestOutput([FakeCompletion() for _ in range(3)])]
            outputs.extend(FakeRequestOutput([FakeCompletion() for _ in range(4)]) for _ in prompts[1:])
            return outputs

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (DummyTokenizer(), "/tok"))
    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: lambda **kwargs: 1.0)

    probe.run_generate(config, 4, force=True)

    metadata = json.loads((probe.raw_dir(config, 4) / "metadata.json").read_text())
    assert metadata["generation_error_requests"] == 1
    assert metadata["generation_error_probes"] == 1
    assert metadata["equivalent_branch_opportunities"] == 24
    assert metadata["equivalent_branch_errors"] == 1
    assert metadata["stage_error_rate"] == pytest.approx(1 / 24)


def test_request_generation_failure_records_every_missing_branch(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0], n=4))
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame().to_parquet(path, index=False)

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            if len(prompts) > 1:
                raise RuntimeError("retry requests individually")
            if len(prompts[0]["prompt_token_ids"]) == 3:
                raise RuntimeError("request failed")
            return [FakeRequestOutput([FakeCompletion() for _ in range(4)])]

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (DummyTokenizer(), "/tok"))
    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: lambda **kwargs: 1.0)

    with pytest.raises(RuntimeError, match="stage error rate 50.00%"):
        probe.run_generate(config, 4, force=True)

    raw = pd.read_parquet(probe.raw_dir(config, 4) / "raw.parquet")
    errors = raw.loc[raw["status"].eq("generation_error")]
    assert sorted(errors["probe_branch_id"].tolist()) == [0, 1, 2, 3]
    metadata = json.loads((probe.raw_dir(config, 4) / "metadata.json").read_text())
    assert metadata["generation_error_requests"] == 1
    assert metadata["generation_error_probes"] == 4
    assert metadata["equivalent_branch_opportunities"] == 8
    assert metadata["equivalent_branch_errors"] == 4


def test_scoring_error_keeps_completion_and_is_written_before_failure(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0], n=1))
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame().to_parquet(path, index=False)

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            return [FakeRequestOutput([FakeCompletion(text="kept completion")]) for _ in prompts]

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (DummyTokenizer(), "/tok"))

    def broken_reward(**kwargs):
        raise ValueError("verifier failed")

    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: broken_reward)
    with pytest.raises(RuntimeError, match=r"no valid generated probes.*scoring=2"):
        probe.run_generate(config, 4, force=True)
    raw = pd.read_parquet(probe.raw_dir(config, 4) / "raw.parquet")
    assert raw.loc[0, "completion_text"] == "kept completion"
    assert raw.loc[0, "status"] == "scoring_error"
    metadata = json.loads((probe.raw_dir(config, 4) / "metadata.json").read_text())
    assert metadata["stage_error_rate"] == 1.0


def test_all_context_overflow_writes_diagnostics_then_fails_closed(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = make_config(tmp_path, horizons=[0], n=1)
    config["rollout"]["max_model_len"] = 2
    config = probe.normalize_forced_answer_config(config)
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame().to_parquet(path, index=False)

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            pytest.fail("overflow request reached vLLM")

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (DummyTokenizer(), "/tok"))
    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: lambda **kwargs: 1.0)
    with pytest.raises(RuntimeError, match=r"no valid generated probes.*overflow=2.*generation=0.*scoring=0"):
        probe.run_generate(config, 4, force=True)
    raw = pd.read_parquet(probe.raw_dir(config, 4) / "raw.parquet")
    assert raw.loc[0, "status"] == "context_overflow"
    assert raw.loc[0, "probe_branch_id"] == -1
    metadata = json.loads((probe.raw_dir(config, 4) / "metadata.json").read_text())
    assert metadata["equivalent_branch_opportunities"] == 0
    assert metadata["stage_error_rate"] == 0.0


def test_max_prompts_metadata_counts_source_and_selected_rows(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0], n=1, max_prompts=2))
    rows = []
    for prompt_id in range(3):
        for sample_id in range(8):
            row = source_frame().iloc[0].to_dict()
            row.update(
                {
                    "prompt_id": prompt_id,
                    "sample_id": sample_id,
                    "prompt_token_ids": [10, prompt_id],
                }
            )
            rows.append(row)
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame(rows).to_parquet(path, index=False)

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            return [FakeRequestOutput([FakeCompletion()]) for _ in prompts]

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (DummyTokenizer(), "/tok"))
    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: lambda **kwargs: 1.0)

    probe.run_generate(config, 4, force=True)

    metadata = json.loads((probe.raw_dir(config, 4) / "metadata.json").read_text())
    metadata_names = ["source_trajectories", "source_prompts", "selected_trajectories", "selected_prompts"]
    assert {name: metadata[name] for name in metadata_names} == {
        "source_trajectories": 24,
        "source_prompts": 3,
        "selected_trajectories": 16,
        "selected_prompts": 2,
    }

    probe.run_analyze(config, 4, force=True)
    analysis_metadata = json.loads((probe.analysis_dir(config, 4) / "metadata.json").read_text())
    assert {name: analysis_metadata[name] for name in metadata_names} == {
        "source_trajectories": 24,
        "source_prompts": 3,
        "selected_trajectories": 16,
        "selected_prompts": 2,
    }


def test_max_trajectories_keeps_original_source_order(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0], n=1, max_trajectories=1))
    first = source_frame().iloc[0].to_dict()
    second = {**first, "prompt_id": 1, "prompt_token_ids": [99]}
    path = probe.source_scored_path(config, 4)
    path.parent.mkdir(parents=True)
    source_frame([first, second]).to_parquet(path, index=False)

    class LLM:
        prompts = []

        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            self.__class__.prompts.extend(prompts)
            return [FakeRequestOutput([FakeCompletion()]) for _ in prompts]

    install_fake_vllm(monkeypatch, LLM)
    monkeypatch.setattr(probe.offline_probe, "load_tokenizer", lambda config, checkpoint: (DummyTokenizer(), "/tok"))
    monkeypatch.setattr(probe.offline_probe, "load_reward_function", lambda config: lambda **kwargs: 1.0)
    probe.run_generate(config, 4, force=True)
    assert len(LLM.prompts) == 2
    assert all(prompt["prompt_token_ids"][:2] == [10, 11] for prompt in LLM.prompts)
    assert all(99 not in prompt["prompt_token_ids"] for prompt in LLM.prompts)


def test_skip_existing_does_not_load_tokenizer_or_vllm(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path))
    output = probe.raw_dir(config, 4) / "raw.parquet"
    output.parent.mkdir(parents=True)
    output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        probe.offline_probe,
        "load_tokenizer",
        lambda *args: pytest.fail("skip path loaded tokenizer"),
    )
    sys.modules.pop("vllm", None)
    probe.run_generate(config, 4, force=False)
    assert "vllm" not in sys.modules
    metadata = json.loads((output.parent / "metadata.json").read_text())
    assert metadata["skipped"] is True


def test_analyze_is_lazy_and_writes_outputs_without_model_imports(monkeypatch, tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[0, 4], n=1))
    source_path = probe.source_scored_path(config, 4)
    source_path.parent.mkdir(parents=True)
    source_frame().to_parquet(source_path, index=False)
    raw_path = probe.raw_dir(config, 4) / "raw.parquet"
    raw_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": 0,
                "kind": "fixed",
                "cue": "primary",
                "status": "generated",
                "probe_success": 0.0,
                "probe_reward": -1.0,
                "response_token_len": 3,
                "is_carry_forward": False,
            },
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": 3,
                "kind": "terminal",
                "cue": "<terminal>",
                "status": "generated",
                "probe_success": 1.0,
                "probe_reward": 1.0,
                "response_token_len": 3,
                "is_carry_forward": True,
            },
            *[
                {
                    "step": 4,
                    "prompt_id": 0,
                    "sample_id": 0,
                    "horizon": 1,
                    "kind": "preterminal",
                    "cue": "primary",
                    "status": status,
                    "probe_success": None,
                    "probe_reward": None,
                    "response_token_len": 3,
                    "is_carry_forward": False,
                }
                for status in ["context_overflow", "generation_error", "scoring_error"]
            ],
        ]
    ).to_parquet(raw_path, index=False)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("transformers"):
            raise AssertionError(f"analysis imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    probe.run_analyze(config, 4, force=True)
    assert (probe.analysis_dir(config, 4) / "probe_curve.csv").exists()
    assert (probe.analysis_dir(config, 4) / "counterfactual_advantage.parquet").exists()
    metadata = json.loads((probe.analysis_dir(config, 4) / "metadata.json").read_text())
    metadata_names = ["source_trajectories", "source_prompts", "selected_trajectories", "selected_prompts"]
    assert {name: metadata[name] for name in metadata_names} == {
        "source_trajectories": 1,
        "source_prompts": 1,
        "selected_trajectories": 1,
        "selected_prompts": 1,
    }
    assert {
        name: metadata[name]
        for name in [
            "valid_generated_probes",
            "context_overflow_probes",
            "generation_error_probes",
            "scoring_error_probes",
        ]
    } == {
        "valid_generated_probes": 1,
        "context_overflow_probes": 1,
        "generation_error_probes": 1,
        "scoring_error_probes": 1,
    }


def test_analyze_with_only_terminal_carry_fails_closed(tmp_path):
    probe = load_probe_module()
    config = probe.normalize_forced_answer_config(make_config(tmp_path, horizons=[4], n=1))
    source_path = probe.source_scored_path(config, 4)
    source_path.parent.mkdir(parents=True)
    source_frame().to_parquet(source_path, index=False)
    raw_path = probe.raw_dir(config, 4) / "raw.parquet"
    raw_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "step": 4,
                "prompt_id": 0,
                "sample_id": 0,
                "horizon": 3,
                "kind": "terminal",
                "cue": "<terminal>",
                "status": "generated",
                "probe_success": 1.0,
                "probe_reward": 1.0,
                "response_token_len": 3,
                "is_carry_forward": True,
            }
        ]
    ).to_parquet(raw_path, index=False)

    with pytest.raises(RuntimeError, match=r"no valid generated probes.*overflow=0.*generation=0.*scoring=0"):
        probe.run_analyze(config, 4, force=True)

    assert not (probe.analysis_dir(config, 4) / "probe_curve.csv").exists()
    metadata = json.loads((probe.analysis_dir(config, 4) / "metadata.json").read_text())
    assert metadata["valid_generated_probes"] == 0


def test_main_all_calls_generate_before_analyze(monkeypatch, tmp_path):
    probe = load_probe_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("forced_answer: {}", encoding="utf-8")
    calls = []
    config = make_config(tmp_path)
    monkeypatch.setattr(probe.offline_probe, "load_config", lambda path: config)
    monkeypatch.setattr(probe, "run_generate", lambda *args: calls.append("generate"))
    monkeypatch.setattr(probe, "run_analyze", lambda *args: calls.append("analyze"))
    probe.main(["--config", str(config_path), "--step", "4", "--mode", "all"])
    assert calls == ["generate", "analyze"]


def test_cli_overrides_are_written_into_final_config(tmp_path):
    probe = load_probe_module()
    args = argparse.Namespace(
        output_dir="/override",
        source_scored="/source.parquet",
        max_prompts=2,
        max_trajectories=3,
        probe_n=5,
        checkpoint_path="/checkpoint",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        probe.apply_cli_overrides(make_config(tmp_path), args)

    args.max_trajectories = None
    config, overrides = probe.apply_cli_overrides(make_config(tmp_path), args)
    assert config["paths"]["output_dir"] == "/override"
    assert config["forced_answer"]["source_scored"] == "/source.parquet"
    assert config["forced_answer"]["max_prompts"] == 2
    assert config["forced_answer"]["n"] == 5
    assert set(overrides) == {"output_dir", "source_scored", "max_prompts", "probe_n", "checkpoint_path"}


def test_default_and_vie_configs_have_forced_answer_probe_defaults():
    probe = load_probe_module()
    default = probe.offline_probe.load_config(CONFIG_PATH)
    vie = probe.offline_probe.load_config(VIE_CONFIG_PATH)
    expected_forced = {
        "horizons": [0, 1024, 2048, 3072, 4096, 6144],
        "tail_offsets": [1024, 512, 256],
        "n": 4,
        "max_tokens": 128,
        "top_p": 0.95,
        "seed": 42,
        "batch_size_requests": 64,
        "stability_threshold": 0.75,
        "max_prompts": None,
    }
    for name, value in expected_forced.items():
        assert default["forced_answer"][name] == value
        assert vie["forced_answer"][name] == value
    assert vie["paths"] == default["paths"]
    assert vie["rollout"]["tensor_parallel_size"] == 4
    assert vie["rollout"]["gpu_memory_utilization"] == 0.72
    assert vie["rollout"]["max_num_seqs"] == 64
    assert vie["rollout"]["max_num_batched_tokens"] == 32768
    assert vie["rollout"]["enable_prefix_caching"] is True
