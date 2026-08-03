import json
from pathlib import Path

import pytest
import torch

from verl.experimental.success_support_floor.cache import (
    CacheExpectations,
    SuccessSupportCache,
    canonical_prompt_key,
    reference_model_fingerprint,
    witness_is_eligible,
    write_cache,
)
from verl.experimental.success_support_floor.logprobs import sequence_logprobs


def _manifest():
    return {
        "schema_version": 1,
        "algorithm": "budgeted_success_support_floor",
        "reference_model_id": "base",
        "reference_model_hash": "a" * 64,
        "reference_budget": 8,
        "base_rollouts_per_prompt": 3,
        "support_threshold": 2,
        "tokenizer_fingerprint": "tok",
        "chat_template_fingerprint": "tmpl",
        "prompt_manifest_fingerprint": "prompts",
        "verifier_fingerprint": "verify",
        "logprob_temperature": 1.0,
        "logprob_convention": "response-token-sum",
        "include_eos": True,
        "created_at": "2026-08-03T00:00:00Z",
        "source_git_commit": "798c84a",
    }


def _rows():
    prompts = []
    witnesses = []
    for prompt_id in range(3):
        tokens = [10, prompt_id + 20]
        key = canonical_prompt_key("tok", "tmpl", tokens)
        prompts.append(
            {
                "prompt_key": key,
                "prompt_id": prompt_id,
                "original_dataset_index": prompt_id,
                "prompt_hash": f"hash-{prompt_id}",
                "prompt_token_ids": tokens,
                "prompt_token_count": len(tokens),
                "base_rollout_count": 3,
                "eligible_success_count": 2,
                "q_reference": 2 / 3,
            }
        )
        for witness_id in range(2):
            witnesses.append(
                {
                    "prompt_key": key,
                    "witness_id": witness_id,
                    "source_rollout_index": witness_id,
                    "response_token_ids": [30 + witness_id, 2],
                    "response_token_count": 2,
                    "reference_seq_logprob": -2.5 - witness_id,
                    "reference_mean_logprob": (-2.5 - witness_id) / 2,
                    "finish_reason": "eos",
                    "full_reward": True,
                    "prefix_reward_reference_budget": True,
                    "response_hash": f"response-{prompt_id}-{witness_id}",
                }
            )
    return prompts, witnesses


def _write(tmp_path: Path) -> SuccessSupportCache:
    prompts, witnesses = _rows()
    write_cache(tmp_path, _manifest(), prompts, witnesses)
    return SuccessSupportCache.load(
        tmp_path,
        CacheExpectations(
            reference_budget=8,
            support_threshold=2,
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            logprob_temperature=1.0,
        ),
    )


def test_eligibility_is_strict_about_budget_finish_rewards_and_errors():
    valid = dict(
        full_reward=True,
        prefix_reward=True,
        response_token_count=8,
        reference_budget=8,
        hit_token_cap=False,
        finish_reason="eos",
        generation_error=None,
        verifier_error=None,
    )
    assert witness_is_eligible(**valid)
    for field, value in [
        ("full_reward", False),
        ("prefix_reward", False),
        ("response_token_count", 9),
        ("hit_token_cap", True),
        ("finish_reason", "length"),
        ("generation_error", "boom"),
        ("verifier_error", "boom"),
    ]:
        invalid = valid | {field: value}
        assert not witness_is_eligible(**invalid)


def test_prompt_key_is_stable_and_template_sensitive():
    assert canonical_prompt_key("tok", "a", [1, 2]) == canonical_prompt_key("tok", "a", [1, 2])
    assert canonical_prompt_key("tok", "a", [1, 2]) != canonical_prompt_key("tok", "b", [1, 2])


def test_reference_model_fingerprint_hashes_weight_content(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"first")
    first = reference_model_fingerprint(model_dir)
    weight.write_bytes(b"second")
    second = reference_model_fingerprint(model_dir)

    assert len(first) == 64
    assert first != second


def test_round_trip_sets_counts_hashes_and_samples_unique_prompts(tmp_path):
    cache = _write(tmp_path)
    first = cache.sample(batch_size=3, seed=7, global_step=5, support_update_count=2)
    second = cache.sample(batch_size=3, seed=7, global_step=5, support_update_count=2)

    assert first == second
    assert len({row["prompt_key"] for row in first}) == 3
    assert cache.manifest["protected_prompt_count"] == 3
    assert cache.manifest["witness_count"] == 6
    assert cache.fingerprint
    assert json.loads((tmp_path / "hashes.json").read_text())["cache_fingerprint"] == cache.fingerprint


def test_sampler_fails_instead_of_replacement(tmp_path):
    cache = _write(tmp_path)
    with pytest.raises(ValueError, match="without replacement"):
        cache.sample(batch_size=4, seed=1, global_step=1, support_update_count=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_budget", 9),
        ("support_threshold", 3),
        ("tokenizer_fingerprint", "wrong"),
        ("chat_template_fingerprint", "wrong"),
        ("logprob_temperature", 0.5),
    ],
)
def test_metadata_mismatch_fails_closed(tmp_path, field, value):
    prompts, witnesses = _rows()
    write_cache(tmp_path, _manifest(), prompts, witnesses)
    kwargs = dict(
        reference_budget=8,
        support_threshold=2,
        tokenizer_fingerprint="tok",
        chat_template_fingerprint="tmpl",
        logprob_temperature=1.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        SuccessSupportCache.load(tmp_path, CacheExpectations(**kwargs))


def test_corrupt_reference_logprob_fails_closed(tmp_path):
    prompts, witnesses = _rows()
    witnesses[0]["reference_seq_logprob"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_cache(tmp_path, _manifest(), prompts, witnesses)


def test_manifest_model_hash_and_prompt_counts_fail_closed(tmp_path):
    prompts, witnesses = _rows()
    invalid_hash = _manifest() | {"reference_model_hash": "path-plus-config"}
    with pytest.raises(ValueError, match="reference_model_hash"):
        write_cache(tmp_path / "hash", invalid_hash, prompts, witnesses)

    prompts[0]["q_reference"] = 0.1
    with pytest.raises(ValueError, match="q_reference"):
        write_cache(tmp_path / "counts", _manifest(), prompts, witnesses)


def test_teacher_forcing_scores_only_response_tokens_in_fp32():
    class UniformModel:
        def __call__(self, *, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            logits = torch.zeros((*input_ids.shape, 5), dtype=torch.bfloat16)
            return type("Output", (), {"logits": logits})()

    values = sequence_logprobs(
        UniformModel(),
        [([1, 2], [3, 4]), ([2], [1])],
        pad_token_id=0,
        temperature=1.0,
        device="cpu",
    )

    assert values == pytest.approx([-2 * torch.log(torch.tensor(5.0)).item(), -torch.log(torch.tensor(5.0)).item()])
