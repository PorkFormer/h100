import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from verl.experimental.capability_constraints.identity import canonical_prompt_key
from verl.experimental.on_policy_budgeted_capability_floor.cache import (
    CacheExpectations,
    CapabilityFloorCache,
    build_floor_rows,
    write_cache,
)
from tools.on_policy_budgeted_capability_floor import build_floor_cache as builder


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "algorithm": "on_policy_budgeted_capability_floor",
        "reference_model_id": "base",
        "reference_model_hash": "a" * 64,
        "reference_budget": 2048,
        "base_rollouts_per_prompt": 8,
        "support_threshold": 2,
        "reference_tolerance_count": 1,
        "prefix_reward_field": "prefix_reward_2048",
        "tokenizer_fingerprint": "tok",
        "chat_template_fingerprint": "tmpl",
        "prompt_manifest_fingerprint": "b" * 64,
        "rollout_fingerprint": "c" * 64,
        "score_fingerprint": "d" * 64,
        "verifier_fingerprint": "e" * 64,
        "created_at": "2026-08-03T00:00:00+00:00",
        "source_git_commit": "f4282a317f1ffdbd52ab0c83e3ddac987ef0d72d",
        "prompt_count": 3,
    }


def _source_rows():
    prompts = []
    rollouts = []
    scores = []
    for prompt_id, success_count in enumerate((2, 1, 8)):
        tokens = [10, 20 + prompt_id]
        prompts.append(
            {
                "prompt_id": prompt_id,
                "original_dataset_index": 100 + prompt_id,
                "prompt_hash": f"prompt-{prompt_id}",
                "prompt_token_ids": tokens,
                "prompt_token_count": len(tokens),
            }
        )
        for rollout_index in range(8):
            identity = {
                "model_id": "base",
                "prompt_id": prompt_id,
                "rollout_index": rollout_index,
            }
            rollouts.append(
                identity
                | {
                    "prompt_hash": f"prompt-{prompt_id}",
                    "prompt_token_ids": tokens,
                }
            )
            scores.append(
                identity
                | {
                    "prompt_hash": f"prompt-{prompt_id}",
                    "prefix_reward_2048": rollout_index < success_count,
                    "prefix_error_2048": None,
                    # These must be ignored by OBCF floor construction.
                    "full_reward": False,
                    "finish_reason": "length",
                }
            )
    return prompts, rollouts, scores


def _rows():
    prompts, rollouts, scores = _source_rows()
    return build_floor_rows(
        prompts=prompts,
        rollouts=rollouts,
        scores=scores,
        model_id="base",
        tokenizer_fingerprint="tok",
        chat_template_fingerprint="tmpl",
        reference_budget=2048,
        base_rollouts_per_prompt=8,
        support_threshold=2,
        reference_tolerance_count=1,
    )


def _write(tmp_path: Path) -> CapabilityFloorCache:
    write_cache(tmp_path, _manifest(), _rows())
    return CapabilityFloorCache.load(tmp_path, _expectations())


def _expectations(**overrides) -> CacheExpectations:
    values = dict(
        reference_budget=2048,
        base_rollouts_per_prompt=8,
        support_threshold=2,
        reference_tolerance_count=1,
        tokenizer_fingerprint="tok",
        chat_template_fingerprint="tmpl",
        verifier_fingerprint="e" * 64,
    )
    values.update(overrides)
    return CacheExpectations(**values)


def test_prefix_only_threshold_and_floor_derivation():
    rows = _rows()

    assert [row["prompt_id"] for row in rows] == [0, 2]
    assert rows[0]["base_prefix_success_count"] == 2
    assert rows[0]["q_reference"] == pytest.approx(2 / 8)
    assert rows[0]["floor_count"] == 1
    assert rows[0]["capability_floor"] == pytest.approx(1 / 8)
    assert rows[1]["floor_count"] == 7
    assert rows[1]["capability_floor"] == pytest.approx(7 / 8)


@pytest.mark.parametrize("missing_from", ["rollouts", "scores"])
def test_missing_identity_fails_closed(missing_from):
    prompts, rollouts, scores = _source_rows()
    if missing_from == "rollouts":
        rollouts.pop()
    else:
        scores.pop()
    with pytest.raises(ValueError, match="identit|count"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
        )


@pytest.mark.parametrize("duplicate_in", ["rollouts", "scores"])
def test_duplicate_identity_fails_closed(duplicate_in):
    prompts, rollouts, scores = _source_rows()
    (rollouts if duplicate_in == "rollouts" else scores).append(
        dict((rollouts if duplicate_in == "rollouts" else scores)[0])
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
        )


def test_missing_prefix_reward_and_prefix_error_fail_closed():
    prompts, rollouts, scores = _source_rows()
    scores[0].pop("prefix_reward_2048")
    with pytest.raises(ValueError, match="prefix_reward_2048"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
        )

    prompts, rollouts, scores = _source_rows()
    scores[0].pop("prefix_error_2048")
    with pytest.raises(ValueError, match="prefix_error_2048"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
        )


def test_cross_artifact_prompt_identity_mismatch_fails_closed():
    prompts, rollouts, scores = _source_rows()
    rollouts[0]["prompt_token_ids"] = [999]
    with pytest.raises(ValueError, match="token"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
        )

    prompts, rollouts, scores = _source_rows()
    scores[0]["prompt_hash"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
        )

    prompts, rollouts, scores = _source_rows()
    scores[0]["prefix_error_2048"] = "verifier failed"
    with pytest.raises(ValueError, match="prefix_error_2048"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
        )


def test_score_artifact_must_attest_to_verifier_and_source_commit():
    prompts, rollouts, scores = _source_rows()
    for score in scores:
        score["verifier_fingerprint"] = "e" * 64
        score["source_git_commit"] = "f4282a317f1ffdbd52ab0c83e3ddac987ef0d72d"
    scores[0]["verifier_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="verifier_fingerprint"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
            verifier_fingerprint="e" * 64,
            source_git_commit="f4282a317f1ffdbd52ab0c83e3ddac987ef0d72d",
        )
    scores[0]["verifier_fingerprint"] = "e" * 64
    scores[0]["source_git_commit"] = "0" * 40
    with pytest.raises(ValueError, match="source_git_commit"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
            verifier_fingerprint="e" * 64,
            source_git_commit="f4282a317f1ffdbd52ab0c83e3ddac987ef0d72d",
        )


def test_prompt_artifact_provenance_is_required_by_strict_builder_contract():
    prompts, rollouts, scores = _source_rows()
    with pytest.raises(ValueError, match="tokenizer/chat-template"):
        build_floor_rows(
            prompts=prompts,
            rollouts=rollouts,
            scores=scores,
            model_id="base",
            tokenizer_fingerprint="tok",
            chat_template_fingerprint="tmpl",
            reference_budget=2048,
            base_rollouts_per_prompt=8,
            support_threshold=2,
            reference_tolerance_count=1,
            require_prompt_provenance=True,
        )


def test_builder_supports_quoted_and_shell_expanded_parquet_globs(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"scores-{index}.parquet"
        pq.write_table(pa.Table.from_pylist([{"row": index}]), path)
        paths.append(path)

    quoted = builder._rows([tmp_path / "scores-*.parquet"])
    expanded = builder._rows(list(reversed(paths)))

    assert quoted == expanded == [{"row": 0}, {"row": 1}]
    assert builder._file_hash([tmp_path / "scores-*.parquet"]) == builder._file_hash(paths)


def test_builder_validates_local_tokenizer_and_template_fingerprints():
    tokenizer = type(
        "Tokenizer",
        (),
        {
            "get_vocab": lambda self: {"a": 0},
            "special_tokens_map": {"eos_token": "<eos>"},
            "chat_template": "{{ messages }}",
        },
    )()
    actual = builder._validate_tokenizer_fingerprints(tokenizer, None, None)
    assert builder._validate_tokenizer_fingerprints(tokenizer, *actual) == actual
    with pytest.raises(ValueError, match="tokenizer fingerprint"):
        builder._validate_tokenizer_fingerprints(tokenizer, "bad", actual[1])


def test_builder_converts_hash_bound_legacy_artifact_attestation(tmp_path):
    prompts, rollouts, scores = _source_rows()
    for rows in (prompts, rollouts, scores):
        for row in rows:
            row["config_fingerprint"] = "a" * 64
    input_paths = {}
    for name, rows in (("prompts", prompts), ("rollouts", rollouts), ("scores", scores)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rows))
        input_paths[name] = [path]
    attestation = {
        "schema_version": 1,
        "passed": True,
        "config_fingerprint": "a" * 64,
        "source_git_commit": "f4282a317f1ffdbd52ab0c83e3ddac987ef0d72d",
        "verifier_fingerprint": "e" * 64,
        "tokenizer_fingerprint": "tok",
        "chat_template_fingerprint": "tmpl",
        "prompt_manifest_fingerprint": builder._file_hash(input_paths["prompts"]),
        "rollout_fingerprint": builder._file_hash(input_paths["rollouts"]),
        "score_fingerprint": builder._file_hash(input_paths["scores"]),
    }
    converted_prompts, converted_scores = builder._apply_legacy_attestation(
        prompts=prompts,
        rollouts=rollouts,
        scores=scores,
        attestation=attestation,
        tokenizer_fingerprint="tok",
        chat_template_fingerprint="tmpl",
        verifier_fingerprint="e" * 64,
        source_git_commit="f4282a317f1ffdbd52ab0c83e3ddac987ef0d72d",
        artifact_fingerprints={
            "prompt_manifest_fingerprint": builder._file_hash(input_paths["prompts"]),
            "rollout_fingerprint": builder._file_hash(input_paths["rollouts"]),
            "score_fingerprint": builder._file_hash(input_paths["scores"]),
        },
    )

    assert converted_prompts[0]["tokenizer_fingerprint"] == "tok"
    assert converted_scores[0]["verifier_fingerprint"] == "e" * 64
    assert "verifier_fingerprint" not in scores[0]


def test_round_trip_validates_audit_and_supports_key_lookup(tmp_path):
    cache = _write(tmp_path)
    key = canonical_prompt_key("tok", "tmpl", [10, 20])

    assert cache.get(key)["capability_floor"] == pytest.approx(1 / 8)
    assert cache.get("missing") is None
    assert cache.manifest["protected_prompt_count"] == 2
    assert cache.audit_report["passed"] is True
    assert cache.audit_report["base_prefix_success_count_histogram"] == {"1": 1, "2": 1, "8": 1}
    assert cache.audit_report["cache_fingerprint"] == cache.fingerprint
    assert set(path.name for path in tmp_path.iterdir()) == {
        "manifest.json",
        "prompts.parquet",
        "hashes.json",
        "audit_report.json",
    }


def test_hash_corruption_fails_closed(tmp_path):
    _write(tmp_path)
    with (tmp_path / "prompts.parquet").open("ab") as stream:
        stream.write(b"corrupt")

    with pytest.raises(ValueError, match="hash"):
        CapabilityFloorCache.load(tmp_path, _expectations())


def test_audit_report_must_pass_even_when_hashes_are_consistent(tmp_path):
    _write(tmp_path)
    report = json.loads((tmp_path / "audit_report.json").read_text())
    report["passed"] = False
    (tmp_path / "audit_report.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    hashes = json.loads((tmp_path / "hashes.json").read_text())
    import hashlib

    hashes["files"]["audit_report.json"] = hashlib.sha256(
        (tmp_path / "audit_report.json").read_bytes()
    ).hexdigest()
    (tmp_path / "hashes.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="passed"):
        CapabilityFloorCache.load(tmp_path, _expectations())


def test_audit_success_histogram_must_match_each_protected_success_bin(tmp_path):
    _write(tmp_path)
    report = json.loads((tmp_path / "audit_report.json").read_text())
    report["base_prefix_success_count_histogram"] = {"1": 1, "3": 1, "7": 1}
    (tmp_path / "audit_report.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    hashes = json.loads((tmp_path / "hashes.json").read_text())
    import hashlib

    hashes["files"]["audit_report.json"] = hashlib.sha256(
        (tmp_path / "audit_report.json").read_bytes()
    ).hexdigest()
    (tmp_path / "hashes.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="success-count histogram"):
        CapabilityFloorCache.load(tmp_path, _expectations())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reference_budget", 1024),
        ("base_rollouts_per_prompt", 4),
        ("support_threshold", 3),
        ("reference_tolerance_count", 0),
        ("tokenizer_fingerprint", "other"),
        ("chat_template_fingerprint", "other"),
        ("verifier_fingerprint", "f" * 64),
    ],
)
def test_expectation_mismatch_fails_closed(tmp_path, field, value):
    _write(tmp_path)
    with pytest.raises(ValueError, match=field):
        CapabilityFloorCache.load(tmp_path, _expectations(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("base_rollout_count", 7, "base_rollout_count"),
        ("base_prefix_success_count", 1, "support_threshold"),
        ("q_reference", 0.5, "q_reference"),
        ("floor_count", 0, "floor_count"),
        ("capability_floor", 0.5, "capability_floor"),
    ],
)
def test_malformed_prompt_rows_fail_closed(tmp_path, field, value, match):
    rows = _rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        write_cache(tmp_path, _manifest(), rows)


def test_strict_parquet_schema_rejects_extra_or_nullable_contract(tmp_path):
    _write(tmp_path)
    table = pq.read_table(tmp_path / "prompts.parquet").append_column(
        "unexpected", pa.array([1, 2], type=pa.int32())
    )
    pq.write_table(table, tmp_path / "prompts.parquet")
    with pytest.raises(ValueError, match="schema|hash"):
        CapabilityFloorCache.load(tmp_path, _expectations())
