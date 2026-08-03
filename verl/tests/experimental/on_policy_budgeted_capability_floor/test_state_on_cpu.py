import pytest

from verl.experimental.capability_constraints.dual import update_projected_dual
from verl.experimental.capability_constraints.identity import (
    canonical_prompt_key,
    reference_model_fingerprint,
)
from verl.experimental.on_policy_budgeted_capability_floor.state import (
    OnPolicyBudgetedCapabilityFloorState,
    load_state,
    save_state,
    scientific_config_fingerprint,
)
from verl.trainer.config import OnPolicyBudgetedCapabilityFloorConfig


def test_first_observation_initializes_ema_without_smoothing():
    state = update_projected_dual(
        lambda_value=0.2,
        violation_ema=99.0,
        ema_initialized=False,
        observed_constraint=0.3,
        delta=0.05,
        dual_lr=1.0,
        ema_beta=0.9,
        lambda_max=10.0,
    )

    assert state.violation_ema == pytest.approx(0.3)
    assert state.lambda_value == pytest.approx(0.45)
    assert state.ema_initialized is True


@pytest.mark.parametrize(
    ("lambda_value", "observed_constraint", "expected"),
    [(0.1, 0.0, 0.0), (0.9, 1.0, 1.0)],
)
def test_dual_update_projects_to_interval(lambda_value, observed_constraint, expected):
    state = update_projected_dual(
        lambda_value=lambda_value,
        violation_ema=observed_constraint,
        ema_initialized=True,
        observed_constraint=observed_constraint,
        delta=0.5,
        dual_lr=10.0,
        ema_beta=0.0,
        lambda_max=1.0,
    )

    assert state.lambda_value == pytest.approx(expected)


@pytest.mark.parametrize(
    "override",
    [
        {"lambda_value": float("nan")},
        {"violation_ema": float("inf")},
        {"observed_constraint": float("-inf")},
        {"delta": float("nan")},
        {"dual_lr": float("inf")},
        {"ema_beta": float("nan")},
        {"lambda_max": float("inf")},
    ],
)
def test_dual_update_rejects_non_finite_values(override):
    kwargs = dict(
        lambda_value=0.0,
        violation_ema=0.0,
        ema_initialized=False,
        observed_constraint=0.1,
        delta=0.05,
        dual_lr=0.01,
        ema_beta=0.9,
        lambda_max=10.0,
    )
    kwargs.update(override)

    with pytest.raises(ValueError, match="finite"):
        update_projected_dual(**kwargs)


def test_prompt_key_is_deterministic_and_token_sensitive():
    first = canonical_prompt_key("tok", "template", iter([1, 2, 3]))

    assert first == canonical_prompt_key("tok", "template", [1, 2, 3])
    assert first != canonical_prompt_key("tok", "template", [1, 2, 4])


def test_reference_model_fingerprint_hashes_only_model_weight_bytes(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"first")
    first = reference_model_fingerprint(model_dir)
    (model_dir / "config.json").write_text("unrelated")
    assert reference_model_fingerprint(model_dir) == first

    weight.write_bytes(b"second")
    assert reference_model_fingerprint(model_dir) != first


def _obcf_state():
    return OnPolicyBudgetedCapabilityFloorState(
        global_step=7,
        lambda_value=0.3,
        violation_ema=0.1,
        ema_initialized=True,
        constraint_observation_count=4,
        last_constraint_step=7,
        cache_fingerprint="cache",
        config_fingerprint="config",
    )


def test_obcf_state_atomic_exact_round_trip(tmp_path):
    path = tmp_path / "on_policy_budgeted_capability_floor" / "state.json"
    save_state(path, _obcf_state())
    loaded = load_state(
        path,
        expected_global_step=7,
        expected_cache_fingerprint="cache",
        expected_config_fingerprint="config",
        lambda_max=1.0,
    )
    assert loaded == _obcf_state()
    payload = __import__("json").loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["lambda"] == pytest.approx(0.3)
    assert "lambda_value" not in payload
    assert not path.with_suffix(".json.tmp").exists()


def test_obcf_state_fingerprint_allows_cache_relocation_only():
    first = OnPolicyBudgetedCapabilityFloorConfig(cache_path="a", delta=0.05)
    relocated = OnPolicyBudgetedCapabilityFloorConfig(cache_path="b", delta=0.05)
    changed = OnPolicyBudgetedCapabilityFloorConfig(cache_path="a", delta=0.1)
    assert scientific_config_fingerprint(first) == scientific_config_fingerprint(relocated)
    assert scientific_config_fingerprint(first) != scientific_config_fingerprint(changed)


def test_obcf_state_rejects_inconsistent_ema_observation_count(tmp_path):
    state = OnPolicyBudgetedCapabilityFloorState(
        global_step=0,
        lambda_value=0.0,
        violation_ema=0.0,
        ema_initialized=False,
        constraint_observation_count=1,
        last_constraint_step=0,
        cache_fingerprint="cache",
        config_fingerprint="config",
    )
    path = tmp_path / "state.json"
    save_state(path, state)
    with pytest.raises(ValueError, match="initialization"):
        load_state(
            path,
            expected_global_step=0,
            expected_cache_fingerprint="cache",
            expected_config_fingerprint="config",
            lambda_max=1.0,
        )
