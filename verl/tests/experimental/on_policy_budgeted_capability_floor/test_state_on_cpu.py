import pytest

from verl.experimental.capability_constraints.dual import update_projected_dual
from verl.experimental.capability_constraints.identity import (
    canonical_prompt_key,
    reference_model_fingerprint,
)


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
