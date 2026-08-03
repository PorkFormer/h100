import pytest

from verl.trainer.config import SuccessSupportFloorConfig


def test_defaults_are_inert_and_valid():
    config = SuccessSupportFloorConfig()
    config.validate()

    assert config.mode == "off"
    assert config.cache_path is None
    assert config.lambda_init == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "active"),
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("delta", -0.1),
        ("delta", float("nan")),
        ("reference_budget", 0),
        ("support_threshold", 0),
        ("constraint_batch_size", -1),
        ("update_interval", 0),
        ("lambda_init", -0.1),
        ("lambda_max", -0.1),
        ("dual_lr", -0.1),
        ("dual_lr", float("inf")),
        ("dual_ema_beta", 1.0),
    ],
)
def test_invalid_values_fail_closed(field, value):
    config = SuccessSupportFloorConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        config.validate()


def test_lambda_max_must_cover_initial_value():
    config = SuccessSupportFloorConfig(lambda_init=2.0, lambda_max=1.0)

    with pytest.raises(ValueError, match="lambda_max"):
        config.validate()
