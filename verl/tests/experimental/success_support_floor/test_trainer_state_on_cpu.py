import json

import pytest

from verl.experimental.success_support_floor.state import (
    SuccessSupportFloorState,
    load_state,
    scientific_config_fingerprint,
    save_state,
)
from verl.trainer.config import SuccessSupportFloorConfig


def _state():
    return SuccessSupportFloorState(
        global_step=7,
        lambda_value=0.3,
        violation_ema=0.1,
        support_update_count=4,
        last_support_step=6,
        cache_fingerprint="cache",
        config_fingerprint="config",
    )


def test_atomic_state_round_trip(tmp_path):
    path = tmp_path / "success_support_floor" / "state.json"
    save_state(path, _state())
    loaded = load_state(
        path,
        expected_global_step=7,
        expected_cache_fingerprint="cache",
        expected_config_fingerprint="config",
        lambda_max=1.0,
    )
    assert loaded == _state()
    assert json.loads(path.read_text())["schema_version"] == 1
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_global_step", 8, "global_step"),
        ("expected_cache_fingerprint", "wrong", "cache_fingerprint"),
        ("expected_config_fingerprint", "wrong", "config_fingerprint"),
        ("lambda_max", 0.2, "lambda"),
    ],
)
def test_state_mismatch_fails_closed(tmp_path, field, value, message):
    path = tmp_path / "state.json"
    save_state(path, _state())
    kwargs = dict(
        expected_global_step=7,
        expected_cache_fingerprint="cache",
        expected_config_fingerprint="config",
        lambda_max=1.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        load_state(path, **kwargs)


def test_missing_state_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_state(
            tmp_path / "missing.json",
            expected_global_step=1,
            expected_cache_fingerprint="cache",
            expected_config_fingerprint="config",
            lambda_max=1.0,
        )


def test_scientific_fingerprint_changes_for_alpha_but_not_cache_path():
    first = SuccessSupportFloorConfig(alpha=0.5, cache_path="a")
    same_science = SuccessSupportFloorConfig(alpha=0.5, cache_path="b")
    changed = SuccessSupportFloorConfig(alpha=0.75, cache_path="a")
    assert scientific_config_fingerprint(first) == scientific_config_fingerprint(same_science)
    assert scientific_config_fingerprint(first) != scientific_config_fingerprint(changed)
