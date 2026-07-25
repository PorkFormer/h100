import pytest

from verl.trainer.config import AlgoConfig, ProbeCreditConfig, ReadinessDominanceConfig


def test_readiness_dominance_defaults_are_off_and_canonical():
    config = ReadinessDominanceConfig()

    assert config.mode == "off"
    assert config.absolute_horizons == [256, 512, 1024, 2048]
    assert config.n == 4
    assert config.temperature == 0.7
    assert config.top_p == 0.95
    assert config.top_k == -1
    assert config.max_tokens == 32
    assert config.stop == ["\n"]
    assert config.answer_prefix == "\n\nAnswer:"
    assert config.strict is True
    assert config.strict_branch_margin == 1
    assert config.min_common_positions == 2
    assert config.max_concurrent_requests == 128
    assert config.request_batch_size == 512
    config.validate()


def test_algo_config_owns_independent_dominance_and_probe_credit_configs():
    first = AlgoConfig()
    second = AlgoConfig()

    assert isinstance(first.readiness_dominance, ReadinessDominanceConfig)
    assert isinstance(first.probe_credit, ProbeCreditConfig)
    assert first.readiness_dominance is not second.readiness_dominance
    assert first.readiness_dominance.absolute_horizons is not second.readiness_dominance.absolute_horizons
    assert first.readiness_dominance.stop is not second.readiness_dominance.stop
    assert first.probe_credit is not second.probe_credit
    assert first.readiness_dominance is not first.probe_credit


@pytest.mark.parametrize("mode", ["enabled", "disable", "analyze", "", None])
def test_readiness_dominance_rejects_invalid_mode(mode):
    with pytest.raises(ValueError, match="mode"):
        ReadinessDominanceConfig(mode=mode).validate()


@pytest.mark.parametrize(
    "horizons",
    [
        [],
        [0, 1],
        [-1, 1],
        [1, 1],
        [2, 1],
        [1, 2.5],
        [True, 2],
    ],
)
def test_readiness_dominance_rejects_invalid_absolute_horizons(horizons):
    with pytest.raises(ValueError, match="absolute_horizons"):
        ReadinessDominanceConfig(absolute_horizons=horizons).validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": 0}, "n"),
        ({"n": -1}, "n"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"strict_branch_margin": 0}, "strict_branch_margin"),
        ({"strict_branch_margin": 5}, "strict_branch_margin"),
        ({"min_common_positions": 0}, "min_common_positions"),
        ({"max_concurrent_requests": 0}, "max_concurrent_requests"),
        ({"request_batch_size": 0}, "request_batch_size"),
        (
            {"max_concurrent_requests": 9, "request_batch_size": 8},
            "request_batch_size",
        ),
    ],
)
def test_readiness_dominance_rejects_invalid_integer_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReadinessDominanceConfig(**kwargs).validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": -2}, "top_k"),
        ({"answer_prefix": ""}, "answer_prefix"),
        ({"stop": []}, "stop"),
        ({"stop": [""]}, "stop"),
        ({"stop": ["\n", 1]}, "stop"),
        ({"strict": False}, "strict=true"),
    ],
)
def test_readiness_dominance_rejects_invalid_sampling_or_strict_protocol(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReadinessDominanceConfig(**kwargs).validate()


@pytest.mark.parametrize("mode", ["off", "shadow", "reweight"])
def test_readiness_dominance_accepts_all_supported_modes(mode):
    ReadinessDominanceConfig(mode=mode).validate()
