# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from verl.trainer.config import AlgoConfig, ProbeCreditConfig


def test_probe_credit_defaults_are_disabled_and_canonical():
    config = ProbeCreditConfig()

    assert config.enable is False
    assert config.coef == 0.0
    assert config.rho == 0.5
    assert config.relative_positions == [0.0, 0.25, 0.5, 0.75, 0.9]
    assert config.n == 4
    assert config.temperature == 0.7
    assert config.top_p == 0.95
    assert config.top_k == -1
    assert config.max_tokens == 64
    assert config.stop == ["\n"]
    assert config.answer_prefix == "\n\nAnswer:"
    assert config.norm_probe_by_std is True
    assert config.epsilon == 1.0e-6
    assert config.probe_zero_position is True
    assert config.strict is True
    assert config.debug_dump is False
    assert config.max_concurrent_requests == 128
    assert config.request_batch_size == 512
    config.validate()


def test_algo_config_owns_independent_probe_credit_configs():
    first = AlgoConfig()
    second = AlgoConfig()

    assert isinstance(first.probe_credit, ProbeCreditConfig)
    assert first.probe_credit is not second.probe_credit


@pytest.mark.parametrize("rho", [-0.01, 1.01])
def test_probe_credit_rejects_rho_outside_unit_interval(rho):
    with pytest.raises(ValueError, match="rho"):
        ProbeCreditConfig(rho=rho).validate()


@pytest.mark.parametrize(
    "positions",
    [
        [],
        [0.25, 0.5],
        [0.0, -0.1, 0.5],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 1.0],
        [0.0, 1.1],
    ],
)
def test_probe_credit_rejects_invalid_relative_positions(positions):
    with pytest.raises(ValueError, match="relative_positions"):
        ProbeCreditConfig(relative_positions=positions).validate()


def test_probe_credit_rejects_negative_coefficient():
    with pytest.raises(ValueError, match="coef"):
        ProbeCreditConfig(coef=-0.01).validate()


@pytest.mark.parametrize(("field_name", "value"), [("n", 0), ("n", -1), ("max_tokens", 0), ("max_tokens", -1)])
def test_probe_credit_rejects_nonpositive_integer_limits(field_name, value):
    config = ProbeCreditConfig(**{field_name: value})

    with pytest.raises(ValueError, match=field_name):
        config.validate()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_concurrent_requests", 0),
        ("max_concurrent_requests", -1),
        ("request_batch_size", 0),
        ("request_batch_size", -1),
    ],
)
def test_probe_credit_rejects_nonpositive_request_limits(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        ProbeCreditConfig(**{field_name: value}).validate()


def test_probe_credit_rejects_request_batch_smaller_than_concurrency():
    with pytest.raises(ValueError, match="request_batch_size"):
        ProbeCreditConfig(max_concurrent_requests=9, request_batch_size=8).validate()
