"""CPU regression tests for grouped vLLM Probe generation."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

pytest.importorskip("vllm")

from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

sys.modules.pop("verl.workers.rollout.utils", None)

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer  # noqa: E402


class _FakeConfig:
    max_model_len = 4096
    prompt_length = 2048
    response_length = 2048
    enable_rollout_routing_replay = False
    mtp = None

    def get(self, _key, default=None):
        return default


class _FakeModelConfig:
    processor = None
    lora_rank = 0
    lora = {}


class _FakeEngine:
    def __init__(self, indexes=(0, 1), *, completions=None, no_yields=False):
        self.indexes = tuple(indexes)
        self.completions = completions
        self.no_yields = no_yields
        self.captured_sampling_params = None

    @staticmethod
    def _completion(branch_id):
        return CompletionOutput(
            index=branch_id,
            text=f"branch-{branch_id}",
            token_ids=[100 + branch_id],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="stop",
        )

    def generate(self, *, prompt, sampling_params, request_id, lora_request=None, priority=0):
        self.captured_sampling_params = sampling_params

        async def _stream():
            if self.no_yields:
                return
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=[1, 2, 3],
                prompt_logprobs=None,
                outputs=(
                    list(self.completions)
                    if self.completions is not None
                    else [self._completion(branch_id) for branch_id in self.indexes]
                ),
                finished=True,
            )

        return _stream()


def _make_server(engine):
    server = object.__new__(vLLMHttpServer)
    server.config = _FakeConfig()
    server.model_config = _FakeModelConfig()
    server.engine = engine
    server.global_steps = 7
    return server


def _generate_grouped(engine, *, n=2, request_id="probe-test", sampling_overrides=None):
    sampling_params = {
        "n": n,
        "seed": 123,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "max_tokens": 64,
    }
    if sampling_overrides:
        sampling_params.update(sampling_overrides)
    outputs = asyncio.run(
        _make_server(engine).generate_grouped(
            prompt_ids=[1, 2, 3],
            sampling_params=sampling_params,
            request_id=request_id,
        )
    )
    return outputs, sampling_params


def test_generate_grouped_forces_final_only():
    engine = _FakeEngine()

    outputs, _ = _generate_grouped(engine)

    assert engine.captured_sampling_params.n == 2
    assert engine.captured_sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert len(outputs) == 2


def test_generate_grouped_final_only_cannot_be_overridden_by_caller():
    engine = _FakeEngine()

    outputs, sampling_params = _generate_grouped(
        engine,
        sampling_overrides={"output_kind": RequestOutputKind.CUMULATIVE},
    )

    assert engine.captured_sampling_params.output_kind == RequestOutputKind.FINAL_ONLY
    assert sampling_params["output_kind"] == RequestOutputKind.CUMULATIVE
    assert len(outputs) == 2


def test_generate_grouped_uses_completion_indexes_and_sorts_branches():
    engine = _FakeEngine(indexes=(1, 0))

    outputs, _ = _generate_grouped(engine)

    assert [output.extra_fields["branch_id"] for output in outputs] == [0, 1]
    assert outputs[0].extra_fields["text"] == "branch-0"
    assert outputs[0].token_ids == [100]
    assert outputs[1].extra_fields["text"] == "branch-1"
    assert outputs[1].token_ids == [101]


def test_generate_grouped_rejects_missing_branch_at_server_boundary():
    engine = _FakeEngine(indexes=(0,))

    with pytest.raises(RuntimeError) as exc_info:
        _generate_grouped(engine, request_id="probe-missing")

    message = str(exc_info.value)
    assert "request_id='probe-missing'" in message
    assert "requested_n=2" in message
    assert "received=[0]" in message
    assert "missing=[1]" in message


def test_generate_grouped_rejects_duplicate_branch():
    engine = _FakeEngine(indexes=(0, 0))

    with pytest.raises(RuntimeError, match="duplicate") as exc_info:
        _generate_grouped(engine, request_id="probe-duplicate")

    message = str(exc_info.value)
    assert "request_id='probe-duplicate'" in message
    assert "requested_n=2" in message
    assert "duplicates=[0]" in message


def test_generate_grouped_rejects_out_of_range_branch():
    engine = _FakeEngine(indexes=(0, 2))

    with pytest.raises(RuntimeError, match="out-of-range") as exc_info:
        _generate_grouped(engine, request_id="probe-out-of-range")

    message = str(exc_info.value)
    assert "request_id='probe-out-of-range'" in message
    assert "requested_n=2" in message
    assert "received=[0, 2]" in message
    assert "missing=[1]" in message
    assert "extra=[2]" in message


@pytest.mark.parametrize(
    ("completion", "expected_message"),
    [
        (
            SimpleNamespace(text="missing-index", token_ids=[100], finish_reason="stop"),
            "missing completion.index",
        ),
        (
            CompletionOutput(
                index="bad",
                text="bad-index",
                token_ids=[100],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            ),
            "invalid completion.index",
        ),
    ],
)
def test_generate_grouped_rejects_missing_or_invalid_completion_index(completion, expected_message):
    engine = _FakeEngine(completions=[completion])

    with pytest.raises(RuntimeError, match=expected_message):
        _generate_grouped(engine, request_id="probe-invalid-index")


def test_generate_grouped_accepts_single_indexed_branch():
    engine = _FakeEngine(indexes=(0,))

    outputs, _ = _generate_grouped(engine, n=1)

    assert len(outputs) == 1
    assert outputs[0].extra_fields["branch_id"] == 0
    assert outputs[0].extra_fields["text"] == "branch-0"
    assert outputs[0].token_ids == [100]


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(_FakeEngine(no_yields=True), id="no-yields"),
        pytest.param(_FakeEngine(indexes=()), id="empty-outputs"),
    ],
)
def test_generate_grouped_rejects_missing_final_outputs(engine):
    with pytest.raises(RuntimeError) as exc_info:
        _generate_grouped(engine, request_id="probe-empty")

    message = str(exc_info.value)
    assert "request_id='probe-empty'" in message
    assert "requested_n=2" in message
    assert "received=[]" in message
    assert "missing=[0, 1]" in message


def test_generate_keeps_ordinary_streaming_semantics():
    engine = _FakeEngine(indexes=(0,))
    server = _make_server(engine)

    output = asyncio.run(
        server.generate(
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 8},
            request_id="ordinary-test",
        )
    )

    assert engine.captured_sampling_params.n == 1
    assert engine.captured_sampling_params.output_kind == RequestOutputKind.CUMULATIVE
    assert output.token_ids == [100]
    assert output.stop_reason == "completed"
    assert output.extra_fields == {"global_steps": 7}
