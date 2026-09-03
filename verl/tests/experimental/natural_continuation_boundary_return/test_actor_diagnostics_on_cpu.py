from __future__ import annotations

import os
import tempfile
from multiprocessing import get_context

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.actor_diagnostics import ActorDiagnosticsAccumulator
from verl.workers.utils.padding import left_right_2_no_padding


def _diagnostic_config(enable=True):
    return {
        "enable": enable,
        "numerical_log_ratio_min": -20.0,
        "numerical_log_ratio_max": 20.0,
        "entropy_bucket_size": 256,
    }


def _batch(rows: int, *, rank_offset: int = 0):
    prompt = torch.tensor([[10, 11]] * rows)
    response = torch.tensor([[20, 21, 22, 23]] * rows)
    attention = torch.ones(rows, 6, dtype=torch.long)
    data = TensorDict(
        {
            "input_ids": torch.cat((prompt, response), dim=-1),
            "prompts": prompt,
            "responses": response,
            "attention_mask": attention,
            "response_mask": torch.ones(rows, 4, dtype=torch.bool),
            "position_ids": torch.arange(6).repeat(rows, 1),
            "old_log_probs": torch.full((rows, 4), -1.0),
            "advantages": torch.tensor([[1.0, 1.0, -1.0, -1.0]] * rows),
            "old_policy_entropies": torch.tensor([[1.0, 2.0, 3.0, 4.0]] * rows),
            "boundary_hit_cap": torch.tensor([rank_offset % 2 == 0] * rows),
            "boundary_eligible": torch.tensor([rank_offset % 2 == 0] * rows),
            "boundary_applied": torch.tensor([rank_offset % 2 == 0] * rows),
            "boundary_changed": torch.tensor([rank_offset % 2 == 0] * rows),
            "boundary_recovered": torch.tensor([rank_offset % 2 == 0] * rows),
            "boundary_regressed": torch.zeros(rows, dtype=torch.bool),
            "boundary_task_delta": torch.tensor([float(1 - rank_offset)] * rows),
            "boundary_group_unlocked": torch.tensor([rank_offset % 2 == 0] * rows),
        },
        batch_size=rows,
    )
    tu.assign_non_tensor(data, clip_ratio_low=0.2, clip_ratio_high=0.28)
    return left_right_2_no_padding(data)


def _model_output(data, response_log_probs):
    sequences = []
    for row in response_log_probs:
        # no_padding_2_padding extracts [prompt_last, response[:-1]] at indices 1:5
        sequences.append(torch.tensor([0.0, *row, 0.0]))
    return {"log_probs": torch.nested.as_nested_tensor(sequences, layout=torch.jagged)}


def _accumulate(rows: int, rank_offset: int = 0):
    data = _batch(rows, rank_offset=rank_offset)
    current = [[-0.9, -1.3, -1.0, -0.5]] * rows
    accumulator = ActorDiagnosticsAccumulator(_diagnostic_config())
    accumulator.accumulate(_model_output(data, current), data)
    return accumulator


def test_actor_diagnostics_reports_raw_effective_clip_ess_and_entropy_buckets():
    result = _accumulate(1).finalize()
    metrics = result.metrics
    assert result.reduction_calls == 0
    assert metrics["actor_diagnostics/all/token_count"] == 4
    assert metrics["actor_diagnostics/all/raw_log_ratio_mean"] == pytest.approx(0.075)
    assert metrics["actor_diagnostics/all/numerical_log_ratio_mean"] == pytest.approx(0.075)
    assert metrics["actor_diagnostics/all/clip_lower_exceedance_fraction"] == pytest.approx(0.25)
    assert metrics["actor_diagnostics/all/clip_upper_exceedance_fraction"] == pytest.approx(0.25)
    assert 0 < metrics["actor_diagnostics/all/effective_ratio_ess_fraction"] <= 1
    assert metrics["actor_diagnostics/entropy_bucket_0000_0256/token_weighted"] == pytest.approx(2.5)
    assert metrics["actor_diagnostics/entropy_bucket_0000_0256/sequence_balanced"] == pytest.approx(2.5)
    assert metrics["actor_diagnostics/entropy_bucket_0000_0256/token_count"] == 4
    assert metrics["actor_diagnostics/entropy_bucket_0000_0256/trajectory_count"] == 1
    assert metrics["actor_diagnostics/boundary_recovered/token_count"] == 4


def test_effective_log_ratio_matches_positive_negative_and_dual_clip_loss_branches():
    data = _batch(1)
    current = [[0.0, -3.0, 2.0, 0.0]]
    accumulator = ActorDiagnosticsAccumulator(_diagnostic_config())
    accumulator.accumulate(_model_output(data, current), data)
    metrics = accumulator.finalize().metrics
    expected = torch.tensor([torch.log(torch.tensor(1.28)), -2.0, torch.log(torch.tensor(3.0)), 1.0]).mean().item()
    assert metrics["actor_diagnostics/all/effective_log_ratio_mean"] == pytest.approx(expected)


def _distributed_worker(rank: int, init_file: str, output_dir: str):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        result = _accumulate(1, rank_offset=rank).finalize(torch.distributed.group.WORLD)
        torch.save(result, os.path.join(output_dir, f"rank-{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("NCBR_RUN_GLOO_TEST") != "1",
    reason="set NCBR_RUN_GLOO_TEST=1 where loopback sockets are permitted",
)
def test_two_process_gloo_matches_single_process_and_reduces_once():
    with tempfile.TemporaryDirectory(prefix="ncbr-gloo-") as directory:
        init_file = os.path.join(directory, "init")
        context = get_context("spawn")
        processes = [
            context.Process(target=_distributed_worker, args=(rank, init_file, directory)) for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        distributed = [torch.load(os.path.join(directory, f"rank-{rank}.pt"), weights_only=False) for rank in range(2)]

    combined = ActorDiagnosticsAccumulator(_diagnostic_config())
    for rank in range(2):
        data = _batch(1, rank_offset=rank)
        combined.accumulate(_model_output(data, [[-0.9, -1.3, -1.0, -0.5]]), data)
    expected = combined.finalize().metrics
    for result in distributed:
        assert result.reduction_calls == 1
        assert result.metrics["actor_diagnostics/dp_reduction_calls"] == 1
        for key, value in expected.items():
            if key == "actor_diagnostics/dp_reduction_calls":
                continue
            assert result.metrics[key] == pytest.approx(value)


def _fixed_replay(enable: bool):
    torch.manual_seed(42)
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-3, weight_decay=0.01)
    data = _batch(1)
    values = [parameter - 0.1, parameter * 0 - 1.3, parameter * 0 - 1.0, parameter * 0 - 0.5]
    nested = torch.nested.as_nested_tensor([torch.stack([parameter * 0, *values, parameter * 0])], layout=torch.jagged)
    model_output = {"log_probs": nested}
    old = data["old_log_probs"]
    response_mask = data["response_mask"].bool()
    # Use the exact same differentiable loss in both arms; diagnostics only reads detached tensors.
    differentiable_current = torch.stack(values).unsqueeze(0)
    loss = -((differentiable_current - old) * data["advantages"] * response_mask).mean()
    accumulator = ActorDiagnosticsAccumulator(_diagnostic_config(enable))
    accumulator.accumulate(model_output, data)
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.detach().clone(),
        "parameter": parameter.detach().clone(),
        "optimizer": optimizer.state_dict(),
        "rng": torch.random.get_rng_state().clone(),
    }


def test_fixed_batch_replay_diagnostics_are_loss_gradient_optimizer_and_rng_inert():
    disabled = _fixed_replay(False)
    enabled = _fixed_replay(True)
    assert torch.equal(disabled["loss"], enabled["loss"])
    assert torch.equal(disabled["parameter"], enabled["parameter"])
    assert disabled["optimizer"].keys() == enabled["optimizer"].keys()
    for key in disabled["optimizer"]["state"][0]:
        assert torch.equal(disabled["optimizer"]["state"][0][key], enabled["optimizer"]["state"][0][key])
    assert torch.equal(disabled["rng"], enabled["rng"])
