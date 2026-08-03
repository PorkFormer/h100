import math

import pytest
import torch

from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss


def _config(**kwargs):
    defaults = {
        "strategy": "fsdp",
        "rollout_n": 1,
        "use_dynamic_bsz": True,
        "loss_agg_mode": "token-mean",
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "entropy_coeff": 0.0,
        "use_kl_loss": False,
    }
    defaults.update(kwargs)
    return ActorConfig(**defaults)


def _data(*, support=True, ppo_mask=(1, 0), support_mask=(0, 1)):
    tensors = {
        "prompts": torch.tensor([[10], [11]]),
        "responses": torch.tensor([[20], [21]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
        "response_mask": torch.ones(2, 1, dtype=torch.long),
        "old_log_probs": torch.zeros(2, 1),
        "advantages": torch.ones(2, 1),
    }
    if support:
        tensors.update(
            {
                "ppo_response_mask": torch.tensor(ppo_mask, dtype=torch.long).unsqueeze(-1),
                "support_response_mask": torch.tensor(support_mask, dtype=torch.long).unsqueeze(-1),
                "support_sample_mask": torch.tensor(support_mask, dtype=torch.bool),
                "support_ref_seq_logprob": torch.tensor([0.0, -1.0]),
                "support_log_alpha": torch.full((2,), math.log(0.5)),
                "support_lambda": torch.full((2,), 0.25),
                "support_global_batch_size": torch.full((2,), 1),
            }
        )
    return tu.get_tensordict(
        tensors,
        non_tensor_dict={"dp_size": 1, "batch_num_tokens": 1, "global_batch_size": 1},
    )


def _model_output(response_log_probs=(-0.1, -3.0), *, entropy=None):
    # no_padding_2_padding reads prediction positions 0 and 2 from two [prompt,response] sequences.
    values = torch.tensor([response_log_probs[0], 0.0, response_log_probs[1], 0.0], requires_grad=True)
    output = {"log_probs": values}
    if entropy is not None:
        output["entropy"] = torch.tensor([entropy[0], 0.0, entropy[1], 0.0], requires_grad=True)
    return output


def test_missing_support_fields_preserves_original_path_exactly():
    data_a = _data(support=False)
    data_b = data_a.clone()
    output_a = _model_output()
    output_b = {"log_probs": output_a["log_probs"].detach().clone().requires_grad_()}

    loss_a, metrics_a = ppo_loss(_config(), output_a, data_a)
    loss_b, metrics_b = ppo_loss(_config(), output_b, data_b)

    assert torch.equal(loss_a, loss_b)
    assert metrics_a.keys() == metrics_b.keys()
    assert "actor/support_floor_loss" not in metrics_a


def test_mixed_batch_decomposes_ppo_and_support_loss():
    loss, metrics = ppo_loss(_config(), _model_output(), _data())
    expected_pg = -math.exp(-0.1)
    expected_shortfall = math.log(0.5) - (-3.0 - -1.0)

    assert metrics["actor/pg_loss"].aggregate() == pytest.approx(expected_pg)
    assert metrics["actor/support_floor_unweighted_shortfall"].aggregate() == pytest.approx(expected_shortfall)
    assert metrics["actor/support_floor_loss"].aggregate() == pytest.approx(0.25 * expected_shortfall)
    assert loss.item() == pytest.approx(expected_pg + 0.25 * expected_shortfall)


def test_support_only_microbatch_has_differentiable_zero_ppo():
    data = _data(ppo_mask=(0, 0), support_mask=(1, 1))
    data["support_ref_seq_logprob"] = torch.tensor([-1.0, -1.0])
    data["support_global_batch_size"] = torch.full((2,), 2)
    output = _model_output((-2.0, -3.0))
    loss, metrics = ppo_loss(_config(), output, data)

    assert metrics["actor/pg_loss"].aggregate() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert output["log_probs"].grad is not None


def test_ppo_only_microbatch_has_zero_support_term():
    data = _data(ppo_mask=(1, 1), support_mask=(0, 0))
    loss, metrics = ppo_loss(_config(), _model_output(), data)

    assert torch.isfinite(loss)
    assert metrics["actor/support_floor_loss"].aggregate() == 0.0


def test_entropy_uses_only_ppo_mask():
    loss, metrics = ppo_loss(
        _config(entropy_coeff=1.0),
        _model_output(entropy=(2.0, 1000.0)),
        _data(),
    )
    assert metrics["actor/entropy_loss"].aggregate() == pytest.approx(2.0)
    assert torch.isfinite(loss)


def test_non_finite_support_logprob_fails_closed():
    with pytest.raises(ValueError, match="finite"):
        ppo_loss(_config(), _model_output((-0.1, float("nan"))), _data())


def test_support_normalization_is_dp_partition_invariant():
    full_data = _data(ppo_mask=(0, 0), support_mask=(1, 1))
    full_data["support_ref_seq_logprob"] = torch.tensor([-1.0, -1.0])
    full_data["support_global_batch_size"] = torch.full((2,), 2)
    _, full_metrics = ppo_loss(_config(), _model_output((-2.0, -3.0)), full_data)

    parts = []
    for current in (-2.0, -3.0):
        part = _data(ppo_mask=(0, 0), support_mask=(1, 0))
        part["support_ref_seq_logprob"] = torch.tensor([-1.0, 0.0])
        part["support_global_batch_size"] = torch.full((2,), 2)
        tu.assign_non_tensor(part, dp_size=2)
        _, metrics = ppo_loss(_config(), _model_output((current, 0.0)), part)
        parts.append(metrics["actor/support_floor_unweighted_shortfall"].aggregate())

    assert sum(parts) / 2 == pytest.approx(
        full_metrics["actor/support_floor_unweighted_shortfall"].aggregate()
    )
