"""Execute the existing standard fit with CPU services and real GRPO/loss."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import source_harness
from verl import DataProto
from verl.trainer.ppo import ray_trainer
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, get_policy_loss_fn
from verl.workers.config.actor import ActorConfig


def run_standard(mp, *, mode="replace", enable=None, loss_name="gspo", tail=(99,),
                 long_scores=(0, 1, 0, 1), failure=None, baseline_fit=None, balance=False, raw_naive=False):
    actual_fit = baseline_fit or ray_trainer.RayPPOTrainer.fit
    trace, actors, snapshots, calls = [], [], {}, {"client": 0, "long": 0}
    class Logger:
        def __init__(self, **kw): pass
        def log(self, **kw): pass
    import verl.utils.tracking
    mp.setattr(verl.utils.tracking, "Tracking", Logger)
    for name in ("compute_data_metrics", "compute_timing_metrics", "compute_throughout_metrics",
                 "compute_variance_proxy_metrics", "compute_spec_decode_metrics"):
        mp.setattr(ray_trainer, name, lambda *a, **kw: {})
    mp.setattr(ray_trainer, "should_save_ckpt_esi", lambda **kw: False)
    mp.setattr(source_harness, "RayDAPOBoundaryReturnTrainer", ray_trainer.RayPPOTrainer)

    def fit(self):
        cfg = self.config
        cfg.ncbr = {"enable": enable}
        cfg.algorithm.filter_groups.enable = False
        cfg.actor_rollout_ref.actor = {"loss_agg_mode": "token-mean", "loss_scale_factor": None}
        if raw_naive:
            cfg.reward.reward_manager.name = "naive"
            cfg.actor_rollout_ref.actor.policy_loss = {"loss_mode": loss_name}
        cfg.actor_rollout_ref.rollout.skip = {"enable": False}
        cfg.global_profiler.profile_continuous_steps = False
        cfg.trainer.critic_warmup = 0
        cfg.trainer.balance_batch = balance
        def balance_batch(batch, **kw):
            trace.append("balance")
            snapshots["identities"] = batch.non_tensor_batch["trajectory_id"].copy()
            batch.reorder(torch.tensor([2, 3, 0, 1]))
        self._balance_batch = balance_batch
        self.actor_rollout_wg = SimpleNamespace()
        self._start_profiling = lambda *a: None
        self._stop_profiling = lambda *a: None
        self.use_rm = True
        self.use_reference_policy = not raw_naive
        old_sleep = self.checkpoint_manager.sleep_replicas
        self.checkpoint_manager.sleep_replicas = lambda: (trace.append("sleep"), old_sleep())[-1]
        old_rollout = self.async_rollout_manager.generate_sequences
        def rollout(batch):
            trace.append("rollout")
            out = old_rollout(batch)
            out.non_tensor_batch["multi_modal_inputs"] = np.asarray([{} for _ in range(len(out))], dtype=object)
            out.batch["loss_mask"] = out.batch["attention_mask"][:, -4:].clone()
            out.batch["rollout_log_probs"] = torch.full((len(out), 4), -1.0)
            snapshots["rollout"] = copy.deepcopy(out)
            return out
        self.async_rollout_manager.generate_sequences = rollout
        old_client = self.llm_server_manager.get_client
        def get_client():
            calls["client"] += 1
            client = old_client()
            start = client.start_grouped
            async def tracked(*a, **kw):
                trace.append("continue")
                handle = await start(*a, **kw)
                original_result = handle.result
                async def result():
                    items = await original_result()
                    items[0].token_ids = list(tail)
                    if failure == "version": items[0].extra_fields["global_steps"] = 6
                    if failure == "missing": return []
                    if failure == "duplicate": return items * 2
                    return items
                async def release():
                    trace.append("release")
                    if failure == "release": raise RuntimeError("release failed")
                handle.result, handle.release = result, release
                return handle
            client.start_grouped = tracked
            return client
        self.llm_server_manager.get_client = get_client
        def reward(batch):
            is_long = batch.meta_info.get("boundary_reward_only", False)
            trace.append("long_reward" if is_long else "short_reward")
            if is_long:
                calls["long"] += 1
                if failure == "verifier": raise RuntimeError("verifier failed")
            values = list(long_scores if is_long else (0, 0, 1, 1))
            scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
            scores[:, -1] = torch.tensor(values)
            scores[:, 0] -= .125
            task_values = np.asarray(values, dtype=float)
            if raw_naive:
                task_values = 2 * task_values - 1
                scores.zero_()
                mask = batch.batch["attention_mask"][:, -scores.shape[1]:]
                for row in range(len(batch)):
                    last = torch.nonzero(mask[row], as_tuple=True)[0][-1]
                    scores[row, last] = float(task_values[row])
                batch.non_tensor_batch["data_source"] = np.asarray(["math_dapo"] * len(batch))
            if not is_long:
                snapshots["raw"] = scores.clone()
                snapshots["pre_hook"] = copy.deepcopy(batch)
            extras = {"acc": np.asarray(values, dtype=float), "score": task_values}
            if is_long and failure in ("error", "timeout"):
                extras[failure] = np.asarray([True] * len(batch))
            return DataProto.from_dict(tensors={"rm_scores": scores}, non_tensors=extras,
                                       meta_info={"reward_extra_keys": list(extras)})
        self._compute_reward_colocate = reward
        def old(batch):
            trace.append("old")
            return DataProto.from_dict(tensors={"old_log_probs": torch.full((len(batch), 4), -1.0),
                                               "entropys": torch.zeros((len(batch), 4))}), 0
        self._compute_old_log_prob = old
        def ref(batch):
            trace.append("ref")
            return DataProto.from_dict(tensors={"ref_log_prob": torch.full((len(batch), 4), -1.1)})
        self._compute_ref_log_prob = ref
        def actor(batch):
            trace.append("actor")
            old = batch.batch["old_log_probs"]
            current = (old + torch.linspace(-.1, .1, old.numel()).reshape(old.shape)).requires_grad_()
            mask = batch.batch["response_mask"]
            loss, _ = get_policy_loss_fn(loss_name)(old, current, batch.batch["advantages"], mask,
                config=ActorConfig(strategy="fsdp", rollout_n=2, use_dynamic_bsz=True))
            loss.backward()
            actors.append({"batch": copy.deepcopy(batch), "loss": loss.detach(), "grad": current.grad.clone(),
                           "ratio": torch.exp(((current.detach()-old)*mask).sum(-1)/mask.sum(-1))})
            return DataProto.from_dict(tensors={}, meta_info={"metrics": {}})
        self._update_actor = actor
        try:
            return actual_fit(self)
        except ValueError as error:
            # The source fixture records RuntimeError; keep the real cause attached.
            raise RuntimeError(str(error)) from error
    mp.setattr(ray_trainer.RayPPOTrainer, "fit", fit)
    result = source_harness._run_fit_harness(mp, mode=mode, generation_batches=1)
    return SimpleNamespace(result=result, trace=trace, actors=actors, snapshots=snapshots, calls=calls)


@pytest.mark.parametrize("loss_name", ["vanilla", "gspo"])
@pytest.mark.parametrize("mode,enable", [("off", None), ("shadow", None), ("replace", True), ("replace", False)])
def test_standard_fit_reward_flow_and_actor_prefix(monkeypatch, mode, enable, loss_name):
    run = run_standard(monkeypatch, mode=mode, enable=enable, loss_name=loss_name)
    assert run.result.error is None
    assert len(run.actors) == 1
    batch = run.actors[0]["batch"]
    active = mode != "off" and enable is not False
    assert run.calls == {"client": int(active), "long": int(active)}
    expected = run.snapshots["raw"].clone()
    if mode == "replace" and active: expected[:, -1] += torch.tensor([0, 1, -1, 0])
    assert torch.equal(batch.batch["token_level_scores"], expected)
    advantage, _ = compute_grpo_outcome_advantage(expected, batch.batch["response_mask"], batch.non_tensor_batch["uid"])
    assert torch.equal(batch.batch["advantages"], advantage)
    for key, value in run.snapshots["pre_hook"].batch.items():
        assert torch.equal(batch.batch[key], value), key
    assert torch.equal(batch.batch["rm_scores"], run.snapshots["raw"])
    assert batch.batch["old_log_probs"].shape == (4, 4)
    assert batch.batch["ref_log_prob"].shape == (4, 4)
    assert not any(key.startswith("boundary_") for key in batch.batch.keys())
    assert run.trace.index("sleep") < run.trace.index("old") < run.trace.index("actor")
    if active:
        assert run.trace.index("long_reward") < run.trace.index("sleep")
        assert max(i for i, event in enumerate(run.trace) if event == "release") < run.trace.index("sleep")
    else:
        assert run.trace.index("sleep") < run.trace.index("short_reward")


def test_gspo_tail_isolation_and_score_sensitivity():
    runs = []
    for tail, scores in [((99,), (0, 1, 0, 1)), ((71, 72, 73), (0, 1, 0, 1)), ((71,), (1, 0, 1, 0))]:
        with pytest.MonkeyPatch.context() as mp:
            runs.append(run_standard(mp, tail=tail, long_scores=scores))
    a, b, c = [run.actors[0] for run in runs]
    for key in a["batch"].batch.keys(): assert torch.equal(a["batch"].batch[key], b["batch"].batch[key]), key
    for key in ("loss", "grad", "ratio"): assert torch.equal(a[key], b[key]), key
    assert torch.equal(a["ratio"], c["ratio"])
    assert not torch.equal(a["batch"].batch["advantages"], c["batch"].batch["advantages"])


@pytest.mark.parametrize("failure", ["version", "missing", "duplicate", "verifier", "error", "timeout", "release"])
def test_standard_failure_blocks_actor_and_respects_cleanup(monkeypatch, failure):
    run = run_standard(monkeypatch, failure=failure)
    assert run.result.error is not None
    assert not run.actors
    assert "old" not in run.trace
    assert ("sleep" not in run.trace) if failure == "release" else run.trace.count("sleep") == 1


@pytest.mark.parametrize("loss_name", ["vanilla", "gspo"])
def test_disabled_is_identical_to_original_standard_fit(loss_name):
    namespace = dict(vars(ray_trainer))
    exec(compile(Path(__file__).with_name("baseline_fit.py.txt").read_text(), "bfa0886-fit", "exec"), namespace)
    with pytest.MonkeyPatch.context() as mp:
        baseline = run_standard(mp, mode="off", loss_name=loss_name, baseline_fit=namespace["fit"])
    with pytest.MonkeyPatch.context() as mp:
        disabled = run_standard(mp, mode="replace", enable=False, loss_name=loss_name)
    assert baseline.trace == disabled.trace
    assert baseline.calls == disabled.calls == {"client": 0, "long": 0}
    a, b = baseline.actors[0], disabled.actors[0]
    assert list(a["batch"].batch.keys()) == list(b["batch"].batch.keys())
    for key in a["batch"].batch.keys(): assert torch.equal(a["batch"].batch[key], b["batch"].batch[key]), key
    for key in ("loss", "grad", "ratio"): assert torch.equal(a[key], b[key]), key


def test_standard_balancing_keeps_trajectory_identity(monkeypatch):
    run = run_standard(monkeypatch, balance=True)
    assert run.result.error is None
    batch = run.actors[0]["batch"]
    assert batch.non_tensor_batch["trajectory_id"].tolist() == run.snapshots["identities"][[2, 3, 0, 1]].tolist()
    assert run.trace.index("balance") < run.trace.index("short_reward") < run.trace.index("continue")


@pytest.mark.parametrize("loss_name", ["vanilla", "gspo"])
def test_raw_naive_off_shadow_replace_and_tail(loss_name):
    runs = []
    for mode, tail in [("off", (99,)), ("shadow", (99,)), ("replace", (99,)), ("replace", (71, 72, 73))]:
        with pytest.MonkeyPatch.context() as mp:
            runs.append(run_standard(mp, mode=mode, loss_name=loss_name, raw_naive=True, tail=tail))
    for run in runs:
        assert run.result.error is None, run.result.error
        assert len(run.actors) == 1
        batch = run.actors[0]["batch"]
        assert "ref_log_prob" not in batch.batch
        for key in ("responses", "input_ids", "attention_mask"):
            assert torch.equal(batch.batch[key], run.snapshots["pre_hook"].batch[key])
    off, shadow, replace, tail = [r.actors[0] for r in runs]
    for key in ("loss", "grad"):
        assert torch.equal(off[key], shadow[key])
        assert torch.equal(replace[key], tail[key])
    for key in ("token_level_rewards", "advantages"):
        assert torch.equal(off["batch"].batch[key], shadow["batch"].batch[key])
        assert torch.equal(replace["batch"].batch[key], tail["batch"].batch[key])
    assert not torch.equal(off["batch"].batch["token_level_rewards"], replace["batch"].batch["token_level_rewards"])
    assert torch.equal(replace["batch"].batch["token_level_rewards"].sum(-1), torch.tensor([-1., 1., -1., 1.]))
