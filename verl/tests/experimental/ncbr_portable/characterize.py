"""CPU-only characterization; source root is explicit, never installed.

Freeze once against d23da0e, then compare without rewriting the golden.
External inference and verification are mocked by the unmodified source harness.
"""
import argparse
import copy
import dataclasses
import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.root) / "verl"))
import numpy as np
import pytest
import torch
from verl.experimental.natural_continuation_boundary_return import dapo_trainer, profiling, runtime
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer
from verl.trainer.ppo.core_algos import get_policy_loss_fn
from verl.workers.config.actor import ActorConfig

spec = importlib.util.spec_from_file_location("source_harness", Path(__file__).with_name("source_harness.py"))
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

def serialize(value):
    if torch.is_tensor(value):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": serialize(value.detach().tolist())}
    if isinstance(value, np.ndarray):
        return serialize(value.tolist())
    if isinstance(value, np.generic):
        return serialize(value.item())
    if dataclasses.is_dataclass(value):
        return serialize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value

def batch_record(batch):
    return {"tensors": serialize(dict(batch.batch.items())), "extras": serialize(batch.non_tensor_batch)}

records = {}
for mode in ("off", "shadow", "replace"):
    for cycles in (1, 2):
        for loss_name in ("vanilla", "gspo"):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            evidence = {"candidates": [], "captures": [], "corrections": [], "actor": []}
            with pytest.MonkeyPatch.context() as mp:
                ids = itertools.count()
                fixed_time = SimpleNamespace(perf_counter=lambda: 10.0, time=lambda: 100.0)
                mp.setattr(runtime, "time", fixed_time)
                mp.setattr(profiling, "time", fixed_time)
                for module in (runtime, profiling):
                    mp.setattr(module, "_profile_uuid4", lambda: SimpleNamespace(hex=str(next(ids))))
                original_runtime = dapo_trainer.run_boundary_continuations
                def capture(**kwargs):
                    evidence["candidates"].append(batch_record(kwargs["rollout_batch"]))
                    result = original_runtime(**kwargs)
                    evidence["captures"].append(serialize(result))
                    return result
                mp.setattr(dapo_trainer, "run_boundary_continuations", capture)
                original_apply = dapo_trainer.apply_boundary_return
                def correction(*a, **kw):
                    result = original_apply(*a, **kw)
                    evidence["corrections"].append(serialize(result))
                    return result
                mp.setattr(dapo_trainer, "apply_boundary_return", correction)
                def bind(function, trainer):
                    if function.__name__ != "advantage_actor":
                        return MethodType(function, trainer)
                    def update(self, batch):
                        mask = batch.batch["response_mask"]
                        old = torch.linspace(-2, -1, mask.numel()).reshape(mask.shape)
                        current = (old + torch.linspace(-.15, .15, mask.numel()).reshape(mask.shape)).requires_grad_()
                        batch.batch["old_log_probs"] = old
                        batch.batch["ref_log_prob"] = old - .03
                        batch.batch["rollout_log_probs"] = old - .02
                        loss, metrics = get_policy_loss_fn(loss_name)(
                            old_log_prob=old, log_prob=current,
                            advantages=batch.batch["advantages"], response_mask=mask,
                            config=ActorConfig(strategy="fsdp", rollout_n=2, use_dynamic_bsz=True),
                        )
                        loss.backward()
                        evidence["actor"].append({"batch": batch_record(batch), "current": serialize(current),
                                                  "loss": serialize(loss), "gradient": serialize(current.grad),
                                                  "metrics": serialize(metrics)})
                        return SimpleNamespace(meta_info={"metrics": {}})
                    trainer._update_actor = MethodType(update, trainer)
                    return MethodType(RayDAPOProbeCreditTrainer._compute_advantage_and_actor_update, trainer)
                mp.setattr(harness, "MethodType", bind)
                result = harness._run_fit_harness(mp, mode=mode, generation_batches=cycles)
                if result.error:
                    raise result.error
                assert len(evidence["actor"]) == 1
                harness._assert_rng_equal(result.rng_before, result.rng_after)
                evidence["events"] = result.events
                evidence["filter"] = serialize([(m, sorted(keys), uids) for m, keys, uids in result.filter_spy])
                evidence["metrics"] = serialize([{k: v for k, v in row.items() if k.startswith(("boundary_return/", "train/"))}
                                                  for row, step in result.logger.records])
                records[f"{mode}/{cycles}/{loss_name}"] = evidence
Path(args.output).write_text(json.dumps(records, sort_keys=True, indent=2, allow_nan=False) + "\n")
