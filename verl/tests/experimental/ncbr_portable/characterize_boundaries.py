"""Additional source-derived boundary cases; never rewrite the original fit golden."""
import argparse
import copy
import dataclasses
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.root)/"verl"))
import numpy as np
import pytest
import torch
from verl import DataProto
from verl.experimental.natural_continuation_boundary_return import dapo_trainer, runtime, profiling
from verl.experimental.probe_credit.dynamic_sampling import filter_dapo_generation_batch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, get_policy_loss_fn
from verl.workers.config.actor import ActorConfig
from source_candidate_helpers import _trainer, _hook_candidate

def encode(v):
    if torch.is_tensor(v): return {"dtype": str(v.dtype), "shape": list(v.shape), "values": encode(v.detach().tolist())}
    if isinstance(v, np.ndarray): return encode(v.tolist())
    if isinstance(v, np.generic): return encode(v.item())
    if dataclasses.is_dataclass(v): return encode(dataclasses.asdict(v))
    if isinstance(v, dict): return {str(k): encode(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [encode(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return str(v)
    return v

records = {}
for mode in ("off", "shadow", "replace"):
    for case in ("four_transitions", "locked", "no_cap", "fallback", "eos_at_limit"):
        with pytest.MonkeyPatch.context() as mp:
            serial = itertools.count()
            clock = SimpleNamespace(time=lambda: 100., perf_counter=lambda: 10.)
            for module in (runtime, profiling):
                mp.setattr(module, "time", clock)
                mp.setattr(module, "_profile_uuid4", lambda: SimpleNamespace(hex=str(next(serial))))
            trainer = _trainer(mode)
            trainer.global_steps = 8
            trainer._rollout_policy_version = 7
            trainer.tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
            trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: None)
            trainer.reward_loop_manager = SimpleNamespace(reward_loop_workers=[None, None])
            trainer.config.actor_rollout_ref.rollout.boundary_return = dataclasses.replace(
                trainer.config.actor_rollout_ref.rollout.boundary_return, long_reward_chunk_size=3)
            batch = DataProto.concat([_hook_candidate(), _hook_candidate()])
            batch.non_tensor_batch["uid"] = np.array(["a","a","b","b"], dtype=object)
            batch.non_tensor_batch["trajectory_id"] = np.array([f"t{i}" for i in range(4)], dtype=object)
            short = [0,1,0,1] if case == "locked" else [0,0,1,1]
            long = [0,0,1,1] if case == "locked" else [0,1,0,1]
            batch.non_tensor_batch["acc"] = np.array(short, dtype=float)
            batch.non_tensor_batch["score"] = np.array(short, dtype=float)
            raw = torch.tensor([[-.125,0,0,v] for v in short])
            batch.batch["token_level_scores"] = raw.clone()
            batch.batch["token_level_rewards"] = raw.clone()
            if case == "no_cap": batch.non_tensor_batch["finish_reason"][:] = "stop"
            if case in ("fallback", "eos_at_limit"): batch.non_tensor_batch.pop("finish_reason")
            if case == "eos_at_limit": batch.batch["responses"][0,-1] = 99
            before = copy.deepcopy(batch)
            captures, corrections, scored = [], [], []
            original = dapo_trainer.run_boundary_continuations
            def run(**kw):
                result = original(**kw); captures.append(encode(result)); return result
            mp.setattr(dapo_trainer, "run_boundary_continuations", run)
            original_apply = dapo_trainer.apply_boundary_return
            def apply(*a, **kw):
                result = original_apply(*a, **kw); corrections.append(encode(result)); return result
            mp.setattr(dapo_trainer, "apply_boundary_return", apply)
            class Client:
                async def start_grouped(self, request_id, **kw):
                    class Handle:
                        server_id = "mock"
                        backend_request_id = request_id
                        async def result(self):
                            return [SimpleNamespace(token_ids=[80], stop_reason="completed",
                                extra_fields={"global_steps":7,"branch_id":0,"finish_reason":"stop"})]
                        async def abort(self): pass
                        async def drain(self): pass
                        async def release(self): pass
                    return Handle()
            trainer.llm_server_manager = SimpleNamespace(get_client=lambda: Client())
            def score(b):
                scored.append(len(b))
                values = np.array([long[int(t[1:])] for t in b.non_tensor_batch["trajectory_id"]], dtype=float)
                return SimpleNamespace(reward_tensor=torch.zeros((len(b), 5)), extra_info={"acc":values,"score":values})
            trainer._score_batch_with_existing_reward_pipeline = score
            trainer._process_candidate_after_reward_before_filter(batch, {}, {}, 1)
            filtered = filter_dapo_generation_batch(batch, "boundary_acc" if mode == "replace" else "acc")
            actor = {}
            if len(filtered):
                advantage, returns = compute_grpo_outcome_advantage(filtered.batch["token_level_rewards"],
                    filtered.batch["response_mask"], filtered.non_tensor_batch["uid"])
                actor.update(advantages=encode(advantage), returns=encode(returns))
                for loss_name in ("vanilla", "gspo"):
                    old = torch.full_like(advantage, -1.)
                    current = (old + torch.linspace(-.1,.1,old.numel()).reshape(old.shape)).requires_grad_()
                    loss, metrics = get_policy_loss_fn(loss_name)(old, current, advantage, filtered.batch["response_mask"],
                        config=ActorConfig(strategy="fsdp",rollout_n=2,use_dynamic_bsz=True))
                    loss.backward()
                    actor[loss_name] = encode({"old":old,"current":current,"loss":loss,"grad":current.grad,"metrics":metrics})
            records[f"{mode}/{case}"] = {"raw":encode(raw), "captures":captures,"corrections":corrections,
                "scores":encode(batch.batch["token_level_scores"]), "row_tensors":encode(dict(batch.batch.items())),
                "retained_ids":encode(filtered.non_tensor_batch["trajectory_id"]), "actor":actor, "scored_rows":scored}
            for key in before.batch.keys():
                if key not in ("token_level_scores","token_level_rewards"):
                    assert torch.equal(before.batch[key],batch.batch[key]), key
Path(args.output).write_text(json.dumps(records,sort_keys=True,indent=2,allow_nan=False)+"\n")
