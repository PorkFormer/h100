"""Exact frozen old/new scheduler replay; real tokens and unchanged verifier."""
import copy, dataclasses, hashlib, json, sys, types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'verl'))
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.natural_continuation_boundary_return.hook import NCBRHook
from verl.experimental.natural_continuation_boundary_return.reward_adapter import BoundaryRewardOutput
from verl.experimental.natural_continuation_boundary_return.scoring import score_long_generations
from verl.experimental.probe_credit.dynamic_sampling import filter_dapo_generation_batch
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.workers.reward_manager.dapo import DAPORewardManager

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/tmp/ncbr_8gpu_scale_20260911/validation/evidence')
OUT = ROOT / 'evidence' / 'replay'
OUT.mkdir(exist_ok=False)
tok = AutoTokenizer.from_pretrained('/workspace/models/Qwen3-1.7B-Base', local_files_only=True)
manager = DAPORewardManager(tok, 0, max_resp_len=2048,
    overlong_buffer_cfg=OmegaConf.create(dict(enable=True, len=410, penalty_factor=1., log=True)))
manifest = {}
def read(path):
    manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return torch.load(path, weights_only=False)

def score(batch):
    result = manager(copy.deepcopy(batch), return_dict=True)
    return BoundaryRewardOutput(result['reward_tensor'], {k: np.asarray(v) for k, v in result['reward_extra_info'].items()})

def long_score(batch, generations, config):
    return score_long_generations(batch, generations, config, score_batch=score,
                                  pad_token_id=tok.pad_token_id, reward_worker_count=2)

summary = {}
for cohort in ['pool2048', 'natural_actor']:
    if cohort == 'pool2048':
        frozen = read(SOURCE / 'replay_h2048_v2/actor_input.pt')
        saved = {}
        for replica in range(8):
            path = SOURCE / f'infer_h2048_replica{replica}/repeat_0_continuations.json'
            manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for r in json.loads(path.read_text()):
                saved[r['request_id']] = dict(tokens=r['tokens'], stop=r['stop_reason'] or r['finish_reason'], finish=r['finish_reason'])
    else:
        frozen = read(SOURCE / 'replay_trainer_natural/actor_input.pt')
        record = json.loads((SOURCE / 'natural_correction_first.json').read_text())
        actual = read(SOURCE.parents[1] / record['hook_output'])
        saved = {g.request_id: dict(tokens=g.tail_token_ids, stop=g.stop_reason, finish=g.finish_reason)
                 for g in actual.aux['capture'].generations}
    batch, raw, extras = frozen['batch'], frozen['raw'], frozen['extras']
    before = copy.deepcopy(batch)
    verified = score(batch)
    assert torch.equal(verified.reward_tensor, raw)
    for k in ['score', 'acc']:
        assert np.array_equal(verified.extra_info[k], extras[k])
    cohort_summary = {}
    snapshots = {}
    for scheduler, concurrency in [('fixed_wave', 4), ('work_conserving', 4), ('work_conserving', 8)]:
        variant = f'{scheduler}_{concurrency}'
        modes = {}; variant_scores = {}
        for mode in ['off', 'shadow', 'replace']:
            calls = []; releases = []
            class Client:
                async def start_grouped(self, request_id, **kwargs):
                    calls.append(dict(request_id=request_id, **kwargs))
                    r = saved[request_id]
                    class Handle:
                        server_id = 'frozen'; backend_request_id = request_id
                        async def result(self):
                            return [types.SimpleNamespace(token_ids=r['tokens'], stop_reason=r['stop'],
                                extra_fields=dict(branch_id=0, global_steps=0, finish_reason=r['finish']))]
                        async def release(self): releases.append(request_id)
                        async def abort(self): raise AssertionError('unexpected abort')
                        async def drain(self): raise AssertionError('unexpected drain')
                    return Handle()
            cfg = BoundaryReturnConfig(mode=mode, scheduler=scheduler, max_concurrent_requests=concurrency,
                                       request_batch_size=8, long_reward_chunk_size=3, seed=42)
            result = NCBRHook().apply(batch, raw, raw_verifier_extras=extras, continuation_client=Client(),
                score_long=long_score, config=cfg, policy_version=0,
                sampling_params=dict(temperature=1., top_p=1., top_k=-1, ignore_eos=False),
                eos_token_id=tok.eos_token_id, short_response_length=2048, max_model_len=9216)
            assert torch.equal(result.effective_scores, frozen['scores'][mode])
            assert len(releases) == len(set(releases)) == len(calls)
            effective = copy.deepcopy(batch)
            effective.batch['token_level_scores'] = result.effective_scores
            effective.batch['token_level_rewards'] = result.effective_scores
            if mode == 'replace': effective.non_tensor_batch['acc'] = result.aux['result'].boundary_acc
            kept = filter_dapo_generation_batch(effective, 'acc')
            advantage, _ = compute_grpo_outcome_advantage(result.effective_scores,
                batch.batch['response_mask'], batch.non_tensor_batch['uid'])
            snapshot = dict(calls=calls, releases=sorted(releases),
                retained=kept.non_tensor_batch['trajectory_id'].tolist(), advantage=advantage,
                tensors={k: v.clone() for k, v in effective.batch.items()},
                generations=[] if mode == 'off' else [dataclasses.asdict(g) for g in result.aux['capture'].generations])
            if scheduler == 'fixed_wave': snapshots[mode] = snapshot
            else:
                reference = snapshots[mode]
                for key in ['calls', 'releases', 'retained', 'generations']: assert snapshot[key] == reference[key], key
                assert torch.equal(advantage, reference['advantage'])
                for key, value in snapshot['tensors'].items(): assert torch.equal(value, reference['tensors'][key]), key
                if mode != 'off':
                    for key in ['short_acc', 'long_acc', 'boundary_acc', 'task_score_delta', 'hit_response_cap']:
                        assert np.array_equal(getattr(result.aux['result'], key), arrays[mode][key], equal_nan=True), key
            if scheduler == 'fixed_wave' and mode != 'off':
                if mode == 'shadow': arrays = {}
                arrays[mode] = {k: getattr(result.aux['result'], k).copy() for k in
                               ['short_acc', 'long_acc', 'boundary_acc', 'task_score_delta', 'hit_response_cap']}
            variant_scores[mode] = result.effective_scores
            modes[mode] = dict(exact=True, requests=len(calls), retained=len(kept), changed_rows=int((raw != result.effective_scores).any(-1).sum()))
            (OUT / f'{cohort}_{variant}_{mode}_calls.json').write_text(json.dumps(calls))
        for key, value in before.batch.items(): assert torch.equal(value, batch.batch[key]), key
        if cohort == 'natural_actor':
            actor = read(SOURCE / 'replay_trainer_natural/actor_actual_input.pt')
            indices = [batch.non_tensor_batch['trajectory_id'].tolist().index(t) for t in actor['batch'].non_tensor_batch['trajectory_id']]
            for mode in variant_scores: assert torch.equal(variant_scores[mode][indices], actor['scores'][mode])
            torch.save(actor, OUT / f'actor_{variant}.pt')
        cohort_summary[variant] = modes
    summary[cohort] = cohort_summary
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
(OUT / 'input_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(summary, indent=2))
