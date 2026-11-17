"""Opt-in full-response shaping; default behavior covered by existing goldens."""
from dataclasses import replace, fields
import copy
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_reward_adapter_on_cpu import _candidate, _capture, _long_output, _config
from verl.experimental.natural_continuation_boundary_return.reward_adapter import apply_boundary_return, build_long_reward_batch
from verl.experimental.natural_continuation_boundary_return.length_penalty import penalty, parameters
from verl.experimental.natural_continuation_boundary_return.accumulator import BoundaryReturnStepAccumulator

KW = dict(max_resp_len=8192, overlong_buffer_cfg=dict(enable=True, len=1638, penalty_factor=1., log=False))


@pytest.mark.parametrize('length,expected', [(2048,0), (6554,0), (6555,-1/1638), (7373,-.5), (8192,-1)])
def test_matches_actual_dapo_manager(length, expected):
    from verl.workers.reward_manager.dapo import DAPORewardManager
    capture = _capture()
    g = replace(capture.generations[0], tail_token_ids=(200,)*(length-4))
    batch = build_long_reward_batch(_candidate(), [g], pad_token_id=0)
    class Tokenizer:
        eos_token = '<eos>'
        def decode(self, *args, **kwargs): return 'Answer: 1'
    manager = DAPORewardManager(Tokenizer(), 0, compute_score=lambda **kw: {'acc':1., 'score':1.},
                               **{**KW, 'overlong_buffer_cfg': OmegaConf.create(KW['overlong_buffer_cfg'])})
    actual = float(manager(batch, return_dict=True)['reward_tensor'].sum())-1
    assert actual == pytest.approx(expected, abs=1e-7)
    assert penalty(np.array([length]), KW, 8192)[0] == pytest.approx(actual, abs=1e-7)


def apply(mode='replace', lengths=(6554,6555,7373,8192), kwargs=KW):
    candidate = _candidate(); original=copy.deepcopy(candidate)
    cap = _capture()
    cap = replace(cap, generations=tuple(replace(g, tail_token_ids=(200,)*(n-4))
                  for g,n in zip(cap.generations, lengths)))
    cfg = replace(_config(mode), long_response_length=8192, length_penalty_mode='full_response')
    result = apply_boundary_return(candidate, capture=cap, long_reward_output=_long_output(),
                                  config=cfg, length_penalty_kwargs=kwargs)
    return candidate, original, result


def test_full_penalty_only_at_terminal_preserves_task_accuracy_and_other_shaping():
    candidate, old, result = apply()
    diff = candidate.batch['token_level_scores']-old.batch['token_level_scores']
    expected = torch.tensor([0., 1-1/1638, -.5, -3., 0.])
    assert torch.allclose(diff.sum(-1),expected,atol=1e-6)
    assert torch.equal(diff[:,:3],torch.zeros_like(diff[:,:3]))
    assert result.task_score_delta.tolist() == [0,1,0,-2,0]
    assert result.boundary_acc.tolist() == [0,1,1,0,1]
    for key in ('responses','input_ids','attention_mask','response_mask','position_ids'):
        assert torch.equal(candidate.batch[key],old.batch[key])
    assert candidate.batch['boundary_length_penalty_delta'].tolist() == pytest.approx([0,-1/1638,-.5,-1,0])
    # A correct answer can change solely due to length penalty.
    assert candidate.batch['boundary_changed'][2]
    assert result.metrics['boundary_return/prefix_penalty_drift_max'] == 0


def test_shadow_unchanged_and_step_metrics_weighted():
    candidate, old, result = apply('shadow')
    for key in old.batch.keys(): assert torch.equal(candidate.batch[key],old.batch[key])
    assert set(candidate.batch.keys()) == set(old.batch.keys())
    acc=BoundaryReturnStepAccumulator(correctness_threshold=.5);acc.add(result)
    assert acc.metrics()['boundary_return/full_length_penalty_mean'] == pytest.approx(-(1/1638+.5+1)/5)


def test_subtract_short_penalty_exactly_once():
    kw=dict(max_resp_len=6,overlong_buffer_cfg=dict(enable=True,len=4,penalty_factor=1.,log=False))
    candidate=_candidate()
    # Short lengths4 => P(short)=-.5; length2 =>0.
    candidate.batch['token_level_scores'][:4,-1]-=.5
    old=candidate.batch['token_level_scores'].clone()
    cfg=replace(_config(),length_penalty_mode='full_response')
    apply_boundary_return(candidate,capture=_capture(),long_reward_output=_long_output(),config=cfg,length_penalty_kwargs=kw)
    assert (candidate.batch['token_level_scores']-old).sum(-1).tolist()==pytest.approx([-.25,.5,0,-2.25,0])


@pytest.mark.parametrize('lengths', [(8193,4,4,4), (3,4,4,4)])
def test_invalid_full_lengths(lengths):
    if lengths[0]==3:
        # Negative repetition creates empty tail; exercise prefix mismatch explicitly.
        cap=_capture();g=replace(cap.generations[0],prefix_token_ids=(1,2,3))
        cap=replace(cap,generations=(g,*cap.generations[1:]))
        with pytest.raises(ValueError,match='prefix length'):
            apply_boundary_return(_candidate(),capture=cap,long_reward_output=_long_output(),
                config=replace(_config(),long_response_length=8192,length_penalty_mode='full_response'),length_penalty_kwargs=KW)
    else:
        with pytest.raises(ValueError,match='token lengths'):apply(lengths=lengths)


@pytest.mark.parametrize('kwargs',[None,{},dict(max_resp_len=2048,overlong_buffer_cfg=KW['overlong_buffer_cfg']),
    dict(max_resp_len=8192,overlong_buffer_cfg=dict(enable=True,len=0,penalty_factor=1.)),
    dict(max_resp_len=8192,overlong_buffer_cfg=dict(enable=True,len=1638,penalty_factor=float('nan')))])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError): parameters(kwargs,8192)


def test_no_cap_keeps_rewards_and_zero_penalty():
    candidate=_candidate();cap=replace(_capture(),hit_response_cap=np.zeros(5,dtype=bool),generations=())
    out=_long_output();out=replace(out,extra_info={'acc':np.array([]),'score':np.array([])})
    old=candidate.batch['token_level_scores'].clone()
    apply_boundary_return(candidate,capture=cap,long_reward_output=out,
        config=replace(_config(),long_response_length=8192,length_penalty_mode='full_response'),length_penalty_kwargs=KW)
    assert torch.equal(candidate.batch['token_level_scores'],old)


def test_hook_forwards_penalty_and_does_not_mutate_input():
    from verl.experimental.natural_continuation_boundary_return.hook import NCBRHook
    batch=_candidate();old=batch.batch['token_level_scores'].clone()
    cap=_capture();cap=replace(cap,generations=tuple(replace(g,tail_token_ids=(200,)*8188) for g in cap.generations))
    outcome=NCBRHook(continuation_runner=lambda **kw: cap).apply(
        batch,old,raw_verifier_extras={k:batch.non_tensor_batch[k] for k in ('acc','score')},
        config=replace(_config(),long_response_length=8192,length_penalty_mode='full_response'),
        policy_version=7,sampling_params={},continuation_client=None,
        score_long=lambda *args:_long_output(),eos_token_id=None,short_response_length=4,max_model_len=9216,
        length_penalty_kwargs=KW)
    assert torch.equal(batch.batch['token_level_scores'],old)
    assert (outcome.effective_scores-old).sum(-1).tolist()==pytest.approx([-1,0,-1,-3,0])


def test_reordered_chunked_long_scoring_and_padding():
    from verl.experimental.natural_continuation_boundary_return.scoring import score_long_generations
    cap=_capture();cap=replace(cap,generations=tuple(reversed(cap.generations)))
    batch=_candidate()
    def score_batch(long):
        values=np.array([int(x['index']) for x in long.non_tensor_batch['extra_info']],dtype=float)
        from verl.experimental.natural_continuation_boundary_return.reward_adapter import BoundaryRewardOutput
        return BoundaryRewardOutput(torch.zeros_like(long.batch['responses'],dtype=torch.float32),
                                    {'score':values,'acc':np.ones(len(long))})
    cfg=replace(_config(),length_penalty_mode='full_response',long_reward_chunk_size=3)
    out=score_long_generations(batch,cap.generations,cfg,score_batch=score_batch,pad_token_id=0,reward_worker_count=2)
    assert out.extra_info['score'].tolist()==[3,2,1,0]
    result=apply_boundary_return(batch,capture=cap,long_reward_output=out,config=cfg,
        length_penalty_kwargs=dict(max_resp_len=6,overlong_buffer_cfg=dict(enable=True,len=2,penalty_factor=1.)))
    assert result.boundary_task_score.tolist()==[0,1,2,3,.2]
    assert batch.batch['boundary_full_length_penalty'].tolist()==[-.5,-1,0,-.5,0]


def test_nonfinite_input_reward_rejected():
    batch=_candidate();batch.batch['token_level_scores'][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite input'):
        apply_boundary_return(batch,capture=_capture(),long_reward_output=_long_output(),
            config=replace(_config(),long_response_length=8192,length_penalty_mode='full_response'),length_penalty_kwargs=KW)


def test_step_metric_uses_candidate_count_not_batch_count():
    _,_,first=apply()
    doubled={f.name:np.tile(getattr(first,f.name),2) for f in fields(first)
             if isinstance(getattr(first,f.name),np.ndarray) and len(getattr(first,f.name))==5}
    second=replace(first,**doubled,metrics={**first.metrics,'boundary_return/full_length_penalty_mean':-.8})
    acc=BoundaryReturnStepAccumulator(correctness_threshold=.5);acc.add(first);acc.add(second)
    expected=(first.metrics['boundary_return/full_length_penalty_mean']-1.6)/3
    assert acc.metrics()['boundary_return/full_length_penalty_mean']==pytest.approx(expected)


@pytest.mark.parametrize('length', [6554,6555,7373,8192])
def test_matches_async_training_reward_manager(length):
    import asyncio
    from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
    g=replace(_capture().generations[0],tail_token_ids=(200,)*(length-4))
    batch=build_long_reward_batch(_candidate(),[g],pad_token_id=0)
    class Tokenizer:
        eos_token='<eos>'
        def decode(self,*args,**kwargs):return 'Answer: 1'
    async def compute(**kwargs):return {'score':1.,'acc':1.}
    async def run():
        manager=DAPORewardManager(OmegaConf.create({'reward':{'reward_kwargs':KW}}),Tokenizer(),compute)
        # Token decoding is synchronous in this CPU unit test; avoid executor teardown.
        class InlineDecode:
            async def run_in_executor(self, executor, fn): return fn()
        manager.loop = InlineDecode()
        return await manager.run_single(batch)
    result=asyncio.run(run())
    assert result['reward_score']-1==pytest.approx(penalty(np.array([length]),KW,8192)[0],abs=1e-7)
    assert result['reward_extra_info']['score']==1 and result['reward_extra_info']['acc']==1
