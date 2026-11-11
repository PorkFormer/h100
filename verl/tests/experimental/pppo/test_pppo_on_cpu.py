import asyncio
import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.experimental.pppo.objective import group_statistics, prefix_objective
from verl.experimental.pppo.prefix import prefix_lengths, actor_batch
from verl.experimental.pppo.runtime import build_requests, generate_chunk, validate_result, Generation
from verl.experimental.pppo.scoring import scoring_batch, correctness
from verl.experimental.pppo.schedule import PrefixSchedule


def candidate():
    lengths = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32] * 2)
    mask = torch.arange(32)[None, :] < lengths[:, None]
    prompts = torch.tensor([[0, 13, 14]] * 16)
    responses = torch.arange(32)[None, :].expand(16, -1).clone() + 20
    responses[~mask] = 0
    attention = torch.cat([torch.tensor([[0, 1, 1]] * 16), mask.long()], -1)
    reward = torch.tensor([0., 1, 2, 3, 4, 5, 6, 7] * 2)
    ids = ['a'] * 8 + ['b'] * 8
    a, total, w = group_statistics(reward, lengths, ids)
    return DataProto.from_dict(tensors={
        'prompts': prompts, 'responses': responses, 'attention_mask': attention,
        'input_ids': torch.cat([prompts, responses], -1), 'position_ids': (attention.cumsum(-1)-1).clamp(min=0),
        'response_mask': mask, 'pppo_original_length': lengths, 'pppo_prefix_length': lengths // 4,
        'pppo_reward': reward, 'pppo_advantage': a, 'pppo_row_weight': w, 'pppo_group_total_length': total,
        'rm_scores': torch.ones(16,32)}, non_tensors={
        'uid': np.asarray(ids, dtype=object), 'trajectory_id': np.asarray([f'{i//8}:{i%8}' for i in range(16)], dtype=object),
        'data_source': np.asarray(['math']*16, dtype=object),
        'reward_model': np.asarray([{'ground_truth': str(i)} for i in range(16)], dtype=object),
        'acc': np.ones(16), 'extra_info': np.asarray([{'reward_scores': 1}]*16, dtype=object)})


def test_equation_gradient_and_microbatch():
    torch.manual_seed(7)
    n, b, width = 8, 3, 37
    lengths = torch.randint(1, width+1, (n*b,))
    lengths[:8] = 4
    h = lengths * 25 // 100
    reward = torch.randint(0,10,(n*b,)).double()
    ids = [str(i//n) for i in range(n*b)]
    a,total,w = group_statistics(reward,lengths,ids)
    mask = torch.arange(width)[None,:] < h[:,None]
    old = torch.randn(n*b,width,dtype=torch.float64)
    current = (old + torch.randn_like(old)*.5).requires_grad_()
    ratio = (current-old).exp()
    terms = torch.maximum(-ratio*a[:,None], -ratio.clamp(.8,1.28)*a[:,None])
    reference = sum((terms[k*n:(k+1)*n]*mask[k*n:(k+1)*n]).sum()/lengths[k*n:(k+1)*n].sum() for k in range(b))/b
    actual = prefix_objective(current,old,a,mask,w,n*b)
    torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-10)
    gradient = torch.autograd.grad(reference,current,retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(actual,current,retain_graph=True)[0],gradient,atol=1e-12,rtol=1e-10)
    assert torch.count_nonzero(gradient[~mask]) == 0
    order = torch.randperm(n*b)
    simulated = sum(prefix_objective(current[ix],old[ix],a[ix],mask[ix],w[ix],n*b,dp_size=2)/2 for ix in order.split(3))
    torch.testing.assert_close(torch.autograd.grad(simulated,current)[0],gradient,atol=1e-12,rtol=1e-10)


def test_prefix_and_zero_variance():
    mask = torch.arange(100)[None,:] < torch.tensor([1,20,100])[:,None]
    lengths,h=prefix_lengths(mask,15)
    assert h.tolist()==[0,3,15]
    a,_,_=group_statistics(torch.ones(8),torch.ones(8,dtype=torch.long),['a']*8)
    assert not a.any()
    with pytest.raises(ValueError):
        prefix_lengths(torch.zeros(1,4),15)
    with pytest.raises(ValueError):
        group_statistics(torch.ones(7),torch.ones(7),['a']*7)


def test_crop_and_raw_tokens_and_scoring():
    batch=candidate()
    requests=build_requests(batch,8,8192,3,42,9216)
    assert len(requests)==128 and len({r.seed for r in requests})==128
    assert requests[0].max_tokens==8191
    assert requests[0].prompt+requests[0].prefix==(13,14,20)
    same=build_requests(batch,8,8192,3,42,9216)
    assert same==requests
    generations=[Generation(r,(77,), 'eos','stop') for r in requests[:8]]
    fresh=scoring_batch(batch,generations,0)
    assert 'rm_scores' not in fresh.batch and 'acc' not in fresh.non_tensor_batch
    assert 'extra_info' not in fresh.non_tensor_batch
    assert len(fresh)==8 and all(x['ground_truth']=='0' for x in fresh.non_tensor_batch['reward_model'])
    assert fresh.batch['responses'][0].tolist()==[20,77]
    old_lengths=batch.batch['pppo_original_length'].clone()
    cropped=actor_batch(batch,'cropped',0)
    assert cropped.batch['responses'].shape==(16,8)
    assert torch.equal(cropped.batch['pppo_original_length'],old_lengths)
    assert torch.equal(cropped.batch['response_mask'].sum(-1),old_lengths//4)
    roundtrip=DataProto.from_tensordict(cropped.to_tensordict())
    assert torch.equal(roundtrip.batch['pppo_row_weight'],cropped.batch['pppo_row_weight'])
    fallback=candidate()
    fallback.batch['pppo_prefix_length'][0]=0
    actor_batch(fallback,'cropped',0)
    assert fallback.batch['responses'].shape[-1]==8
    assert fallback.batch['response_mask'][0].sum()==1
    assert fallback.batch['responses'][0,0]==20
    assert fallback.meta_info['pppo_zero_prefix_fallback'] is False
    assert not fallback.batch['pppo_loss_mask'][0].any()


@pytest.mark.parametrize('failure',['none','version','duplicate','timeout','release'])
def test_runtime_lifecycle(failure):
    requests=build_requests(candidate(),1,8192,3,42,9216)[:4]
    handles=[]
    class Handle:
        server_id='server'
        def __init__(self,request):
            self.request=request
            self.events=[]
        async def result(self):
            if failure=='timeout':
                await asyncio.sleep(1)
            await asyncio.sleep(.001*(4-self.request.parent))
            output=NS(token_ids=[9],stop_reason='eos',extra_fields={'global_steps':2 if failure=='version' else 3,'finish_reason':'stop','branch_id':0})
            return [output,output] if failure=='duplicate' else [output]
        async def release(self):
            self.events.append('release')
            if failure=='release':
                raise RuntimeError('release failed')
        async def abort(self): self.events.append('abort')
        async def drain(self): self.events.append('drain')
    class Client:
        async def start_grouped(self,request_id,**kwargs):
            assert kwargs['sampling_params']['n']==1
            h=Handle(next(r for r in requests if r.request_id==request_id))
            handles.append(h)
            return h
    if failure=='none':
        result=asyncio.run(generate_chunk(requests,Client(),{},2,.1))
        assert [r.request for r in result]==requests
        assert all('release' in h.events for h in handles)
        assert any('drain' in h.events for h in handles)
    else:
        with pytest.raises((ValueError,RuntimeError,TimeoutError)):
            asyncio.run(generate_chunk(requests,Client(),{},2,.02))
        assert any('drain' in h.events for h in handles)
        if failure=='timeout': assert all('abort' in h.events for h in handles)


def test_schedule_roundtrip_and_idempotence():
    continuous=PrefixSchedule()
    resumed=PrefixSchedule()
    for step in range(0,201,10):
        continuous.observe(step,.25)
        resumed=PrefixSchedule(**resumed.state_dict())
        resumed.observe(step,.25)
        snapshot=resumed.state_dict()
        resumed.observe(step,.25)
        assert resumed.state_dict()==snapshot==continuous.state_dict()
    assert continuous.eta==35


def test_scoring_failure_is_not_wrong_answer():
    with pytest.raises(ValueError): correctness({'score':[0]},1)
    with pytest.raises(ValueError): correctness({'acc':[float('nan')]},1)
    assert correctness({'acc':[0,1]},2).tolist()==[0,1]


def test_actual_padding_and_loss_dispatch():
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
    from verl.workers.utils.losses import ppo_loss
    batch=actor_batch(candidate(),'cropped',0)
    td=left_right_2_no_padding(batch.to_tensordict())
    # Padded response fields and scalar weights must survive the actual conversion.
    assert torch.equal(td['pppo_row_weight'],batch.batch['pppo_row_weight'])
    nnz=int(batch.batch['attention_mask'].sum())
    values=torch.linspace(-2.,-1.,nnz,requires_grad=True)
    model={'log_probs':values}
    padded=no_padding_2_padding(values,td)
    td['old_log_probs']=padded.detach().clone()
    from verl.utils import tensordict_utils as tu
    tu.assign_non_tensor(td,dp_size=1,global_batch_size=16)
    cfg=NS(policy_loss={'loss_mode':'pppo'},use_kl_loss=False,entropy_coeff=0.,clip_ratio_c=None,
           clip_ratio_low=.2,clip_ratio_high=.28)
    loss,metrics=ppo_loss(cfg,model,td)
    expected=prefix_objective(padded,td['old_log_probs'],td['pppo_advantage'],td['pppo_loss_mask'],td['pppo_row_weight'],16)
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert values.grad.isfinite().all()
    assert 'actor/pg_loss' in metrics


def test_trainer_import_and_stable_identity():
    from verl.experimental.pppo.trainer import RayPPPOTrainer
    t=object.__new__(RayPPPOTrainer)
    t.pppo=NS(seed=42)
    t.occurrences=0
    t.data_epoch=0
    t.epoch_batches=0
    batch=DataProto.from_dict(tensors={'dummy_tensor':torch.zeros(2,1)},non_tensors={
        'raw_prompt':np.asarray([[{'role':'user','content':'hi'}]]*2,dtype=object)})
    t._initialize_candidate_identity(batch,0)
    first=batch.non_tensor_batch['uid'].copy()
    assert len(set(first))==2
    t.occurrences=0
    t._initialize_candidate_identity(batch,0)
    assert np.array_equal(first,batch.non_tensor_batch['uid'])


@pytest.mark.parametrize('consumed',[2,4])
def test_data_cursor_restore_including_epoch_boundary(consumed):
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler
    def loader():
        ds=list(range(16))
        sampler=RandomSampler(ds,generator=torch.Generator().manual_seed(42))
        return StatefulDataLoader(ds,batch_size=4,sampler=sampler,num_workers=0,drop_last=True)
    original=loader()
    iterator=iter(original)
    for _ in range(consumed): next(iterator)
    state=copy.deepcopy(original.state_dict())
    expected=list(iterator)+list(iter(original))
    restored=loader()
    restored.load_state_dict(state)
    actual=list(iter(restored))+list(iter(restored))
    assert len(actual)==len(expected)
    assert all(torch.equal(x,y) for x,y in zip(actual,expected,strict=True))


def test_checkpoint_pending_commit_restore_and_corruption(tmp_path,monkeypatch):
    import json
    from verl.experimental.pppo.trainer import RayPPPOTrainer
    from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer
    def trainer():
        t=object.__new__(RayPPPOTrainer)
        t.config=NS(trainer=NS(default_local_dir=str(tmp_path/'checkpoints'),test_freq=10,
                              nnodes=1,n_gpus_per_node=8,resume_mode='auto',resume_from_path=None))
        t.pppo=NS(holdout_hash='holdout',schedule=NS(type='plateau',patience=3,min_delta=0.,cooldown_validations=1))
        t.config_hash='config'
        t.prefix_schedule=PrefixSchedule(best=.5,last_validation_step=0)
        t.global_steps=10
        t.occurrences=123
        t.data_epoch=2
        t.epoch_batches=3
        t.output=tmp_path
        t.pending_checkpoint=None
        t.actor_rollout_wg=NS(load_checkpoint=lambda *args,**kwargs:None)
        t.train_dataloader=NS(load_state_dict=lambda state:None)
        return t
    def save(t):
        d=tmp_path/'checkpoints/global_step_10'
        (d/'actor').mkdir(parents=True)
        for kind in ('model','optim','extra_state'):
            for rank in range(8):
                torch.save({'value':rank},d/'actor'/f'{kind}_world_size_8_rank_{rank}.pt')
        torch.save({'cursor':123},d/'data.pt')
    monkeypatch.setattr(RayDAPOProbeCreditTrainer,'_save_checkpoint',save)
    monkeypatch.setattr(RayDAPOProbeCreditTrainer,'_validate',lambda *args,**kwargs:{'pppo/holdout_accuracy':.5})
    t=trainer()
    t._save_checkpoint()
    state_path=tmp_path/'checkpoints/global_step_10/pppo_state.json'
    assert json.loads(state_path.read_text())['status']=='validation_pending'
    assert not (tmp_path/'checkpoints/pppo_latest.json').exists()
    # Simulate crash between model save and validation, followed by explicit recovery.
    resumed=trainer()
    resumed.config.trainer.resume_mode='resume_path'
    resumed.config.trainer.resume_from_path=str(state_path.parent)
    resumed._load_checkpoint()
    assert resumed.occurrences==123 and resumed.data_epoch==2
    resumed._validate()
    assert json.loads(state_path.read_text())['status']=='committed'
    assert resumed.prefix_schedule.bad_count==1
    resumed._validate()
    assert resumed.prefix_schedule.bad_count==1
    (state_path.parent/'data.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='size mismatch'):
        resumed._load_checkpoint()


@pytest.mark.parametrize('all_zero',[False,True])
def test_zero_prefix_guard_preserves_objective_and_gradients(all_zero):
    original=candidate()
    original.batch['pppo_prefix_length'][0]=0
    if all_zero:original.batch['pppo_prefix_length'].zero_()
    reference=actor_batch(copy.deepcopy(original),'full_mask_reference',0)
    cropped=actor_batch(copy.deepcopy(original),'cropped',0)
    width=cropped.batch['responses'].shape[-1]
    assert width==(1 if all_zero else 8)
    assert (cropped.batch['response_mask'].sum(-1)>=1).all()
    for name in ['pppo_original_length','pppo_group_total_length','pppo_row_weight','pppo_advantage']:
        torch.testing.assert_close(cropped.batch[name],reference.batch[name])
    torch.manual_seed(11)
    current=torch.randn(16,32,dtype=torch.double,requires_grad=True)
    old=torch.randn_like(current)
    def objective(batch,logp,oldp):
        return prefix_objective(logp,oldp,batch.batch['pppo_advantage'].double(),
                                batch.batch['pppo_loss_mask'],batch.batch['pppo_row_weight'].double(),16)
    full=objective(reference,current,old)
    short=objective(cropped,current[:,:width],old[:,:width])
    torch.testing.assert_close(full,short,rtol=1e-12,atol=1e-12)
    grad_full=torch.autograd.grad(full,current,retain_graph=True)[0]
    grad_crop=torch.autograd.grad(short,current)[0]
    torch.testing.assert_close(grad_full,grad_crop,rtol=1e-12,atol=1e-12)
    assert not grad_crop[0].any()
    assert not grad_crop[:,width:].any()
