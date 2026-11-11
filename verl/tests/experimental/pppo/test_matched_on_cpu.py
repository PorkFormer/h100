import asyncio
import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.pppo.runtime import Request, Generation, validate_result
from verl.experimental.pppo.scoring import scoring_batch, correctness
from verl import DataProto


def test_missing_branch_is_failure():
    request=Request('r',0,0,'t',3,(1,),(2,),8,42)
    output=NS(token_ids=[4],stop_reason='eos',extra_fields={'global_steps':3,'finish_reason':'stop'})
    with pytest.raises(ValueError,match='branch or policy'):
        validate_result(request,[output])


@pytest.mark.parametrize('enabled',[False,True])
def test_optimizer_audit_explicit_opt_in(tmp_path,enabled):
    from verl.experimental.pppo.trainer import RayPPPOTrainer
    t=object.__new__(RayPPPOTrainer)
    t.pppo=OmegaConf.create({'optimizer_audit':{'enabled':enabled}})
    t.output=tmp_path
    t.global_steps=1
    t.config=NS(data=NS(train_batch_size=256),actor_rollout_ref=NS(actor=NS(ppo_mini_batch_size=16,ppo_epochs=1)))
    t._update_actor=lambda batch: NS(meta_info={'metrics':{'actor/pppo_optimizer_updates_this_outer':[16],
                                                        'actor/pppo_optimizer_updates_total':[16]}})
    batch=DataProto.from_dict(tensors={'x':torch.zeros(8,1)})
    batch.meta_info['pppo_audit_dir']='stale'
    t._compute_advantage_and_actor_update(batch,{}, {})
    assert ('pppo_audit_dir' in batch.meta_info)==enabled
    assert batch.meta_info['pppo_strict_update'] is True


def test_real_verifier_roundtrip_with_clean_metadata():
    from verl.utils import hf_tokenizer
    from verl.workers.reward_manager.dapo import DAPORewardManager
    tokenizer=hf_tokenizer('/workspace/models/Qwen3-4B-Base')
    texts=['The answer is 42.\nAnswer: 42',r'We obtain \boxed{42}.',
           'Answer: 41','No solution.', '', 'Answer: 42'+tokenizer.eos_token]
    tokens=[tokenizer.encode(x,add_special_tokens=False) for x in texts]
    width=max(map(len,tokens))
    prompts=torch.tensor([[tokenizer.pad_token_id,10,11]]*len(texts))
    responses=torch.full((len(texts),width),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(responses)
    for i,row in enumerate(tokens):
        responses[i,:len(row)]=torch.tensor(row,dtype=torch.long)
        mask[i,:len(row)]=1
    attention=torch.cat((torch.tensor([[0,1,1]]*len(texts)),mask),-1)
    original=DataProto.from_dict(tensors={'prompts':prompts,'responses':responses,'response_mask':mask,
          'attention_mask':attention,'input_ids':torch.cat((prompts,responses),-1),
          'position_ids':(attention.cumsum(-1)-1).clamp(min=0)},non_tensors={
          'data_source':np.asarray(['math_dapo']*len(texts),dtype=object),
          'reward_model':np.asarray([{'ground_truth':'42'}]*len(texts),dtype=object),
          'extra_info':np.asarray([{'index':0,'repeat':7,'acc':1,'rm_scores':[100],'reward_scores':42}]*len(texts),dtype=object)})
    verifier=DAPORewardManager(tokenizer=tokenizer,num_examine=0,max_resp_len=8192,
                              overlong_buffer_cfg=NS(enable=False,len=0,log=False))
    normal=correctness(verifier(copy.deepcopy(original),return_dict=True)['reward_extra_info'],len(texts))
    generations=[Generation(Request(str(i),i,0,str(i),0,(10,11),(),8192,42),tuple(row),'eos','stop')
                 for i,row in enumerate(tokens)]
    fresh=scoring_batch(original,generations,tokenizer.pad_token_id)
    adapted=correctness(verifier(fresh,return_dict=True)['reward_extra_info'],len(texts))
    assert np.array_equal(normal,adapted)
    assert 'extra_info' not in fresh.non_tensor_batch
    assert 'rm_scores' not in fresh.batch


def test_optimizer_counter_reads_actual_adamw_state():
    # A fresh process reconstructs the count from optimizer state, not a driver guess.
    parameter=torch.nn.Parameter(torch.tensor([2.]))
    optimizer=torch.optim.AdamW([parameter],lr=1e-6)
    for _ in range(3):
        parameter.grad=torch.ones_like(parameter)
        optimizer.step()
    restored=torch.optim.AdamW([parameter],lr=1e-6)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert float(next(iter(restored.state.values()))['step'])==3
    parameter.grad=torch.ones_like(parameter)
    restored.step()
    assert float(next(iter(restored.state.values()))['step'])==4
