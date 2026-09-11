"""Read-only optimizer receipts from each validation worker."""
import functools,hashlib,json,os,time
from pathlib import Path
import torch
ROOT=Path(os.environ['NCBR_AUDIT_DIR']);ROOT.mkdir(parents=True,exist_ok=True)
def digest(tensors):
    h=hashlib.sha256()
    for t in tensors:
        x=t.detach().contiguous().cpu();h.update(str((x.shape,x.dtype)).encode());h.update(x.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
original=torch.optim.AdamW.step
@functools.wraps(original)
def step(self,*args,**kwargs):
    params=[p for group in self.param_groups for p in group['params']];grads=[p.grad for p in params if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads),'nonfinite gradient'
    marker=f'{os.getpid()}_{time.time_ns()}';before=digest(params)
    torch.save([g.detach().cpu() for g in grads],ROOT/f'{marker}_grads.pt')
    result=original(self,*args,**kwargs)
    receipt=dict(event='actual_AdamW_step',pid=os.getpid(),before=before,after=digest(params),gradient_hash=digest(grads),gradient_nonzero=sum(int(torch.count_nonzero(g)) for g in grads),cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    (ROOT/f'{marker}.json').write_text(json.dumps(receipt,indent=2));return result
torch.optim.AdamW.step=step
# Stable trainer-assigned identities; runtime profiling UUIDs remain independent.
import inspect,uuid
_uuid4=uuid.uuid4
_uid_counter=0
def stable_trainer_uuid():
    global _uid_counter
    caller=inspect.currentframe().f_back.f_code.co_filename
    if caller.endswith('/ray_trainer.py') or caller.endswith('/dapo_trainer.py'):
        _uid_counter+=1
        return uuid.uuid5(uuid.NAMESPACE_URL,f'ncbr-validation-seed42:{_uid_counter}')
    return _uuid4()
uuid.uuid4=stable_trainer_uuid
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
_forward=FSDPEngine.forward_backward_batch
@functools.wraps(_forward)
def capture_forward(self,data,loss_function,forward_only=False):
    marker=f'{os.getpid()}_{time.time_ns()}'
    torch.save(data.cpu(),ROOT/f'{marker}_forward_input.pt')
    (ROOT/f'{marker}_forward_meta.json').write_text(json.dumps(dict(forward_only=forward_only,optimizer_present=getattr(self,'optimizer',None) is not None)))
    result=_forward(self,data,loss_function,forward_only)
    if forward_only:torch.save(result,ROOT/f'{marker}_forward_output.pt')
    return result
FSDPEngine.forward_backward_batch=capture_forward
from verl.workers.engine_workers import ActorRolloutRefWorker
_update=ActorRolloutRefWorker.update_weights
@functools.wraps(_update)
async def update_weights(self,global_steps=None,mode='auto'):
    marker=f'{os.getpid()}_{time.time_ns()}'
    (ROOT/f'{marker}_publish_start.json').write_text(json.dumps(dict(version=global_steps,event='publish_start')))
    result=await _update(self,global_steps=global_steps,mode=mode)
    (ROOT/f'{marker}_publish_done.json').write_text(json.dumps(dict(version=global_steps,event='publish_done')))
    return result
ActorRolloutRefWorker.update_weights=update_weights
(ROOT/f'{os.getpid()}_imports.json').write_text(json.dumps(dict(engine_source=inspect.getfile(FSDPEngine),worker_source=inspect.getfile(ActorRolloutRefWorker),cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))))
from verl.experimental.natural_continuation_boundary_return.dapo_trainer import RayDAPOBoundaryReturnTrainer
_process=RayDAPOBoundaryReturnTrainer._process_candidate_after_reward_before_filter
@functools.wraps(_process)
def process_candidate(self,candidate,*args,**kwargs):
    marker=f'{os.getpid()}_{time.time_ns()}'
    torch.save(candidate,ROOT/f'{marker}_candidate_before.pt')
    result=_process(self,candidate,*args,**kwargs)
    torch.save(candidate,ROOT/f'{marker}_candidate_after.pt')
    return result
RayDAPOBoundaryReturnTrainer._process_candidate_after_reward_before_filter=process_candidate
from verl.experimental.natural_continuation_boundary_return.hook import NCBRHook
_apply=NCBRHook.apply
@functools.wraps(_apply)
def hook_apply(self,batch,raw_scores,**kwargs):
    marker=f'{os.getpid()}_{time.time_ns()}'
    torch.save(dict(batch=batch,raw_scores=raw_scores,extras=kwargs['raw_verifier_extras']),ROOT/f'{marker}_hook_input.pt')
    oracle_evidence=live_oracle(self,batch,raw_scores,kwargs,marker) if os.environ.get('NCBR_LIVE_ORACLE') else None
    if os.environ.get('NCBR_FAULT'):kwargs=inject_fault(kwargs,marker)
    result=_apply(self,batch,raw_scores,**kwargs)
    if oracle_evidence:check_live_oracle(result,oracle_evidence,marker)
    torch.save(result,ROOT/f'{marker}_hook_output.pt')
    if oracle_evidence is not None:
        (ROOT/f'{marker}_live_validation_complete.json').write_text(json.dumps(dict(request_count=len(oracle_evidence['requests']),actor_update_executed=False)))
        raise RuntimeError('controlled live oracle validation complete before actor update')
    return result
NCBRHook.apply=hook_apply
_init=ActorRolloutRefWorker.init_model
@functools.wraps(_init)
def init_model(self,*args,**kwargs):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','');actual=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    if visible.startswith('GPU-') and ',' not in visible:assert actual==visible.removeprefix('GPU-'),(visible,actual)
    (ROOT/f'{os.getpid()}_actual_device.json').write_text(json.dumps(dict(visible=visible,actual_uuid=actual,rank=os.environ.get('RANK'),local_rank=os.environ.get('LOCAL_RANK'))))
    return _init(self,*args,**kwargs)
ActorRolloutRefWorker.init_model=init_model
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
_actor_update=RayPPOTrainer._update_actor
@functools.wraps(_actor_update)
def actor_update(self,batch):
    marker=f'{os.getpid()}_{time.time_ns()}'
    assert batch.batch['responses'].shape[-1] <= 2048
    torch.save(batch,ROOT/f'{marker}_trainer_actor_batch.pt')
    result=_actor_update(self,batch)
    torch.save(result,ROOT/f'{marker}_trainer_actor_result.pt')
    return result
RayPPOTrainer._update_actor=actor_update
def inject_fault(kwargs,marker):
    import copy
    fault=os.environ['NCBR_FAULT'];kwargs=dict(kwargs)
    def event(request_id=None):
        (ROOT/f'{marker}_{time.time_ns()}_fault.json').write_text(json.dumps(dict(fault=fault,event='fault_injected',time_ns=time.time_ns(),request_id=request_id)))
    if fault in ['verifier_error','verifier_timeout']:
        def fail(*args,**kw):
            event();raise (TimeoutError if fault=='verifier_timeout' else RuntimeError)(f'controlled {fault}')
        kwargs['score_long']=fail;return kwargs
    original_client=kwargs['continuation_client']
    class Client:
        async def start_grouped(self,request_id,**kw):
            tracked=await original_client.start_grouped(request_id,**kw)
            class Handle:
                backend_request_id=tracked.backend_request_id;server_id=tracked.server_id
                async def result(self):
                    outputs=await tracked.result()
                    if fault=='release':return outputs
                    event(request_id)
                    if fault=='missing':return []
                    if fault=='duplicate':return [*outputs,*copy.deepcopy(outputs)]
                    if fault=='version':
                        outputs=copy.deepcopy(outputs)
                        for output in outputs:output.extra_fields['global_steps']=999
                        return outputs
                async def release(self):
                    if fault=='release':event(request_id);raise RuntimeError('controlled release acknowledgement unavailable')
                    return await tracked.release()
                async def abort(self):return await tracked.abort()
                async def drain(self):return await tracked.drain()
            return Handle()
    kwargs['continuation_client']=Client();return kwargs
from verl.checkpoint_engine.base import CheckpointEngineManager
_sleep=CheckpointEngineManager.sleep_replicas
@functools.wraps(_sleep)
def sleep_replicas(self):
    marker=f'{os.getpid()}_{time.time_ns()}'
    (ROOT/f'{marker}_sleep.json').write_text(json.dumps(dict(event='sleep_attempt',time_ns=time.time_ns())))
    return _sleep(self)
CheckpointEngineManager.sleep_replicas=sleep_replicas
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.utils.ray_utils import auto_await
_generate=AgentLoopManager.generate_sequences
@auto_await
@functools.wraps(_generate)
async def generate_sequences(self,prompts):
    marker=f'{os.getpid()}_{time.time_ns()}'
    torch.save(prompts,ROOT/f'{marker}_rollout_input.pt')
    output=await _generate(self,prompts)
    torch.save(output,ROOT/f'{marker}_rollout_output.pt')
    return output
AgentLoopManager.generate_sequences=generate_sequences
def load_oracle(name,filename):
    import subprocess,sys,types
    if name in sys.modules:return sys.modules[name]
    checkout=Path(__file__).resolve().parents[2]
    source=subprocess.check_output(['git','show',f'd23da0e:verl/verl/experimental/natural_continuation_boundary_return/{filename}.py'],cwd=checkout,text=True)
    module=types.ModuleType(name);module.__file__=str(ROOT/f'{filename}_oracle.py.txt');Path(module.__file__).write_text(source);sys.modules[name]=module;exec(compile(source,module.__file__,'exec'),module.__dict__);return module

def live_oracle(self,batch,raw,kwargs,marker):
    import copy,dataclasses
    runtime=load_oracle('audit_live_oracle_runtime','runtime');adapter=load_oracle('audit_live_oracle_adapter','reward_adapter')
    work=copy.deepcopy(batch);work.batch['token_level_scores']=raw.clone();work.batch['token_level_rewards']=raw.clone();work.non_tensor_batch.update(copy.deepcopy(kwargs['raw_verifier_extras']))
    common={k:kwargs[k] for k in ['config','policy_version','sampling_params','eos_token_id','short_response_length','max_model_len']}
    capture=runtime.run_boundary_continuations(rollout_batch=work,client=kwargs['continuation_client'],**common)
    if not capture.generations:raise RuntimeError('live oracle smoke has no natural cap coverage')
    reward=kwargs['score_long'](work,capture.generations,kwargs['config'])
    adapter.apply_boundary_return(work,capture=capture,long_reward_output=reward,config=kwargs['config'])
    frozen=NCBRHook(continuation_runner=lambda **unused:capture)
    frozen_kwargs=dict(kwargs,score_long=lambda *unused:reward,long_reward_complete=None,timing_raw={})
    outcome=_apply(frozen,batch,raw,**frozen_kwargs)
    assert torch.equal(outcome.effective_scores,work.batch['token_level_scores'])
    evidence=dict(requests=[dataclasses.asdict(x) for x in capture.requests],generations=[dataclasses.asdict(x) for x in capture.generations],hit_cap=capture.hit_response_cap.tolist(),release_intervals=sum(x.name=='continuation_cleanup_release' for x in capture.profiling_intervals),long_verifier={k:v.tolist() for k,v in reward.extra_info.items()},frozen_scores_exact=True)
    (ROOT/f'{marker}_live_oracle.json').write_text(json.dumps(evidence,indent=2));return evidence

def check_live_oracle(outcome,evidence,marker):
    import dataclasses
    cap=outcome.aux['capture'];requests=[dataclasses.asdict(x) for x in cap.requests]
    assert requests==evidence['requests'];assert cap.hit_response_cap.tolist()==evidence['hit_cap']
    original={x['request_id']:x for x in evidence['generations']}
    differences=[x.request_id for x in cap.generations if tuple(original[x.request_id]['tail_token_ids'])!=x.tail_token_ids]
    assert evidence['release_intervals']==len(requests)
    (ROOT/f'{marker}_live_comparison.json').write_text(json.dumps(dict(requests_exact=True,masks_exact=True,independent_tail_differences=differences,request_count=len(requests),frozen_scores_exact=evidence['frozen_scores_exact']),indent=2))
_reward_colocate=RayPPOTrainer._compute_reward_colocate
@functools.wraps(_reward_colocate)
def compute_reward_colocate(self,batch):
    marker=f'{os.getpid()}_{time.time_ns()}'
    torch.save(batch,ROOT/f'{marker}_reward_batch.pt')
    result=_reward_colocate(self,batch)
    torch.save(result,ROOT/f'{marker}_reward_result.pt')
    return result
RayPPOTrainer._compute_reward_colocate=compute_reward_colocate
from verl.workers.rollout.llm_server import TrackedGroupedRequest
_tracked_release=TrackedGroupedRequest.release
_tracked_drain=TrackedGroupedRequest.drain
@functools.wraps(_tracked_release)
async def tracked_release(self):
    already=self._released
    result=await _tracked_release(self)
    marker=f'{os.getpid()}_{time.time_ns()}'
    (ROOT/f'{marker}_release_ack.json').write_text(json.dumps(dict(event='remote_release_ack',request_id=self.backend_request_id,server_id=self.server_id,already_released=already)))
    return result
@functools.wraps(_tracked_drain)
async def tracked_drain(self):
    result=await _tracked_drain(self)
    marker=f'{os.getpid()}_{time.time_ns()}'
    (ROOT/f'{marker}_drain_ack.json').write_text(json.dumps(dict(event='remote_drain_ack',server_id=self.server_id)))
    return result
TrackedGroupedRequest.release=tracked_release
TrackedGroupedRequest.drain=tracked_drain
