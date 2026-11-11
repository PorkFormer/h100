"""Persist reproducible Gate 0/1 evidence. GPU gates have separate real runners."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from omegaconf import OmegaConf
from prepare import CONFIG, HERE, MODEL, REPO, RUN, SOURCE, SOURCE_SHA, sha, write_json, code_fingerprint


def gate0():
    from verl.experimental.pppo.config import validate_config
    c=OmegaConf.load(CONFIG)
    OmegaConf.resolve(c)
    validate_config(c)
    expected={
        'trainer.nnodes':2,'trainer.n_gpus_per_node':8,'trainer.total_training_steps':300,
        'trainer.save_freq':50,'trainer.test_freq':10,'trainer.resume_mode':'disable',
        'data.max_prompt_length':1024,'data.max_response_length':8192,'data.train_batch_size':256,'data.gen_batch_size':768,
        'actor_rollout_ref.model.path':str(MODEL),'actor_rollout_ref.actor.ppo_mini_batch_size':16,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,'actor_rollout_ref.actor.ppo_epochs':1,
        'actor_rollout_ref.actor.optim.lr':1e-6,'actor_rollout_ref.actor.optim.weight_decay':.01,
        'actor_rollout_ref.actor.optim.betas':[.9,.999],'actor_rollout_ref.actor.optim.lr_scheduler_type':'constant',
        'actor_rollout_ref.actor.optim.lr_warmup_steps':0,'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio':0.,
        'actor_rollout_ref.actor.grad_clip':1.,'actor_rollout_ref.actor.fsdp_config.optimizer_offload':True,
        'actor_rollout_ref.actor.fsdp_config.param_offload':False,'actor_rollout_ref.actor.fsdp_config.model_dtype':'fp32',
        'actor_rollout_ref.actor.fsdp_config.dtype':'bfloat16','actor_rollout_ref.actor.shuffle':False,
        'actor_rollout_ref.model.enable_gradient_checkpointing':True,
        'actor_rollout_ref.actor.ulysses_sequence_parallel_size':1,
        'actor_rollout_ref.rollout.tensor_model_parallel_size':2,'actor_rollout_ref.rollout.nnodes':0,
        'actor_rollout_ref.rollout.n':8,'actor_rollout_ref.rollout.response_length':8192,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':2,
        'actor_rollout_ref.rollout.temperature':1.,'actor_rollout_ref.rollout.top_p':1.,
        'actor_rollout_ref.rollout.top_k':-1,'actor_rollout_ref.rollout.ignore_eos':False,
        'algorithm.pppo.extra_continuations_per_prefix':8,'algorithm.pppo.eta_percent':15,
        'algorithm.pppo.seed':42,'algorithm.pppo.actor_view':'cropped',
        'algorithm.pppo.optimizer_audit.enabled':False,
        'algorithm.filter_groups.enable':True,'algorithm.filter_groups.metric':'pppo_filter_reward',
        'algorithm.filter_groups.max_num_gen_batches':10,
    }
    for key,value in expected.items():
        actual=OmegaConf.select(c,key)
        if actual!=value: raise ValueError(f'{key}: {actual!r} != {value!r}')
    diff=json.loads((HERE/'config_diff.json').read_text())
    if diff['unapproved_differences'] or diff['config_sha256']!=sha(CONFIG):
        raise ValueError('Frozen config or field audit changed')
    split=json.loads((RUN/'data/split_manifest.json').read_text())
    if sha(SOURCE)!=SOURCE_SHA or split['holdout_questions']!=256:
        raise ValueError('Source/holdout mismatch')
    for name,digest in split['files'].items():
        if sha(RUN/'data'/name)!=digest:raise ValueError('Split file changed: '+name)
    model_files={}
    revision='906bfd4b4dc7f14ee4320094d8b41684abff8539'
    for metadata in sorted((MODEL/'.cache/huggingface/download').glob('*.metadata')):
        name=metadata.name.removesuffix('.metadata')
        fields=metadata.read_text().splitlines()
        if fields[0]!=revision:raise ValueError('Model/tokenizer revision differs: '+name)
        path=MODEL/name
        digest=sha(path)
        if len(fields[1])==64:
            if digest!=fields[1]:raise ValueError('Base weight does not match download hash: '+name)
        else:
            content=path.read_bytes()
            git_blob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            if git_blob!=fields[1]:raise ValueError('Base metadata/tokenizer changed: '+name)
        model_files[name]=digest
    if len([n for n in model_files if n.endswith('.safetensors')])!=3:
        raise ValueError('Missing Base weight download provenance')
    write_json(HERE/'provenance/model_identity.json',dict(path=str(MODEL),revision=revision,files=model_files,
               historical_link='Same Base path and config/tokenizer hashes as NCBR provenance; local download metadata predates NCBR'))
    return dict(checked_invariants=expected,holdout_hash=c.algorithm.pppo.holdout_hash,
                base_model_revision=revision,model_files=model_files,
                source_sha256=SOURCE_SHA,diff_sha256=sha(HERE/'config_diff.json'),
                formal_updates_per_outer_step=16,formal_total_optimizer_updates=4800)


def gate1(directory):
    env=os.environ.copy()
    env.update(PYTHONPATH=str(REPO/'verl'),PYTHONDONTWRITEBYTECODE='1',
               XDG_CACHE_HOME='/tmp/pppo-plan-cache',FLASHINFER_WORKSPACE_BASE='/tmp/pppo-plan-cache/flashinfer')
    command=['python','-m','pytest','-q','-p','no:cacheprovider',
             'verl/tests/experimental/pppo','verl/tests/experimental/natural_continuation_boundary_return',
             'verl/tests/experimental/ncbr_portable']
    with (directory/'pytest.log').open('w') as log:
        result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'CPU tests failed: {directory}/pytest.log')
    return dict(command=command,log_sha256=sha(directory/'pytest.log'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('gate',type=int,choices=[0,1])
    args=parser.parse_args()
    directory=RUN/'acceptance'/f'gate{args.gate}'
    directory.mkdir(parents=True,exist_ok=True)
    report=dict(gate=args.gate,status='FAIL',timestamp=time.time(),config_sha256=sha(CONFIG),
                code_fingerprint=code_fingerprint(),
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    try:
        report['evidence']=gate0() if args.gate==0 else gate1(directory)
        report['status']='PASS'
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        write_json(directory/'result.json',report)
        print(json.dumps(report,indent=2))
