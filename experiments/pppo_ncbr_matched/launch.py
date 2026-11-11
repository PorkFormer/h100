"""Stage-isolated launches; formal training is locked behind real gate receipts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from omegaconf import OmegaConf
from prepare import CONFIG, REPO, RUN, sha, write_json, code_fingerprint


def completed_steps(root):
    path=root/'training_metrics.jsonl'
    if not path.exists():return []
    rows=[]
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:break  # a live writer may be between writes
    return rows


def validate_checkpoints(root,steps):
    records=[]
    for step in steps:
        directory=root/'checkpoints'/f'global_step_{step}'
        state=json.loads((directory/'pppo_state.json').read_text())
        if state['status']!='committed' or state['step']!=step or state['policy_version']!=step:
            raise ValueError(f'Uncommitted or mismatched checkpoint {step}')
        files=[directory/'data.pt',directory/'pppo_driver_rng.pt']
        files += [directory/'actor'/f'{kind}_world_size_16_rank_{rank}.pt'
                  for kind in ('model','optim','extra_state') for rank in range(16)]
        for path in files:
            if not path.exists() or path.stat().st_size!=state['files'].get(str(path.relative_to(directory))):
                raise ValueError(f'Incomplete checkpoint: {path}')
        records.append(dict(step=step,path=str(directory),status='PASS',files=state['files'],
                            holdout_hash=state['holdout_hash'],schedule=state['schedule']))
    write_json(root/'checkpoint_manifest.json',dict(status='PASS',checkpoints=records,
               validation='Exact rank membership and committed file sizes; fresh-process restoration is separate Gate 3 evidence'))
    return records


def require_gates(last):
    for gate in range(last+1):
        path=RUN/'acceptance'/f'gate{gate}'/'result.json'
        receipt=json.loads(path.read_text())
        if (receipt['status']!='PASS' or receipt['config_sha256']!=sha(CONFIG)
                or receipt.get('code_fingerprint')!=code_fingerprint()):
            raise ValueError(f'Gate {gate} missing, failed or stale')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['smoke-g1','smoke-g8','smoke-resume','profile','formal'])
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if socket.gethostname().split('.')[0]!='p-kt-dgx-a100-03':raise SystemExit('Run on allocated NCBR head')
    require_gates(4 if args.stage=='formal' else (3 if args.stage=='profile' else 2))
    c=OmegaConf.load(CONFIG)
    root=RUN if args.stage=='formal' else RUN/'acceptance_runs'/('smoke-g8' if args.stage=='smoke-resume' else args.stage)
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'launch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'training_metrics.jsonl').exists() and not (args.resume or args.stage=='smoke-resume'):
        raise SystemExit('Existing training evidence: refuse fresh overwrite')
    c.trainer.default_local_dir=str(root/'checkpoints')
    stage_name='smoke-g8' if args.stage=='smoke-resume' else args.stage
    c.trainer.experiment_name=RUN.name if args.stage=='formal' else RUN.name+'_'+stage_name
    env=c.ray_kwargs.ray_init.runtime_env.env_vars
    env.WANDB_DIR=str(root/'wandb')
    env.WANDB_NAME=c.trainer.experiment_name
    env.WANDB_RUN_ID='pppo_matched_16a100_s300' if args.stage=='formal' else 'pppo_matched_'+stage_name.replace('-','_')
    if args.stage!='formal':env.WANDB_MODE='offline'
    target=300
    if args.stage.startswith('smoke'):
        c.data.train_batch_size=16
        c.data.gen_batch_size=48
        c.actor_rollout_ref.actor.ppo_mini_batch_size=8
        c.algorithm.pppo.optimizer_audit.enabled=True
        c.trainer.save_freq=1
        target={'smoke-g1':1,'smoke-g8':3,'smoke-resume':4}[args.stage]
        c.algorithm.pppo.extra_continuations_per_prefix=1 if args.stage=='smoke-g1' else 8
        c.algorithm.pppo.stop_after_step=target
    elif args.stage=='profile':
        target=1
        c.algorithm.pppo.stop_after_step=1
        # Save timing must be measured at this scale; schedule horizon stays 300.
        c.trainer.save_freq=1
    if args.stage=='smoke-resume':args.resume=root/'checkpoints/global_step_3'
    if args.resume:
        if args.stage=='formal' and args.resume.resolve().parent != (RUN/'checkpoints').resolve():
            raise ValueError('Formal resume must use this formal experiment, never smoke/profile')
        state=json.loads((args.resume/'pppo_state.json').read_text())
        if state['status']!='committed':raise ValueError('Only committed checkpoints may resume')
        c.trainer.resume_mode='resume_path'
        c.trainer.resume_from_path=str(args.resume)
    from verl.experimental.pppo.config import validate_config
    validate_config(c)
    attempt=root/'attempts'/f'{args.stage}_{time.time_ns()}'
    attempt.mkdir(parents=True)
    OmegaConf.save(c,attempt/'resolved_config.yaml',resolve=True)
    status=dict(state='STARTING',stage=args.stage,job_id=None,driver_pid=os.getpid(),
                started_at=time.time(),last_successful_step=0,formal_target=300,
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                config_sha256=sha(attempt/'resolved_config.yaml'),frozen_config_sha256=sha(CONFIG),attempt=str(attempt))
    write_json(root/'training_status.json',status)
    import ray
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    runtime=OmegaConf.merge(get_ppo_ray_runtime_env(),c.ray_kwargs.ray_init.runtime_env)
    kwargs=OmegaConf.to_container(c.ray_kwargs.ray_init,resolve=True)
    kwargs['runtime_env']=OmegaConf.to_container(runtime,resolve=True)
    ray.init(**kwargs)
    status.update(state='RUNNING',job_id=ray.get_runtime_context().get_job_id())
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            rows=completed_steps(root)
            status['last_successful_step']=rows[-1]['step'] if rows else 0
            status['heartbeat']=time.time()
            write_json(root/'training_status.json',status.copy())
            stop.wait(20)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        from verl.experimental.pppo.main_pppo import PPPOTaskRunner
        from verl.trainer.main_ppo import run_ppo
        run_ppo(c,task_runner_class=ray.remote(num_cpus=1)(PPPOTaskRunner))
        rows=completed_steps(root)
        if not rows or rows[-1]['step']!=target:raise RuntimeError('Training exited before required successful step')
        if args.stage=='formal' and [r['step'] for r in rows]!=list(range(1,301)):
            raise RuntimeError('Formal steps incomplete or duplicated')
        validate_checkpoints(root,[50,100,150,200,250,300] if args.stage=='formal' else [target])
        status.update(state='TRAINING_COMPLETE' if args.stage=='formal' else 'STAGE_COMPLETE',last_successful_step=target)
    except BaseException as error:
        status.update(state='FAILED',error=repr(error))
        raise
    finally:
        stop.set();thread.join()
        status['ended_at']=time.time()
        write_json(root/'training_status.json',status)
        write_json(attempt/'training_status.json',status)
        ray.shutdown()


if __name__=='__main__':main()
