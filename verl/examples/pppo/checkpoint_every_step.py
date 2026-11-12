"""Prepare a fresh run with every-step checkpoints; retain original gate provenance.

Only checkpoint cadence and output identity change. No GPU gate is claimed rerun.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
HERE = REPO / 'experiments/pppo_ncbr_matched'
OLD = Path('/workspace/rl/h100/outputs/pppo_ncbr_matched_16xa10040g_s300')
NEW = OLD.with_name(OLD.name + '_ckpt1_v2')
BASE = REPO / 'verl/examples/pppo/configs/pppo_ncbr_matched_16xa10040g.yaml'
CONFIG = BASE.with_name('pppo_ncbr_matched_16xa10040g_ckpt1.yaml')
PARENT_COMMIT = 'f5f4b6539cef93dce2b5bffa627f6e5f6c5ac2ba'
PARENT_FINGERPRINT = 'af770681ab4ff63f74056a09b2079113460c6248970d7a01211e6074b0b639c9'
sys.path.insert(0, str(HERE))
from prepare import sha, write_json, code_fingerprint, flatten
from omegaconf import OmegaConf


def approved_changes():
    return {
        'trainer.save_freq': 1,
        'trainer.default_local_dir': str(NEW / 'checkpoints'),
        'trainer.experiment_name': NEW.name,
        'ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR': str(NEW / 'wandb'),
        'ray_kwargs.ray_init.runtime_env.env_vars.WANDB_NAME': NEW.name,
    }


def verify_delta(base, candidate):
    expected = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    for key, value in approved_changes().items():
        OmegaConf.update(expected, key, value)
    if OmegaConf.to_container(expected, resolve=True) != OmegaConf.to_container(candidate, resolve=True):
        raise ValueError('Unapproved algorithm/protocol change in checkpoint-only restart')


def prepare():
    if (NEW / 'training_metrics.jsonl').exists():
        raise ValueError('Refuse to reprepare a run with training evidence')
    # Review exact infrastructure-only patch, rather than relabeling changed GPU code.
    for root in ['verl/verl', 'verl/tests/experimental/pppo', 'experiments/pppo_ncbr_matched']:
        names = subprocess.check_output(['git','diff','--name-only',PARENT_COMMIT,'--',root],cwd=REPO,text=True).splitlines()
        unexpected = [n for n in names if n.endswith(('.py','.yaml')) and n != 'experiments/pppo_ncbr_matched/prepare.py']
        if unexpected:
            raise ValueError(f'GPU implementation changed; revalidation needed: {unexpected}')
    old_prepare = subprocess.check_output(['git','show',PARENT_COMMIT+':experiments/pppo_ncbr_matched/prepare.py'],cwd=REPO,text=True)
    expected_prepare = old_prepare.replace('import math\n','import math\nimport os\n').replace(
        "RUN = ROOT / 'outputs/pppo_ncbr_matched_16xa10040g_s300'",
        "RUN = Path(os.environ.get('PPPO_MATCHED_RUN', ROOT / 'outputs/pppo_ncbr_matched_16xa10040g_s300'))").replace(
        "CONFIG = REPO / 'verl/examples/pppo/configs/pppo_ncbr_matched_16xa10040g.yaml'",
        "CONFIG = Path(os.environ.get('PPPO_MATCHED_CONFIG', REPO / 'verl/examples/pppo/configs/pppo_ncbr_matched_16xa10040g.yaml'))")
    if (HERE/'prepare.py').read_text() != expected_prepare:
        raise ValueError('Unreviewed infrastructure changes')
    c = OmegaConf.load(BASE)
    for key,value in approved_changes().items():
        OmegaConf.update(c,key,value)
    verify_delta(OmegaConf.load(BASE),c)
    from verl.experimental.pppo.config import validate_config
    validate_config(c)
    if c.trainer.resume_mode != 'disable' or c.trainer.resume_from_path is not None:
        raise ValueError('Restart must initialize fresh from Base')
    NEW.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(c,CONFIG,resolve=True)
    before=flatten(OmegaConf.to_container(OmegaConf.load(BASE),resolve=True))
    after=flatten(OmegaConf.to_container(c,resolve=True))
    changes=[dict(field=k,before=before[k],after=after[k]) for k in before if before[k]!=after[k]]
    audit=dict(parent_commit=PARENT_COMMIT,old_run=str(OLD),new_run=str(NEW),changes=changes,
        config_sha256=sha(CONFIG),code_fingerprint=code_fingerprint(),
        authorization='User requested stopping and restarting with revised checkpoint saving for equal GPU-hour comparison',
        core_algorithm_changed=False,checkpoint_retention='All steps 1..300; no automatic deletion',
        validation='GPU Gates 2-4 inherited, NOT rerun. Identical GPU source; save_freq=1 already used in real profile and multi-step fresh-process smoke.')
    write_json(NEW/'restart_audit.json',audit)
    # Preserve original receipts verbatim, with an explicit scoped reuse receipt beside them.
    for gate in range(5):
        parent=OLD/f'acceptance/gate{gate}/result.json'
        receipt=json.loads(parent.read_text())
        if receipt['status']!='PASS' or receipt['config_sha256']!=sha(BASE) or receipt['code_fingerprint']!=PARENT_FINGERPRINT:
            raise ValueError(f'Parent gate {gate} invalid')
        directory=NEW/f'acceptance/gate{gate}';directory.mkdir(parents=True,exist_ok=True)
        (directory/'parent_result.json').write_bytes(parent.read_bytes())
        write_json(directory/'result.json',dict(gate=gate,status='PASS',
            validation_mode='SCOPED_REUSE_NOT_RERUN',config_sha256=sha(CONFIG),code_fingerprint=code_fingerprint(),
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            timestamp=time.time(),parent_receipt=str(parent),parent_sha256=sha(parent),
            evidence=audit))
    write_json(NEW/'training_status.json',dict(state='PREPARED',stage='formal',job_id=None,last_successful_step=0))
    reports=HERE/'reports/checkpoint_every_step_restart';reports.mkdir(parents=True,exist_ok=True)
    write_json(reports/'restart_audit.json',audit)
    print(json.dumps(audit,indent=2))


def pipeline():
    os.environ.update(PPPO_MATCHED_RUN=str(NEW),PPPO_MATCHED_CONFIG=str(CONFIG),
                      PYTHONPATH=str(REPO/'verl'),PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='offline',
                      XDG_CACHE_HOME='/tmp/pppo-matched-cache',FLASHINFER_WORKSPACE_BASE='/tmp/pppo-matched-cache/flashinfer')
    import prepare as context
    context.RUN, context.CONFIG = NEW, CONFIG
    from continue_ncbr_matched import notify
    import fcntl
    lock=(NEW/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (NEW/'training_metrics.jsonl').exists():raise ValueError('Fresh restart cannot overwrite updates')
    stages=[['launch.py','formal'],['evaluate_step300.py'],['finalize.py','reports']]
    state=dict(state='RUNNING',pid=os.getpid(),started_at=time.time(),completed=[])
    try:
        for args in stages:
            state['stage']=' '.join(args);write_json(NEW/'pipeline_status.json',state)
            notify('ckpt1_start_'+args[0], '每步保存的新运行：开始 '+state['stage'])
            with (NEW/(args[0]+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-u',str(HERE/args[0]),*args[1:]],cwd=REPO,
                                      stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{args} failed: {result.returncode}')
            state['completed'].append(args)
            notify('ckpt1_done_'+args[0], '每步保存的新运行：完成 '+state['stage'])
        state['state']='COMPLETE'
    except BaseException as error:
        state.update(state='FAILED',error=repr(error))
        notify('ckpt1_failure', '每步保存的新运行失败，请检查保留日志：'+repr(error))
        raise
    finally:
        state['updated_at']=time.time();write_json(NEW/'pipeline_status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','pipeline']);args=parser.parse_args()
    (prepare if args.action=='prepare' else pipeline)()
