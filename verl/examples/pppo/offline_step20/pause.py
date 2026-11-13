"""One-shot pause after committed Step20. Refuses an early stop; no polling scheduler."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluate import RUN, ROOT, ARMS, read, write, sha


def main():
    status=read(RUN/'training_status.json')
    rows=[json.loads(l) for l in (RUN/'training_metrics.jsonl').open()]
    target=[r for r in rows if r['step']==20]
    if len(target)!=1 or status['last_successful_step']<20:
        raise RuntimeError('Step20 has not finished; do not stop the active driver')
    metrics=target[0]['metrics']
    assert metrics['actor/pppo_optimizer_updates_total']==320 and metrics['actor/pppo_optimizer_updates_this_outer']==16
    assert metrics['pppo/actor_rows']==2048 and metrics['train/retained_prompt_groups']==256
    assert math.isfinite(metrics['actor/grad_norm']) and math.isfinite(metrics['actor/loss'])
    assert 'pppo/holdout_accuracy' in metrics
    directory=RUN/'checkpoints/global_step_20';sidecar=read(directory/'pppo_state.json')
    assert sidecar['status']=='committed' and sidecar['step']==20 and sidecar['policy_version']==20
    expected={'data.pt','pppo_driver_rng.pt'}|{f'actor/{kind}_world_size_16_rank_{rank}.pt' for kind in ('model','optim','extra_state') for rank in range(16)}
    assert expected<=set(sidecar['files'])
    for name,size in sidecar['files'].items():assert (directory/name).stat().st_size==size and size>0
    assert len(list(ARMS['pppo_s20'][1].glob('*.safetensors')))>=1
    # Coordinator must already be disabled; never terminate an unrelated process.
    pipeline=read(RUN/'pipeline_status.json');assert pipeline['state']=='DISABLED_FOR_STEP20_PAUSE'
    pid=status['driver_pid'];p=Path(f'/proc/{pid}/cmdline')
    command=p.read_bytes() if p.exists() else b''
    if b'/experiments/pppo_ncbr_matched/launch.py' not in command or b'formal' not in command:
        raise RuntimeError('Driver identity mismatch; inspect manually')
    snapshot=dict(requested_step=20,actual_last_successful_step=status['last_successful_step'],
        driver_pid=pid,job_id=status['job_id'],checkpoint=str(directory),sidecar_sha256=sha(directory/'pppo_state.json'),
        paused_at=time.time(),training_status_before_stop=status,step20_metrics=target[0],
        step20_process_gpu_hours=16*((directory/'pppo_state.json').stat().st_mtime-status['started_at'])/3600,
        step20_step_timer_gpu_hours=16*sum(r['timing_seconds']['step'] for r in rows if r['step']<=20)/3600,
        baseline_process_gpu_hours=628.649734220604,baseline_step_plus_testing_gpu_hours=614.376976780982,
        cost_note='Step-only sum excludes timers outside step and is not directly equal to step-plus-testing. Process endpoint uses committed file mtime; trailing generated work is separate.')
    write(RUN/'pause_step20_pending.json',snapshot)
    os.kill(pid,signal.SIGTERM)
    # Bounded teardown wait, never wait for training progress here.
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        try:
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            if fields[0]=='Z':break
        except FileNotFoundError:break
        time.sleep(1)
    else:
        if Path(f'/proc/{pid}/cmdline').read_bytes()==command:os.kill(pid,signal.SIGKILL)
    snapshot.update(status='PAUSED_AT_STEP20',stopped_at=time.time(),resume_policy='Stay paused; explicit user instruction required',
        trailing_process_gpu_hours=16*(time.time()-(directory/'pppo_state.json').stat().st_mtime)/3600)
    write(RUN/'pause_step20_receipt.json',snapshot)
    status.update(state='PAUSED_AT_STEP20',paused_at=snapshot['paused_at'],resume_from_checkpoint=str(directory),
        reason='User requested Step20 offline evaluation; no automatic resumption')
    write(RUN/'training_status.json',status)
    print(json.dumps(snapshot,indent=2),flush=True)


if __name__=='__main__':main()
