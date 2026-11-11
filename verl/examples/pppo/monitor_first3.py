"""Persistent read-only monitoring until three validated formal outer steps.

Does not stop the S300 run, alter configuration, or count smoke/profile steps.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/pppo_ncbr_matched'))
from prepare import RUN, write_json
from launch import completed_steps, require_gates
from continue_ncbr_matched import notify, read_status, process_identity

DIRECTORY = RUN / 'first3_monitor'


def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    if isinstance(value, (int, float)): return math.isfinite(value)
    return True


def verify_three(status, rows):
    if status.get('stage') != 'formal' or status.get('state') not in ('RUNNING', 'TRAINING_COMPLETE'):
        return None
    if len(rows) < 3: return None
    if status['state'] == 'RUNNING' and (process_identity(status.get('driver_pid')) is None or time.time()-status.get('heartbeat',0)>180):
        raise ValueError('Formal driver missing or heartbeat stale')
    first = rows[:3]
    if [r['step'] for r in first] != [1, 2, 3]:
        raise ValueError('Formal first three steps are missing, duplicated or reordered')
    for row in first:
        metrics = row['metrics']
        expected = {'actor/pppo_optimizer_updates_this_outer':16,
                    'actor/pppo_optimizer_updates_total':16 * row['step'],
                    'pppo/actor_rows':2048, 'train/retained_prompt_groups':256}
        for key, value in expected.items():
            if metrics.get(key) != value: raise ValueError(f'Step {row["step"]}: {key} mismatch')
        if not finite(metrics) or not finite(row['timing_seconds']):
            raise ValueError('Nonfinite formal metrics')
        if not isinstance(metrics.get('actor/grad_norm'), (int,float)) or metrics['actor/grad_norm'] <= 0:
            raise ValueError('Missing or invalid actor gradient norm')
        if not isinstance(metrics.get('actor/pg_loss'), (int,float)):
            raise ValueError('Missing actor PPO loss')
    require_gates(4)
    from omegaconf import OmegaConf
    config = OmegaConf.load(Path(status['attempt']) / 'resolved_config.yaml')
    if config.trainer.resume_mode != 'disable' or config.trainer.total_training_steps != 300:
        raise ValueError('Formal run was not a fresh S300 launch')
    if config.algorithm.pppo.stop_after_step is not None:
        raise ValueError('Formal run unexpectedly has a stop hook')
    return dict(first_three_steps=first, actual_optimizer_updates=48,
                job_id=status.get('job_id'), code_commit=status.get('code_commit'),
                config_sha256=status.get('config_sha256'), formal_training_continues=True)


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    lock = (DIRECTORY / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        snapshot = dict(timestamp=time.time(), pid=os.getpid(), target='formal steps 1–3', status='MONITORING')
        try:
            automation = read_status(RUN / 'automation/status.json')
            formal = read_status(RUN / 'training_status.json')
            snapshot.update(automation=automation, formal=formal)
            if automation.get('state') == 'RUNNING' and (
                process_identity(automation.get('pid')) is None or time.time()-automation.get('heartbeat',0)>180):
                snapshot['warning'] = 'Automation process missing or heartbeat stale'
                notify('monitor_stale_'+str(automation.get('pid')), snapshot['warning'])
            if automation.get('state') == 'FAILED':
                snapshot['warning'] = automation.get('error', 'Pipeline failed')
                notify('monitor_failed_'+str(automation.get('pid')), '需要修复: '+snapshot['warning'])
            rows = completed_steps(RUN)
            evidence = verify_three(formal, rows)
            if evidence is not None:
                write_json(DIRECTORY / 'first3_verified.json', evidence)
                snapshot.update(status='TARGET_REACHED', actual_optimizer_updates=48)
                if notify('formal_first3_verified', '正式训练前3步已验证：累计48次optimizer更新；300步训练继续'):
                    write_json(DIRECTORY / 'status.json', snapshot)
                    return
                snapshot['warning'] = 'Target verified; retrying email delivery'
            result = subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu',
                                     '--format=csv,noheader'],capture_output=True,text=True,timeout=15)
            snapshot['head_gpu_snapshot'] = result.stdout.strip()
            snapshot['gpu_query_exit_code'] = result.returncode
        except Exception as error:
            snapshot.update(status='MONITOR_ERROR', error=repr(error))
            notify('monitor_error_'+type(error).__name__, '前3步监控检查异常: '+str(error))
        write_json(DIRECTORY / 'status.json', snapshot)
        with (DIRECTORY / 'observations.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
        time.sleep(60)


if __name__ == '__main__':
    main()
