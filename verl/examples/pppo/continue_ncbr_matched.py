"""Continue the already-started G=1 job, fail closed, and mail authorized progress.

This orchestration wrapper never changes the validated training code or config.
Only the explicit recipient below receives selected status fields, not raw logs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

REPO = Path(__file__).resolve().parents[3]
HERE = REPO / 'experiments/pppo_ncbr_matched'
sys.path.insert(0, str(HERE))
from prepare import RUN, CONFIG, sha, code_fingerprint, write_json
from launch import require_gates

RECIPIENT = '3531125084@qq.com'
DIRECTORY = RUN / 'automation'
STEPS = [('launch.py', 'smoke-g8'), ('launch.py', 'smoke-resume'),
         ('finalize.py', 'gate3'), ('launch.py', 'profile'), ('finalize.py', 'gate4'),
         ('launch.py', 'formal'), ('evaluate_step300.py',), ('finalize.py', 'reports')]


def read_status(path):
    return json.loads(path.read_text()) if path.exists() else {}


def status_body(stage):
    formal = read_status(RUN / 'training_status.json')
    g1 = read_status(RUN / 'acceptance_runs/smoke-g1/training_status.json')
    g8 = read_status(RUN / 'acceptance_runs/smoke-g8/training_status.json')
    profile = read_status(RUN / 'acceptance_runs/profile/training_status.json')
    lines = ['PPPO + DAPO Dynamic Sampling (NCBR-Matched)',
             time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), '当前阶段: ' + stage]
    for gate in range(5):
        receipt = read_status(RUN / f'acceptance/gate{gate}/result.json')
        result = receipt.get('status', 'NOT RUN')
        if not receipt and gate == 3 and g1.get('state') in ('RUNNING', 'STAGE_COMPLETE'):
            result = 'IN PROGRESS'
        if not receipt and gate == 4 and profile.get('state') == 'RUNNING':
            result = 'IN PROGRESS'
        lines.append(f"Gate {gate}: {result}")
    if formal.get('stage') != 'formal':
        formal = {'state': 'NOT RUN'}
    for name, s in [('G=1 smoke', g1), ('G=8 smoke/resume', g8), ('Profiling', profile), ('Formal', formal)]:
        lines.append(f"{name}: {s.get('state', 'NOT RUN')}; job={s.get('job_id')}; last_successful_step={s.get('last_successful_step', 0)}")
    lines += ['正式目标: Base 全新初始化，300 outer steps；每50步保存，仅评测Step 300。',
              '只有Gate 0–4全部通过才会启动正式训练。阶段失败即停止，不降低核心参数。',
              'Gate 2使用已授权的log-prob阈值0.06；其余数值阈值未放宽。',
              '正式训练完成与评测完成分别判断；smoke或profiling不代表正式训练完成。']
    return '\n'.join(lines) + '\n'


def notify(event, stage):
    receipts = DIRECTORY / 'mail'
    receipts.mkdir(parents=True, exist_ok=True)
    path = receipts / (event + '.json')
    prior = read_status(path)
    if prior.get('smtp_accepted'):
        return True
    if time.time() - prior.get('attempted_at', 0) < 300:
        return False
    message = EmailMessage()
    message['From'] = RECIPIENT  # Existing default msmtp account, verified locally.
    message['To'] = RECIPIENT
    message['Subject'] = '[PPPO-NCBR-Matched] ' + stage
    message['Date'] = formatdate(localtime=False, usegmt=True)
    message['Message-ID'] = make_msgid(domain='qq.com')
    message.set_content(status_body(stage))
    receipt = dict(event=event, recipient=RECIPIENT, message_id=message['Message-ID'],
                   attempted_at=time.time(), smtp_accepted=False)
    try:
        result = subprocess.run(['/usr/bin/msmtp', '-t'], input=message.as_bytes(),
                                capture_output=True, timeout=60)
        receipt.update(exit_code=result.returncode, smtp_accepted=result.returncode == 0)
    except Exception as error:
        receipt['error_type'] = type(error).__name__
    write_json(path, receipt)
    print('MAIL', event, 'SMTP_ACCEPTED' if receipt['smtp_accepted'] else 'FAILED', flush=True)
    return receipt['smtp_accepted']


def process_identity(pid):
    try:
        # comm may contain spaces; fields after the final ')' start with state.
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, TypeError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        require_gates(2)
        print(status_body('执行前检查（未发邮件）'))
        print('Next commands:', STEPS)
        return
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    lock = (DIRECTORY / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (DIRECTORY / 'status.json').exists():
        raise RuntimeError('Existing automation evidence; inspect before restarting')
    os.environ.update(PYTHONPATH=str(REPO / 'verl'), PYTHONDONTWRITEBYTECODE='1',
                      XDG_CACHE_HOME='/tmp/pppo-matched-cache',
                      FLASHINFER_WORKSPACE_BASE='/tmp/pppo-matched-cache/flashinfer', WANDB_MODE='offline')
    state = dict(state='RUNNING', pid=os.getpid(), started_at=time.time(),
                 config_sha256=sha(CONFIG), code_fingerprint=code_fingerprint(),
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                 orchestrator_sha256=sha(Path(__file__)),
                 recipient=RECIPIENT, stage='等待已启动的G=1 smoke', completed=[])
    last_hourly = time.time()

    def tick():
        nonlocal last_hourly
        state['heartbeat'] = time.time()
        write_json(DIRECTORY / 'status.json', state)
        if time.time() - last_hourly >= 3600:
            notify('hour_' + str(int(time.time() // 3600)), state['stage'])
            last_hourly = time.time()
        if state['stage'] == 'launch.py formal':
            for step in range(50, 301, 50):
                ckpt = read_status(RUN / f'checkpoints/global_step_{step}/pppo_state.json')
                if ckpt.get('status') == 'committed':
                    notify(f'checkpoint_{step}', f'正式训练 Step {step} checkpoint 已提交')

    try:
        require_gates(2)
        tick()
        notify('automation_started', '自动执行链已启动；Gate 2通过，等待G=1 smoke')
        path = RUN / 'acceptance_runs/smoke-g1/training_status.json'
        initial = read_status(path)
        pid = initial.get('driver_pid')
        identity = process_identity(pid)
        while True:
            status = read_status(path)
            if status.get('job_id') != initial.get('job_id'):
                raise RuntimeError('G=1 job identity changed')
            if status.get('state') == 'FAILED':
                raise RuntimeError('G=1 smoke failed; see its preserved training_status.json')
            if status.get('state') == 'STAGE_COMPLETE':
                if status.get('last_successful_step') != 1:
                    raise RuntimeError('Unexpected G=1 completion step')
                if process_identity(pid) != identity or identity is None:
                    break
            elif identity is None or process_identity(pid) != identity:
                raise RuntimeError('G=1 driver exited without successful completion')
            elif time.time() - status.get('heartbeat', 0) > 180:
                raise RuntimeError('G=1 status heartbeat stale')
            tick()
            time.sleep(15)
        state['completed'].append('existing smoke-g1')
        notify('smoke_g1_done', 'G=1 smoke 已完成；继续G=8')
        for index, command in enumerate(STEPS):
            if sha(CONFIG) != state['config_sha256'] or code_fingerprint() != state['code_fingerprint']:
                raise RuntimeError('Validated config or source changed; revalidation required')
            state['stage'] = ' '.join(command)
            tick()
            # Preserve detailed prior report before legacy report aggregation.
            if command[0] == 'finalize.py':
                backup = DIRECTORY / f'validation_before_{index}.md'
                shutil.copy2(RUN / 'validation_report.md', backup)
            log = DIRECTORY / f'{index:02d}_{command[0]}_{command[-1]}.log'
            with log.open('x') as stream:
                child = subprocess.Popen([sys.executable, str(HERE / command[0]), *command[1:]],
                                         cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                notify(f'stage_{index}_start', '开始 ' + state['stage'])
                while child.poll() is None:
                    tick()
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError(f'{state["stage"]} exited {child.returncode}; log={log}')
            state['completed'].append(state['stage'])
            notify(f'stage_{index}_done', '完成 ' + state['stage'])
        state.update(state='COMPLETE', stage='正式300步训练及Step 300评测已完成')
        notify('complete', state['stage'])
    except BaseException as error:
        state.update(state='FAILED', error=repr(error))
        notify('failed', '流程停止: ' + state['stage'] + '；' + str(error))
        raise
    finally:
        state['ended_at'] = time.time()
        write_json(DIRECTORY / 'status.json', state)


if __name__ == '__main__':
    main()
