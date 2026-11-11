"""Restart only a verified-dead, incomplete G=1 attempt with detached supervision."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from continue_ncbr_matched import REPO, HERE, RUN, notify, read_status, process_identity
from launch import require_gates
from prepare import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    require_gates(2)
    smoke=RUN/'acceptance_runs/smoke-g1'
    auto=RUN/'automation'
    for path,key in [(smoke/'training_status.json','driver_pid'),(auto/'status.json','pid')]:
        status=read_status(path)
        if status and process_identity(status.get(key)) is not None:
            raise RuntimeError(f'Existing process still alive: {path}; do not duplicate it')
    if (smoke/'training_metrics.jsonl').exists():
        raise RuntimeError('Completed/partial update evidence requires explicit checkpoint recovery review')
    if not args.run:
        print('Dead-process prerequisite verified. --run archives interrupted attempt and launches detached jobs.')
        return
    stamp=str(time.time_ns())
    interrupted=RUN/'interrupted'/stamp;interrupted.mkdir(parents=True)
    for root,name,status_name in [(smoke,'smoke-g1','training_status.json'),(auto,'automation','status.json')]:
        if root.exists():
            previous=read_status(root/status_name)
            write_json(root/'observed_stale_status.json',previous)
            previous.update(state='INTERRUPTED',detected_at=time.time(),
                            reason='Driver/coordinator process absent, no successful update; exit cause not established')
            write_json(root/status_name,previous)
    notify('process_interruption_'+stamp,'检测到driver与协调器退出：未完成更新，保留现场并改为持久进程重启')
    for root,name in [(smoke,'smoke-g1'),(auto,'automation')]:
        if root.exists():root.rename(interrupted/name)
    env=os.environ.copy();env.update(PYTHONPATH=str(REPO/'verl'),PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='offline')
    receipts={}
    def spawn(name,command):
        log=RUN/'execution_logs'/f'{name}_detached_{stamp}.log'
        with log.open('x') as out:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                   stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        receipts[name]=dict(pid=child.pid,process_identity=process_identity(child.pid),log=str(log),command=command)
        write_json(interrupted/'detached_restart.json',receipts)
        return child
    driver=spawn('smoke_g1',[sys.executable,'-u',str(HERE/'launch.py'),'smoke-g1'])
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        if driver.poll() is not None:raise RuntimeError('Detached G=1 driver failed at startup; inspect its log')
        status=read_status(smoke/'training_status.json')
        if status.get('driver_pid')==driver.pid and status.get('job_id'):break
        time.sleep(1)
    else:raise RuntimeError('Detached driver did not publish job identity within 120 seconds')
    spawn('continuation',[sys.executable,'-u',str(Path(__file__).with_name('continue_ncbr_matched.py')),'--run'])
    spawn('first3_monitor',[sys.executable,'-u',str(Path(__file__).with_name('monitor_first3.py'))])
    print(json.dumps(receipts,indent=2),flush=True)


if __name__=='__main__':main()
