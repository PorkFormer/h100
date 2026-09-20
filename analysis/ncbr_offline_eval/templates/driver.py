"""Record the controller's actual OS exit, independently of completion status."""
import fcntl,os,subprocess,sys,time
from common import ROOT,atomic_json,sha
lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with (ROOT/'controller.log').open('a') as log:
    child=subprocess.Popen([sys.executable,str(ROOT/'controller.py')],stdout=log,stderr=subprocess.STDOUT)
    atomic_json(ROOT/'driver.json',dict(driver_pid=os.getpid(),controller_pid=child.pid,time=time.time()))
    code=child.wait()
atomic_json(ROOT/'driver_exit.json',dict(returncode=code,normal_exit=code==0,time=time.time()))
atomic_json(ROOT/'SHA256SUMS.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS.json','driver.lock','controller.lock')})
sys.exit(code)
