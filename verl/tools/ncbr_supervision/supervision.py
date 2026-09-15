"""Local monotonic leases; explicit evidence before any stop signal."""
import errno,json,os,signal,time
from pathlib import Path

def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def pulse(path,mirror=None):
 data={'time':time.time(),'monotonic':time.monotonic(),'pid':os.getpid()}
 write(path,data)
 if mirror:write(mirror,data)

class Health:
 def __init__(self,now):
  self.last_ok={k:now for k in ['parent','stage','pipeline']}
 def check(self,parent,heartbeat,lease,now):
  observations={};reasons=[]
  try:
   os.kill(parent,0);self.last_ok['parent']=now;observations['parent']={'alive':True}
  except ProcessLookupError as e:
   observations['parent']={'alive':False,'errno':e.errno};reasons.append('parent_missing')
  except OSError as e:
   observations['parent']={'alive':None,'errno':e.errno,'error':str(e)}
   if now-self.last_ok['parent']>=90:reasons.append('parent_probe_unavailable_90s')
  for name,path in [('stage',heartbeat),('pipeline',lease)]:
   if path is None:continue
   try:
    data=json.loads(Path(path).read_text());stamp=float(data['monotonic'])
    if not 0<=stamp<=now:raise ValueError('invalid monotonic timestamp')
    age=now-stamp;observations[name]={'age_seconds':age,'producer_pid':data.get('pid')}
    self.last_ok[name]=max(self.last_ok[name],stamp)
    if age>=90:reasons.append(name+'_heartbeat_expired')
   except (OSError,ValueError,KeyError,TypeError) as e:
    age=now-self.last_ok[name];observations[name]={'unavailable_seconds':age,'errno':getattr(e,'errno',None),'error':repr(e)}
    if age>=90:reasons.append(name+'_heartbeat_unavailable_90s')
  return {'time':time.time(),'monotonic':now,'reasons':reasons,'observations':observations}

def watchdog(parent,pid,heartbeat,done,run):
 health=Health(time.monotonic());lease=os.environ.get('NCBR_PIPELINE_LEASE')
 while not done.exists():
  evidence=health.check(parent,heartbeat,lease,time.monotonic())
  evidence.update(parent_pid=parent,training_pgid=pid)
  write(heartbeat.parent/'watchdog_health.json',evidence)
  if evidence['reasons']:
   # Persist locally before signaling; shared audit writes cannot delay safety action.
   write(heartbeat.parent/'watchdog_stop.json',evidence)
   try:os.killpg(pid,signal.SIGTERM)
   except ProcessLookupError:pass
   time.sleep(10)
   try:os.killpg(pid,signal.SIGKILL)
   except ProcessLookupError:pass
   write(run/'watchdog_stop.json',evidence);return
  time.sleep(5)
