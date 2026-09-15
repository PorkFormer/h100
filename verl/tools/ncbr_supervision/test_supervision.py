import json,errno
from pathlib import Path
import pytest
import supervision as s

def hb(p,t):p.write_text(json.dumps({'monotonic':t,'pid':123}))
@pytest.fixture
def live(monkeypatch):monkeypatch.setattr(s.os,'kill',lambda *args:None)

def test_fresh_heartbeat_ignores_mtime(tmp_path,live):
 p=tmp_path/'hb';hb(p,99);s.os.utime(p,(0,0))
 assert not s.Health(0).check(123,p,p,100)['reasons']

@pytest.mark.parametrize('name',['stage','pipeline'])
def test_each_expired_lease_is_identified(tmp_path,live,name):
 a=tmp_path/'a';b=tmp_path/'b';hb(a,100 if name=='pipeline' else 0);hb(b,100 if name=='stage' else 0)
 assert s.Health(0).check(123,a,b,100)['reasons']==[name+'_heartbeat_expired']

def test_io_fault_requires_continuous_90_seconds(tmp_path,live,monkeypatch):
 p=tmp_path/'hb';hb(p,100);h=s.Health(100);assert not h.check(123,p,None,100)['reasons']
 def error(*args,**kwargs):raise OSError(errno.EIO,'transient I/O')
 with monkeypatch.context() as m:
  m.setattr(Path,'read_text',error)
  for t in [105,125,189]:assert not h.check(123,p,None,t)['reasons']
  d=h.check(123,p,None,190);assert d['reasons']==['stage_heartbeat_unavailable_90s'];assert d['observations']['stage']['errno']==errno.EIO
 hb(p,191);assert not h.check(123,p,None,191)['reasons']

def test_parent_missing_is_immediate(tmp_path,monkeypatch):
 p=tmp_path/'hb';hb(p,1)
 def missing(*a):raise ProcessLookupError(errno.ESRCH,'missing')
 monkeypatch.setattr(s.os,'kill',missing)
 assert s.Health(1).check(123,p,None,1)['reasons']==['parent_missing']

def test_stop_record_precedes_signal_and_targets_only_owned_group(tmp_path,monkeypatch):
 p=tmp_path/'hb';hb(p,1);calls=[]
 monkeypatch.setenv('NCBR_PIPELINE_LEASE',str(p))
 monkeypatch.setattr(s.time,'monotonic',lambda:100)
 monkeypatch.setattr(s.time,'sleep',lambda _:None)
 monkeypatch.setattr(s.os,'kill',lambda *a:None)
 def signal(pid,sig):
  assert (tmp_path/'watchdog_stop.json').exists();calls.append((pid,sig))
 monkeypatch.setattr(s.os,'killpg',signal)
 s.watchdog(123,456,p,tmp_path/'done',tmp_path/'audit')
 assert calls==[(456,15),(456,9)]
 assert json.loads((tmp_path/'audit/watchdog_stop.json').read_text())['reasons']==['stage_heartbeat_expired','pipeline_heartbeat_expired']
