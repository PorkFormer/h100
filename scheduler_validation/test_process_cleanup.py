"""No real process is signalled by this ownership test."""
import process_cleanup as cleanup


def test_exact_marker_only_and_never_self_or_zombies(monkeypatch):
    actions = []
    class Proc:
        def __init__(self, pid, marker, zombie=False):
            self.pid=pid;self.marker=marker;self.zombie=zombie
            self.info=dict(cmdline=['private-test'],create_time=1)
        def environ(self):return {'TMPDIR':self.marker}
        def status(self):return cleanup.psutil.STATUS_ZOMBIE if self.zombie else 'running'
        def terminate(self):actions.append(('terminate',self.pid))
        def kill(self):actions.append(('kill',self.pid))
        def is_running(self):return True
    processes=[Proc(10,'/private/case'),Proc(11,'/private/case_extra'),
               Proc(12,'/private/case',True),Proc(13,'/private/case'),Proc(14,'/other')]
    monkeypatch.setattr(cleanup.os,'getpid',lambda:13)
    monkeypatch.setattr(cleanup.psutil,'process_iter',lambda fields:processes)
    waits=[]
    def wait(procs,timeout):
        waits.append(timeout)
        return ([],procs) if timeout!=5 else (procs,[])
    monkeypatch.setattr(cleanup.psutil,'wait_procs',wait)
    result=cleanup.cleanup_owned('/private/case')
    assert actions==[('terminate',10),('kill',10)]
    assert waits==[2,8,5]
    assert [p['pid'] for p in result['owned']]==[10]
    assert not result['remaining'] and not result['errors']
