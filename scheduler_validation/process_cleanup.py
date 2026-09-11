"""Teardown only processes carrying an exact, exclusively created instance marker."""
import os
import psutil


def cleanup_owned(runtime_directory):
    marker = str(runtime_directory)
    owned, records, errors = [], [], []
    ignored = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)
    for proc in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        if proc.pid == os.getpid():
            continue
        try:
            if proc.environ().get('TMPDIR') == marker and proc.status() != psutil.STATUS_ZOMBIE:
                owned.append(proc)
                records.append(dict(pid=proc.pid, command=proc.info['cmdline'],
                                    create_time=proc.info['create_time'], marker=marker))
        except ignored:
            pass
    _, alive = psutil.wait_procs(owned, timeout=2)
    for action, timeout in [('terminate', 8), ('kill', 5)]:
        for proc in alive:
            try:
                getattr(proc, action)()
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                errors.append(dict(pid=proc.pid, action=action, error='AccessDenied'))
        _, alive = psutil.wait_procs(alive, timeout=timeout)
    remaining = []
    for proc in alive:
        try:
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                remaining.append(proc.pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            remaining.append(proc.pid)
    return dict(owned=records, remaining=remaining, errors=errors)
