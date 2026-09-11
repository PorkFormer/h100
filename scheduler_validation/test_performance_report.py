"""Synthetic unit fixtures validate the reporting gate, never GPU performance."""
import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('performance_report',Path(__file__).with_name('performance_report.py'))
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

def fixture(root,name,seconds,changed=False,timeout=False):
    folder=root/name;folder.mkdir()
    (folder/'result.json').write_text(json.dumps(dict(status='PASS',concurrency=4,continuation_seconds=seconds,tokens=78)))
    requests=[dict(request_id=str(i)) for i in range(78)]
    generations=[dict(request_id=str(i),branch_id=0,tail_token_ids=[2 if changed and i==0 else 1],stop_reason='completed',finish_reason='stop') for i in range(78)]
    intervals=[dict(name='continuation_request',metadata=dict(request_id=str(i)),wall_end=i*3+1) for i in range(78)]
    events=[dict(event=e,request_id=str(i),time=i*3+j) for i in range(78) for j,e in enumerate(['dispatch','release_start','release_ack'])]
    (folder/'capture.json').write_text(json.dumps(dict(requests=requests,generations=generations,intervals=intervals)))
    (folder/'events.json').write_text(json.dumps(events))
    (root/f'{name}_process.json').write_text(json.dumps(dict(code=0,timed_out=timeout,peak_sampled_mib=[1]*8)))

@pytest.mark.parametrize('changed',[False,True])
def test_six_pairs_only_promote_identical_work(tmp_path,monkeypatch,changed):
    monkeypatch.setattr(report,'E',tmp_path)
    for i in range(2):fixture(tmp_path,f'repeat_old_{i}',11)
    for i in range(6):
        fixture(tmp_path,f'scheduler_{i}_a',11)
        fixture(tmp_path,f'scheduler_{i}_b',10,changed=changed and i==0)
    report.report()
    output=json.loads((tmp_path/'performance_report.json').read_text())
    assert output['scheduler']['performance_pass']==(not changed)
    assert output['scheduler']['diagnostic_paired_95_ci']==pytest.approx([1.1,1.1])
    assert output['capacity']['interpretation']=='UNCOVERED'
    report.report()
    assert json.loads((tmp_path/'performance_report.json').read_text())==output

def test_timeout_never_enters_performance_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'E',tmp_path)
    fixture(tmp_path,'case',10,timeout=True)
    with pytest.raises(AssertionError):report.load('case')
