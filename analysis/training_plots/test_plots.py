"""CPU-only contract tests; use scheduler on HPC4."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from reader import Reader
from metrics import B, group, value, spec
from cli import average, normalize, series, render


def append(path, obj):
    with path.open('ab') as f: f.write(json.dumps(obj).encode()+b'\n')


def test_incremental_partial_merge_conflict_and_parse(tmp_path):
    path = tmp_path/'log'
    path.write_bytes(b'{"step":1,"data":{"x":1}}\n{"step":1,')
    r = Reader('test', path)
    r.poll()
    assert len(r.records) == 1 and r.pending
    with path.open('ab') as f: f.write(b'"data":{"x":2,"y":3}}\nBAD\n')
    r.poll()
    assert r.records[0]['data'] == {'x':2, 'y':3}
    assert [e['kind'] for e in r.events] == ['field_conflict', 'parse_error']
    r.poll()
    assert len(r.events) == 2
    append(path, {'step':0, 'data':{'x':4}})
    r.poll()
    assert r.records[-1]['segment'] == 1


def test_rotation_truncation_and_same_size_rewrite(tmp_path):
    p = tmp_path/'log'
    append(p, {'step':1, 'data':{'x':1}})
    r = Reader('r', p)
    r.poll()
    p.write_bytes(b'{"step":2,"data":{}}\n')
    r.poll()
    assert r.records[-1]['segment'] == 1
    p.rename(tmp_path/'old')
    append(p, {'step':3, 'data':{'x':3}})
    r.poll()
    assert r.records[-1]['segment'] == 2
    p.write_text(p.read_text().replace('3', '4'))
    r.poll()
    assert r.records[-1]['segment'] == 3
    assert len(r.generations) == 3


def test_denominators_and_native_acceptance():
    assert spec('timing_s/agent_loop/slowest/response_length')['unit'] == 'tokens'
    assert spec('timing_s/agent_loop/slowest/generate_sequences')['unit'] == 's'
    k = B+'recovered_rate_given_cap_failure'
    assert value({k:0}, k)[0] is None
    data = {k:.5, B+'transition_h_wrong_l_wrong_count':2, B+'transition_h_wrong_l_correct_count':2}
    assert value(data,k)[0] == .5
    data = {B+'newly_locked_group_rate':0, 'train/generated_prompt_groups':8, B+'short_locked_group_count':8}
    assert value(data,B+'newly_locked_group_rate')[0] is None
    assert value({B+'continuation_decode_engine_seconds_union':0}, B+'continuation_decode_engine_seconds_union')[0] is None
    assert value({'timing_s/agent_loop/num_preempted/mean':-4}, 'timing_s/agent_loop/num_preempted/mean')[0] is None
    k = 'train/dynamic_sampling_accept_rate'
    assert value({k:.4, 'train/generated_prompt_groups':10, 'train/retained_prompt_groups':2}, k)[0] == .4
    for name, denominator in [('unlocked_group_rate','short_locked_group_count'),
                              ('long_success_rate_given_cap','valid_long_score_count'),
                              ('transition_h_wrong_l_wrong_rate','valid_long_score_count')]:
        assert value({B+name:0,B+denominator:0}, B+name)[0] is None
        assert value({B+name:.5,B+denominator:2}, B+name)[0] == .5


def test_gaps_segments_elapsed_and_raw(tmp_path):
    p = tmp_path/'log'
    for step, data in [(0,{'val-core/math_dapo/acc/mean@1':.2}), (1,{'timing_s/step':10,'actor/entropy':1}),
                       (2,{'timing_s/step':20}), (4,{'actor/entropy':3}), (0,{'timing_s/step':2})]:
        append(p, dict(step=step,data=data))
    r = Reader('r', p)
    r.poll()
    key = 'actor/entropy'
    s = series([r], [key], 'step')
    assert s[('r',0,key)][1] == [None,1,None,None,3]
    rows = normalize([r],[key])
    assert [row['recorded_seconds'] for row in rows] == [0,10,30,30,2]
    assert rows[0]['value'] is None
    assert average([1]*10) == [None]*9+[1]
    assert average([1]*9+[None]+[1]*10)[-1] == 1
    assert group('val-core/math_dapo/acc/mean@1') == 1


def test_export_offline_html_csv_and_static(tmp_path):
    p = tmp_path/'log'
    append(p, {'step':1,'data':{'critic/rewards/mean':-.5,'timing_s/step':5}})
    r = Reader('DAPO',p)
    r.poll()
    args = SimpleNamespace(output=str(tmp_path/'plots'),axis='step',names={},smooth=True,interval=30)
    render([r],args)
    out = Path(args.output)
    text = (out/'index.html').read_text()
    assert '<script src=' not in text
    assert 'plotly.js' in text
    assert '未记录或不可用' in text
    assert len(list(out.glob('*.png'))) == len(list(out.glob('*.pdf'))) == 12
    for p in out.glob('*.png'): assert p.read_bytes().startswith(b'\x89PNG')
    for p in out.glob('*.pdf'): assert p.read_bytes().startswith(b'%PDF')
    assert 'raw_value,value,status' in (out/'metrics.csv').read_text()
    assert json.loads((out/'audit.json').read_text())['inputs'][0]['records'] == 1
