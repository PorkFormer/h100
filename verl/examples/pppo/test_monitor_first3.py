import copy
import time

import pytest
from omegaconf import OmegaConf
import monitor_first3 as monitor


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, 'require_gates', lambda n: None)
    monkeypatch.setattr(monitor, 'process_identity', lambda pid: 'alive')
    OmegaConf.save(OmegaConf.create({'trainer':{'resume_mode':'disable','total_training_steps':300},
                                    'algorithm':{'pppo':{'stop_after_step':None}}}),tmp_path/'resolved_config.yaml')
    status={'stage':'formal','state':'RUNNING','heartbeat':time.time(),'driver_pid':123,'attempt':str(tmp_path)}
    rows=[{'step':i,'metrics':{'actor/pppo_optimizer_updates_this_outer':16,
        'actor/pppo_optimizer_updates_total':16*i,'pppo/actor_rows':2048,
        'train/retained_prompt_groups':256,'actor/grad_norm':1.,'actor/pg_loss':-.01},
        'timing_seconds':{'actor':1.}} for i in range(1,4)]
    return status,rows


def test_only_formal_counts(case):
    status,rows=case
    for stage in ['smoke-g1','smoke-g8','profile']:
        assert monitor.verify_three(dict(status,stage=stage),rows) is None
    assert monitor.verify_three(status,rows[:2]) is None
    assert monitor.verify_three(status,rows)['actual_optimizer_updates']==48


@pytest.mark.parametrize('bad', ['update','nan','sequence','stale','zero_grad'])
def test_invalid_completion_rejected(case,bad):
    status,rows=copy.deepcopy(case)
    if bad=='update':rows[2]['metrics']['actor/pppo_optimizer_updates_total']=47
    if bad=='nan':rows[1]['metrics']['actor/pg_loss']=float('nan')
    if bad=='sequence':rows[2]['step']=2
    if bad=='stale':status['heartbeat']=0
    if bad=='zero_grad':rows[0]['metrics']['actor/grad_norm']=0
    with pytest.raises(ValueError):monitor.verify_three(status,rows)
