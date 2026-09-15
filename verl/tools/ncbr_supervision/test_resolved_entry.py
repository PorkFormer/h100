"""Control-flow regressions; numerical gates live in ncbr_portable."""
import sys
from types import ModuleType, SimpleNamespace
import pytest
from omegaconf import OmegaConf
from resolved_entry import validate_resolved

@pytest.fixture
def validators(monkeypatch):
    calls=[]
    modules={
        'verl.experimental.natural_continuation_boundary_return.config': {'map_ncbr_config': lambda c: calls.append('map')},
        'verl.experimental.natural_continuation_boundary_return.validation': {'validate_boundary_return_preflight': lambda c, **kw: calls.append('preflight')},
        'verl.utils.config': {'omega_conf_to_dataclass': lambda c: calls.append('typed'), 'validate_config': lambda c, **kw: calls.append('validate')},
        'verl.trainer.ppo.utils': {'need_reference_policy': lambda c:False,'need_critic':lambda c:False},
    }
    for name, attrs in modules.items():
        m=ModuleType(name);m.__dict__.update(attrs);monkeypatch.setitem(sys.modules,name,m)
    monkeypatch.delenv('VLLM_PORT',raising=False)
    return calls

def config():
    return OmegaConf.create({'ray_kwargs':{'ray_init':{'runtime_env':{'env_vars':{}}}},'actor_rollout_ref':{'actor':{},'rollout':{}},'algorithm':{},'data':{'seed':None}})

def test_post_migration_validation_preserves_config(validators):
    c=config();before=OmegaConf.to_container(c,resolve=True)
    assert validate_resolved(c) is c
    assert OmegaConf.to_container(c,resolve=True)==before
    assert validators==['map','preflight','validate','typed','typed','typed']

@pytest.mark.parametrize('location',['inherited','runtime'])
def test_reject_shared_replica_port(validators,monkeypatch,location):
    c=config()
    if location=='inherited':monkeypatch.setenv('VLLM_PORT','46000')
    else:c.ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PORT='46000'
    with pytest.raises(AssertionError,match='VLLM_PORT'):validate_resolved(c)
    assert not validators

@pytest.mark.parametrize('key',['reward_model','custom_reward_function','sandbox_fusion'])
def test_reject_legacy_config(validators,key):
    c=config();c[key]={}
    with pytest.raises(AssertionError,match='post-migration'):validate_resolved(c)
    assert not validators
