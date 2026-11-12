import importlib.util
from pathlib import Path
import pytest
from omegaconf import OmegaConf

spec=importlib.util.spec_from_file_location('checkpoint_every_step',Path(__file__).with_name('checkpoint_every_step.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def candidate():
    base=OmegaConf.load(m.BASE)
    c=OmegaConf.create(OmegaConf.to_container(base,resolve=True))
    for key,value in m.approved_changes().items():OmegaConf.update(c,key,value)
    return base,c


def test_checkpoint_and_paths_only():
    base,c=candidate();m.verify_delta(base,c)
    assert c.trainer.save_freq==1 and c.trainer.resume_mode=='disable'
    assert c.trainer.total_training_steps==300


@pytest.mark.parametrize('key,value',[
    ('data.train_batch_size',128),('algorithm.pppo.extra_continuations_per_prefix',1),
    ('actor_rollout_ref.actor.optim.lr',2e-6),('trainer.resume_mode','auto'),
    ('trainer.save_freq',50),('data.train_files','different.parquet'),
])
def test_reject_unapproved_changes(key,value):
    base,c=candidate();OmegaConf.update(c,key,value)
    with pytest.raises(ValueError):m.verify_delta(base,c)
