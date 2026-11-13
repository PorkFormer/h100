import importlib.util
from pathlib import Path
import json
import pytest

spec=importlib.util.spec_from_file_location('extended_eval',Path(__file__).with_name('evaluate.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


@pytest.mark.parametrize('count',[100,500,600])
def test_shard_coverage(count):
    ps=[dict(prompt_id=512+i) for i in range(count)]
    jobs=[(p['prompt_id'],r) for s in range(8) for p,r in e.historical.requests(ps,s)]
    assert len(jobs)==count*32 and len(set(jobs))==count*32
    assert len(e.historical.requests(ps,0))==count*4


def test_seed_golden():
    import hashlib
    for pid in range(512,1112):
        for r in range(32):
            expected=int.from_bytes(hashlib.sha256(f'offline-answer-timing-v1\0{42}\0{pid}\0{r}'.encode()).digest()[:4],'little')&0x7fffffff
            assert e.historical.seed(pid,r)==expected


@pytest.mark.parametrize('answer',['42',r'\frac{1}{2}',r'\sqrt{2}','x+1',r'(1,2)'])
def test_verifier_canonical_answers(answer):
    from verl.utils.reward_score import math_dapo
    assert math_dapo.compute_score('Reasoning.\nAnswer: '+answer,answer)['acc']
    assert not math_dapo.compute_score('Answer: obviously_wrong',answer)['acc']


def test_default_verifier_not_boxed_only():
    from verl.utils.reward_score import math_dapo
    assert math_dapo.compute_score('Answer: 42','42')['acc']
    assert not math_dapo.compute_score('No final answer','42')['acc']


class Tokenizer:
    def decode(self,ids,skip_special_tokens=True):return 'ok'


def valid_row():
    return dict(prompt_id=612,rollout=0,horizon=8192,shard=0,seed=e.historical.seed(612,0),
        checkpoint_sha256='cp',response_token_ids=[1,2],response_text='ok',finish_reason='stop')


@pytest.mark.parametrize('key,value', [('prompt_id',613),('rollout',1),('horizon',2048),('shard',1),
    ('seed',0),('checkpoint_sha256','wrong'),('response_token_ids',[-1]),('response_text','different'),('finish_reason','abort')])
def test_bad_generation_fails(key,value):
    row=valid_row();row[key]=value
    with pytest.raises(ValueError):e.historical.validate_row(row,{'prompt_id':612},0,8192,0,'cp',Tokenizer())


def test_good_generation():
    e.historical.validate_row(valid_row(),{'prompt_id':612},0,8192,0,'cp',Tokenizer())


def test_long_generation_fails():
    row=valid_row();row['response_token_ids']=[1]*8193
    with pytest.raises(ValueError):e.historical.validate_row(row,{'prompt_id':612},0,8192,0,'cp',Tokenizer())
