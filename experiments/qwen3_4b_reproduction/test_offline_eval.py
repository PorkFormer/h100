"""CPU protocol and coverage tests; no GPU initialization."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as e


def test_historical_seed():
    assert e.seed(512, 0) == 239391688
    assert e.seed(512, 1) == 1808425106


def test_shards_cover_ordered_requests():
    prompts = [dict(prompt_id=i) for i in range(512, 612)]
    coverage = []
    for shard in range(8):
        jobs = e.requests(prompts, shard)
        assert len(jobs) == 400
        assert [r for _, r in jobs[:4]] == list(range(shard, 32, 8))
        assert all(len(jobs[i:i+8]) == 8 for i in range(0, 400, 8))
        coverage.extend((p['prompt_id'], r) for p, r in jobs)
    assert len(set(coverage)) == 3200


@pytest.mark.parametrize('h', [2048, 8192])
def test_sampling(h):
    assert e.sampling(h, 123) == dict(n=1, temperature=1., top_p=1., top_k=-1,
        repetition_penalty=1., max_tokens=h, ignore_eos=False, seed=123)


class Tokenizer:
    def decode(self, ids, **kwargs):
        return 'correct' if ids == [1] else 'incorrect'


def valid_row():
    return dict(prompt_id=512, rollout=0, horizon=2048, shard=0, seed=e.seed(512, 0),
        checkpoint_sha256='abc', response_token_ids=[1], response_text='correct', finish_reason='stop')


@pytest.mark.parametrize('key,value', [('prompt_id', 513), ('rollout', 1), ('horizon', 8192),
    ('shard', 1), ('seed', 1), ('checkpoint_sha256', 'other'), ('response_text', 'changed'),
    ('response_token_ids', [-1]), ('response_token_ids', [1]*2049), ('finish_reason', 'abort')])
def test_reject_corrupt_generation(key, value):
    row = valid_row()
    row[key] = value
    with pytest.raises(ValueError):
        e.validate_row(row, {'prompt_id': 512}, 0, 2048, 0, 'abc', Tokenizer())


def test_valid_generation():
    e.validate_row(valid_row(), {'prompt_id': 512}, 0, 2048, 0, 'abc', Tokenizer())


def test_bootstrap_question_units():
    assert e.interval([0.25]*30) == [0.25, 0.25]
    assert e.interval([-100.]*40) == [-100., -100.]
    assert e.interval([0, 1]*15) == e.interval([0, 1]*15)


def test_score_complete_and_reject_missing(tmp_path, monkeypatch):
    from verl.utils.reward_score import math_dapo
    monkeypatch.setattr(math_dapo, 'compute_score', lambda solution_str, ground_truth: {'acc': solution_str == 'correct'})
    prompts = [dict(prompt_id=i, benchmark=e.BENCHMARKS[0 if i<542 else 1 if i<572 else 2], ground_truth='1') for i in range(512, 612)]
    monkeypatch.setattr(e, 'validate_prompts', lambda p: p)
    monkeypatch.setattr(e, 'tokenizer', Tokenizer)
    manifest = dict(step=300, checkpoint={'sha256': 'abc'}, protocol=e.read(e.HERE/'provenance/offline_eval.json'))
    e.write(tmp_path/'manifest.json', manifest)
    e.write(tmp_path/'prompts.json', prompts)
    for h in (2048, 8192):
        (tmp_path/f'h{h}').mkdir()
        for shard in range(8):
            rows = []
            for p, r in e.requests(prompts, shard):
                row = valid_row()
                row.update(prompt_id=p['prompt_id'], rollout=r, horizon=h, shard=shard, seed=e.seed(p['prompt_id'], r),
                    response_token_ids=[1] if r == 0 else [2], response_text='correct' if r == 0 else 'incorrect')
                rows.append(e.canonical(row))
            (tmp_path/f'h{h}/shard{shard}.jsonl').write_text('\n'.join(rows)+'\n')
    e.score(tmp_path)
    metrics = e.read(tmp_path/'scores.json')['metrics']
    assert len(metrics) == 6
    assert all(m['avg32'] == 1/32 and m['pass32'] == 1 and m['truncation_fraction'] == 0 for m in metrics)
    (tmp_path/'h8192/shard7.jsonl').write_text('')
    with pytest.raises(ValueError, match='Incomplete'):
        e.score(tmp_path)


def test_compare_requires_same_step(tmp_path, monkeypatch):
    monkeypatch.setenv('EVAL_OUTPUT', str(tmp_path))
    for arm, step in [('dapo', 100), ('ncbr', 300)]:
        e.write(tmp_path/arm/'scores.json', {'manifest': {'step': step}})
    with pytest.raises(ValueError, match='same explicit'):
        e.compare()
