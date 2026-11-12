"""CPU-only historical regression adapter; never labels replay as new generation."""
import argparse
import csv
import json
from pathlib import Path

from protocol import ensure_compute_host, read, require, sha, tokenizer, validate_prompts, verifier, write
from results import calculate


def replay(prompts, names, horizon, raw, expected_metrics, tokenizer_dir, output):
    ensure_compute_host()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prompts = validate_prompts(prompts, names)
    by_id = {p['prompt_id']: p for p in prompts}
    raw = Path(raw)
    before = sha(raw)
    with raw.open() as stream:
        first = json.loads(next(stream))
    identity = first['checkpoint_sha256']
    stored = {}

    def rows():
        with raw.open() as stream:
            for line in stream:
                r = json.loads(line)
                if r['benchmark'] not in names:
                    continue
                p = by_id[r['prompt_id']]
                require(r['ground_truth'] == p['ground_truth'] and r['prompt_token_ids'] == p['prompt_token_ids'], 'Historical prompt/answer changed')
                require(not r.get('scoring_error'), 'Unresolved historical scoring error')
                key = (r['prompt_id'], r['rollout_index'])
                require(key not in stored, 'Duplicated historical sample')
                stored[key] = (r['reward_acc'], r['reward_score'])
                yield dict(prompt_id=r['prompt_id'], rollout=r['rollout_index'], seed=r['stable_seed'],
                    horizon=r['horizon'], checkpoint_sha256=r['checkpoint_sha256'], response_token_ids=r['response_token_ids'],
                    response_text=r['response_text'], finish_reason=r['finish_reason'])

    metrics, questions, trajectories = calculate(prompts, [horizon], identity, rows(), tokenizer(tokenizer_dir), verifier(), output/'scoring_audit.json')
    for r in trajectories:
        require(stored[(r['prompt_id'], r['rollout_index'])] == (r['correct'], r['score']), 'Historical reward mismatch')
    with Path(expected_metrics).open() as stream:
        old = {r['benchmark']: r for r in csv.DictReader(stream) if int(r['horizon']) == horizon and r['benchmark'] in names}
    require(set(old) == set(names), 'Missing historical metric rows')
    for row in metrics:
        for key in ('pass_at_1', 'pass_at_1_ci_low', 'pass_at_1_ci_high', 'pass_at_32',
                    'pass_at_32_ci_low', 'pass_at_32_ci_high', 'mean_response_length', 'truncation_rate'):
            require(abs(row[key] - float(old[row['benchmark']][key])) <= 1e-12, 'Historical metric mismatch: ' + row['benchmark'] + '/' + key)
    require(sha(raw) == before, 'Historical file changed during replay')
    result = dict(passed=True, mode='historical_cpu_replay', new_generation=False, rows=len(trajectories),
                  datasets=names, horizon=horizon, raw_sha256=before, stored_reward_mismatches=0, metrics=metrics)
    write(output/'receipt.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts', type=Path, required=True, help='Frozen protocol parquet')
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--horizon', type=int, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ensure_compute_host()
    import pyarrow.parquet as pq
    prompts = [r for r in pq.read_table(a.prompts).to_pylist() if r['benchmark'] in a.datasets]
    print(json.dumps(replay(prompts, a.datasets, a.horizon, a.raw, a.metrics, a.tokenizer, a.output), indent=2))


if __name__ == '__main__':
    main()
