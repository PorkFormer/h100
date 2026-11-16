"""Streaming coverage validation, frozen scoring, and question bootstrap."""
import csv
import json
import time
from pathlib import Path

from protocol import (check_environment, contract, jobs, read, require, seed, sha, state, tokenizer,
                      validate_prompts, verifier, write)
from data import tokenizer_identity


def interval(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    ix = np.random.default_rng(20260812).integers(0, len(values), size=(10000, len(values)))
    return [float(x) for x in np.quantile(values[ix].mean(axis=1), [.025, .975])]


def validate_row(row, prompt, horizon, identity, tok):
    require(row['horizon'] == horizon and row['checkpoint_sha256'] == identity, 'Generation identity mismatch')
    rollout = row['rollout']
    require(type(rollout) is int and 0 <= rollout < 32, 'Invalid rollout index')
    require(row['seed'] == seed(prompt['prompt_id'], rollout), 'Wrong sampling seed')
    ids = row['response_token_ids']
    require(isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids) and len(ids) <= horizon, 'Invalid response tokens')
    require(row['finish_reason'] in ('stop', 'length'), 'Incomplete/aborted completion')
    require(tok.decode(ids, skip_special_tokens=True) == row['response_text'], 'Response text/token mismatch')


def aggregate(prompts, horizons, records):
    import numpy as np
    metrics, per_question = [], []
    for h in horizons:
        for benchmark in dict.fromkeys(p['benchmark'] for p in prompts):
            chosen = [p for p in prompts if p['benchmark'] == benchmark]
            arr = np.asarray([[records[(h, p['prompt_id'], r)] for r in range(32)] for p in chosen], dtype=float)
            require(arr.shape == (len(chosen), 32, 3), 'Coverage mismatch')
            avg, passed = arr[:, :, 0].mean(axis=1), arr[:, :, 0].max(axis=1)
            low, high = interval(avg)
            low32, high32 = interval(passed)
            for p, a, b in zip(chosen, avg, passed):
                per_question.append(dict(benchmark=benchmark, horizon=h, prompt_id=p['prompt_id'], avg32=float(a), pass32=float(b)))
            metrics.append(dict(benchmark=benchmark, horizon=h, prompts=len(chosen), trajectories=len(chosen)*32,
                                pass_at_1=float(avg.mean()), pass_at_1_ci_low=low, pass_at_1_ci_high=high,
                                pass_at_32=float(passed.mean()), pass_at_32_ci_low=low32, pass_at_32_ci_high=high32,
                                mean_response_length=float(arr[:, :, 1].mean()),
                                response_p50=float(np.quantile(arr[:, :, 1], .5)), response_p90=float(np.quantile(arr[:, :, 1], .9)),
                                truncation_rate=float(arr[:, :, 2].mean())))
    return metrics, per_question


def calculate(prompts, horizons, identity, rows, tok, compute_score, audit_path):
    by_id = {p['prompt_id']: p for p in prompts}
    records, scored, errors = {}, [], []
    for row in rows:
        h, pid, rollout = row['horizon'], row['prompt_id'], row['rollout']
        require(h in horizons and pid in by_id, 'Unexpected horizon/question')
        key = (h, pid, rollout)
        require(key not in records, 'Duplicate trajectory')
        validate_row(row, by_id[pid], h, identity, tok)
        try:
            reward = compute_score(solution_str=row['response_text'], ground_truth=by_id[pid]['ground_truth'])
            require(isinstance(reward, dict) and type(reward.get('acc')) is bool, 'Invalid verifier accuracy')
            correct = reward['acc']
            require(reward.get('score') == (1.0 if correct else -1.0), 'Invalid verifier score')
        except Exception as exc:
            errors.append(dict(horizon=h, prompt_id=pid, rollout=rollout, error=repr(exc)))
            continue
        count = len(row['response_token_ids'])
        cap = row['finish_reason'] == 'length' or count >= h
        records[key] = (int(correct), count, cap)
        scored.append(dict(horizon=h, prompt_id=pid, rollout_index=rollout, benchmark=by_id[pid]['benchmark'],
                           correct=correct, score=reward['score'], pred=reward.get('pred'),
                           response_token_count=count, hit_cap=cap))
    write(audit_path, dict(errors=errors, verifier_errors=len(errors)))
    require(not errors, f'{len(errors)} verifier errors; never counted as incorrect')
    expected = {(h, p['prompt_id'], r) for h in horizons for p in prompts for r in range(32)}
    require(set(records) == expected, 'Missing or unexpected trajectories')
    metrics, questions = aggregate(prompts, horizons, records)
    return metrics, questions, scored


def run_rows(out, prompts, horizons):
    for h in horizons:
        for shard in range(8):
            expected = jobs(prompts, shard)
            path = out / f'h{h}/shard{shard}.jsonl'
            count = 0
            with path.open() as stream:
                for index, line in enumerate(stream):
                    require(index < len(expected), 'Extra shard trajectory')
                    row = json.loads(line)
                    p, r = expected[index]
                    require((row['prompt_id'], row['rollout'], row['shard'], row['horizon']) == (p['prompt_id'], r, shard, h), 'Shard order/coverage mismatch')
                    count += 1
                    yield row
            require(count == len(expected), 'Incomplete shard')


def score(out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = Path(out)
    attempt = out / 'scoring_attempts' / str(time.time_ns())
    attempt.mkdir(parents=True)
    # Make stale derived results visibly unavailable before attempting a new score.
    for name in ('scores.json', 'metrics.csv', 'metrics.parquet', 'trajectory_scores.parquet', 'REPORT_ZH.md', 'scoring_receipt.json'):
        path = out / name
        if path.exists():
            path.replace(attempt / ('previous_' + name))
    state(out, 'SCORING', attempt=str(attempt))
    try:
        check_environment()
        manifest = read(out / 'manifest.json')
        require(manifest['protocol'] == contract(), 'Protocol changed')
        require((out / 'generation_complete.json').is_file(), 'No verified generation completion')
        prompts = validate_prompts(read(out / 'prompts.json'), manifest['datasets'])
        require(tokenizer_identity(manifest['tokenizer']['path']) == manifest['tokenizer'], 'Tokenizer changed')
        files = [out / f'h{h}/shard{s}.jsonl' for h in manifest['horizons'] for s in range(8)]
        raw_hashes = {str(p.relative_to(out)): sha(p) for p in files}
        completion = read(out / 'generation_complete.json')
        require(completion['raw_sha256'] == raw_hashes, 'Raw generation changed after completion')
        metrics, questions, trajectories = calculate(prompts, manifest['horizons'], manifest['checkpoint']['sha256'],
            run_rows(out, prompts, manifest['horizons']), tokenizer(manifest['tokenizer']['path']), verifier(), attempt / 'audit.json')
        require(raw_hashes == {str(p.relative_to(out)): sha(p) for p in files}, 'Generation changed during scoring')
        write(attempt / 'scores.json', dict(manifest=manifest, per_question=questions, metrics=metrics))
        with (attempt / 'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        pq.write_table(pa.Table.from_pylist(metrics), attempt / 'metrics.parquet')
        pq.write_table(pa.Table.from_pylist(trajectories), attempt / 'trajectory_scores.parquet')
        lines = [f"# {manifest['model_id']} step {manifest['step']} 离线评测", '',
                 '每题32次；pass@1为Avg@32；95% CI按题目bootstrap。各预算独立生成。', '',
                 '| 数据集 | H | pass@1 (95% CI) | pass@32 (95% CI) | 平均token | 截断率 |',
                 '|---|---:|---|---|---:|---:|']
        for m in metrics:
            lines.append(f"| {m['benchmark']} | {m['horizon']} | {m['pass_at_1']:.2%} ({m['pass_at_1_ci_low']:.2%}, {m['pass_at_1_ci_high']:.2%}) | {m['pass_at_32']:.2%} ({m['pass_at_32_ci_low']:.2%}, {m['pass_at_32_ci_high']:.2%}) | {m['mean_response_length']:.1f} | {m['truncation_rate']:.2%} |")
        lines += ['', '硬件、权重及冻结协议见 manifest.json；不同硬件不保证逐token相同。',
                  '评分使用历史 math_dapo 的字符串归一化规则，未切换符号等价判分。']
        (attempt / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
        artifacts = ('scores.json', 'metrics.csv', 'metrics.parquet', 'trajectory_scores.parquet', 'REPORT_ZH.md')
        for name in artifacts:
            (attempt / name).replace(out / name)
        write(out / 'scoring_receipt.json', dict(passed=True, rows=len(trajectories), verifier_errors=0,
              raw_sha256=raw_hashes, artifacts={name: sha(out / name) for name in artifacts}, audit=str(attempt / 'audit.json')))
        state(out, 'COMPLETE', rows=len(trajectories), metrics=len(metrics))
    except BaseException as exc:
        state(out, 'FAILED', phase='scoring', error=repr(exc), attempt=str(attempt))
        raise


def verified_scores(out):
    out = Path(out)
    require(read(out / 'status.json')['stage'] == 'COMPLETE', 'Run is not complete')
    receipt = read(out / 'scoring_receipt.json')
    require(receipt['passed'], 'Scoring receipt failed')
    for name, value in {**receipt['raw_sha256'], **receipt['artifacts']}.items():
        require(sha(out / name) == value, 'Changed result: ' + name)
    return read(out / 'scores.json')


def compare(left, right, output):
    a, b = verified_scores(left), verified_scores(right)
    for key in ('step', 'datasets', 'horizons', 'protocol'):
        require(a['manifest'][key] == b['manifest'][key], 'Comparison mismatch: ' + key)
    require(a['manifest']['protocol'] == contract(), 'Comparison protocol changed')
    rows = []
    for h in a['manifest']['horizons']:
        for name in a['manifest']['datasets']:
            cells = [[p for p in s['per_question'] if p['horizon'] == h and p['benchmark'] == name] for s in (a, b)]
            require([p['prompt_id'] for p in cells[0]] == [p['prompt_id'] for p in cells[1]], 'Paired question mismatch')
            row = dict(benchmark=name, horizon=h)
            for metric in ('avg32', 'pass32'):
                delta = [100 * (y[metric] - x[metric]) for x, y in zip(*cells, strict=True)]
                row[metric + '_delta_pp'] = sum(delta) / len(delta)
                row[metric + '_ci95_pp'] = interval(delta)
            rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'comparison.json', dict(left=str(left), right=str(right), direction='right - left', metrics=rows))
    (output / 'REPORT_ZH.md').write_text('# 配对比较：右侧 − 左侧\n\n```json\n' + json.dumps(rows, ensure_ascii=False, indent=2) + '\n```\n')
