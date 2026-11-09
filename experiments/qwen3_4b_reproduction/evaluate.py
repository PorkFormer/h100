#!/usr/bin/env python3
"""Portable, independent historical AIME24/AIME25/AMC23 evaluation."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BENCHMARKS = ('AIME2024', 'AIME2025', 'AMC23')
FIELDS = ('prompt_id', 'benchmark', 'canonical_prompt', 'rendered_prompt', 'prompt_token_ids', 'ground_truth')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(32 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def env(name):
    value = os.environ.get(name)
    require(bool(value), 'Set ' + name + ' in settings.env / eval_settings.env')
    return value


def seed(prompt_id, rollout):
    payload = f'offline-answer-timing-v1\0{42}\0{prompt_id}\0{rollout}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little') & 0x7fffffff


def requests(prompts, shard):
    return [(p, r) for i, p in enumerate(prompts) for r in range(32) if (i * 32 + r) % 8 == shard]


def sampling(horizon, sample_seed):
    return dict(n=1, temperature=1.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                max_tokens=horizon, ignore_eos=False, seed=sample_seed)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(env('MODEL_PATH'), local_files_only=True, trust_remote_code=False)


def semantic(prompts):
    return digest([{k: p[k] for k in FIELDS} for p in prompts])


def validate_prompts(prompts):
    require(len(prompts) == 100, 'Expected 100 unique prompts')
    require(semantic(prompts) == read(HERE / 'provenance/offline_eval.json')['semantic_sha256'],
            'Prompt/answer/order/tokenizer mismatch with historical protocol')
    return prompts


def prepare_data():
    import math
    import pyarrow.parquet as pq
    out = Path(env('EVAL_DATA'))
    require(not out.exists(), 'EVAL_DATA already exists; choose a new path')
    tok = tokenizer()
    prompts = []
    first = None
    for benchmark, variable, count in zip(BENCHMARKS[:2], ('VAL_FILE_1', 'VAL_FILE_2'), (30, 30)):
        seen = {}
        for row in pq.read_table(env(variable)).to_pylist():
            key = canonical(row['prompt'])
            truth = str(row['reward_model']['ground_truth'])
            if key in seen:
                require(seen[key] == truth, 'Conflicting duplicate answers')
                continue
            seen[key] = truth
            if first is None:
                first = row
            prompts.append(dict(benchmark=benchmark, canonical_prompt=key, ground_truth=truth))
        require(len(seen) == count, 'Unexpected unique count for ' + benchmark)
    raw = env('AMC23_FILE')
    require(sha(raw) == read(HERE / 'provenance/offline_eval.json')['amc23_sha256'], 'Wrong AMC23 revision')
    content = first['prompt'][0]['content']
    question = first['extra_info']['raw_problem']
    require(content.count(question) == 1, 'Cannot derive historical prompt wrapper')
    prefix, suffix = content.split(question)
    rows = pq.read_table(raw).to_pylist()
    require(len(rows) == 40 and len({r['question'] for r in rows}) == 40, 'AMC23 must have 40 unique questions')
    for row in rows:
        answer = row['answer']
        require(not isinstance(answer, bool) and math.isfinite(float(answer)) and float(answer).is_integer(), 'Invalid AMC answer')
        messages = [dict(role='user', content=prefix + row['question'] + suffix)]
        prompts.append(dict(benchmark='AMC23', canonical_prompt=canonical(messages), ground_truth=str(int(float(answer)))))
    for i, p in enumerate(prompts):
        messages = json.loads(p['canonical_prompt'])
        p.update(prompt_id=512+i, rendered_prompt=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                 prompt_token_ids=tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False))
        require(len(p['prompt_token_ids']) <= 1024, 'Prompt exceeds historical input limit')
    validate_prompts(prompts)
    write(out, prompts)
    print(f'Historical semantic hash matched; wrote 100 prompts: {out}')


def checkpoint(path):
    from safetensors import safe_open
    path = Path(path).resolve(strict=True)
    config, base = read(path / 'config.json'), read(Path(env('MODEL_PATH')) / 'config.json')
    for key in ('model_type', 'architectures', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'head_dim', 'eos_token_id',
                'bos_token_id', 'tie_word_embeddings', 'rope_theta'):
        require(config.get(key) == base.get(key), 'Incompatible checkpoint: ' + key)
    files = sorted(path.glob('*.safetensors'))
    require(bool(files), 'Expected completed HF safetensors checkpoint')
    index = path / 'model.safetensors.index.json'
    mapping = read(index)['weight_map'] if index.exists() else None
    require(mapping is not None or len(files) == 1, 'Multiple shards require an index')
    keys = {}
    for file in files:
        with safe_open(file, framework='pt', device='cpu') as weights:
            for key in weights.keys():
                require(key not in keys, 'Duplicate tensor')
                keys[key] = file.name
                weights.get_slice(key).get_shape()
    require(bool(keys) and (mapping is None or mapping == keys), 'Invalid tensor index')
    hashes = {}
    for file in files + [path / 'config.json'] + ([index] if index.exists() else []):
        before = file.stat()
        hashes[file.name] = sha(file)
        after = file.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Checkpoint changed during hashing')
    return dict(path=str(path), sha256=digest(hashes), files=hashes)


def check_inputs():
    import run as training
    training.verify_source()
    prompts = validate_prompts(read(env('EVAL_DATA')))
    tok = tokenizer()
    for p in prompts:
        require(tok.apply_chat_template(json.loads(p['canonical_prompt']), tokenize=True, add_generation_prompt=True, return_dict=False)
                == p['prompt_token_ids'], 'Base tokenizer differs')
    return prompts


def worker(args):
    import vllm
    require(vllm.__version__ == '0.18.0', 'vLLM 0.18.0 required')
    prompts = validate_prompts(read(args.output / 'prompts.json'))
    manifest = read(args.output / 'manifest.json')
    h = args.horizon
    tok = tokenizer()
    llm = vllm.LLM(model=manifest['checkpoint']['path'], tokenizer=env('MODEL_PATH'),
                   tensor_parallel_size=1, dtype='bfloat16', max_model_len=4096 if h == 2048 else 9216,
                   gpu_memory_utilization=0.72, max_num_seqs=8, max_num_batched_tokens=32768,
                   enable_chunked_prefill=True, enable_prefix_caching=True, trust_remote_code=False,
                   seed=42, disable_log_stats=True)
    jobs = requests(prompts, args.shard)
    with (args.output / f'h{h}/shard{args.shard}.jsonl').open('x') as stream:
        for start in range(0, len(jobs), 8):
            batch = jobs[start:start+8]
            outputs = llm.generate([dict(prompt_token_ids=p['prompt_token_ids']) for p, _ in batch],
                                  [vllm.SamplingParams(**sampling(h, seed(p['prompt_id'], r))) for p, r in batch], use_tqdm=False)
            require(len(outputs) == len(batch), 'Incomplete vLLM batch')
            for (p, r), output in zip(batch, outputs, strict=True):
                require(len(output.outputs) == 1, 'Expected one completion')
                result = output.outputs[0]
                ids = list(result.token_ids)
                row = dict(prompt_id=p['prompt_id'], rollout=r, horizon=h, shard=args.shard,
                           seed=seed(p['prompt_id'], r), checkpoint_sha256=manifest['checkpoint']['sha256'],
                           response_token_ids=ids, response_text=tok.decode(ids, skip_special_tokens=True),
                           finish_reason=result.finish_reason)
                stream.write(canonical(row) + '\n')
            stream.flush()
            print(f'h{h} shard{args.shard}: {start+len(batch)}/{len(jobs)}', flush=True)


def launch(arm):
    prompts = check_inputs()
    cp = checkpoint(env(arm.upper() + '_HF'))
    step = int(env('EVAL_STEP'))
    require(step >= 0, 'EVAL_STEP must be nonnegative')
    gpus = [g.strip() for g in env('EVAL_GPUS').split(',')]
    require(1 <= len(gpus) <= 8 and all(gpus) and len(set(gpus)) == len(gpus), 'Use 1..8 distinct GPUs')
    out = Path(env('EVAL_OUTPUT')) / arm
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'prompts.json', prompts)
    write(out / 'manifest.json', dict(arm=arm, step=step, checkpoint=cp, gpus=gpus,
                                    protocol=read(HERE / 'provenance/offline_eval.json')))
    write(out / 'status.json', dict(stage='RUNNING'))
    try:
        for h in (2048, 8192):
            (out / f'h{h}').mkdir()
            for start in range(0, 8, len(gpus)):
                children = []
                try:
                    for shard, gpu in zip(range(start, min(start+len(gpus), 8)), gpus):
                        log = (out / f'h{h}/shard{shard}.log').open('x')
                        try:
                            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker',
                                '--output', str(out.resolve()), '--horizon', str(h), '--shard', str(shard)],
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        finally:
                            log.close()
                        children.append(proc)
                    for proc in children:
                        require(proc.wait() == 0, 'Worker failed; inspect shard logs; use a new output directory to rerun')
                finally:
                    for proc in children:
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                    for proc in children:
                        proc.wait()
        require(checkpoint(cp['path'])['sha256'] == cp['sha256'], 'Checkpoint changed during generation')
        score(out)
        write(out / 'status.json', dict(stage='COMPLETE'))
    except BaseException:
        write(out / 'status.json', dict(stage='FAILED'))
        raise


def interval(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    ix = np.random.default_rng(20260812).integers(0, len(values), size=(10000, len(values)))
    return [float(v) for v in np.quantile(values[ix].mean(axis=1), [.025, .975])]


def validate_row(row, p, r, h, shard, identity, tok):
    require(row['prompt_id'] == p['prompt_id'] and row['rollout'] == r and row['horizon'] == h
            and row['shard'] == shard and row['seed'] == seed(p['prompt_id'], r)
            and row['checkpoint_sha256'] == identity, 'Generation identity/seed/order mismatch')
    ids = row['response_token_ids']
    require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids)
            and len(ids) <= h, 'Invalid response tokens')
    require(row['finish_reason'] in ('stop', 'length'), 'Incomplete/aborted completion')
    require(tok.decode(ids, skip_special_tokens=True) == row['response_text'], 'Response text/token mismatch')


def score(out):
    import numpy as np
    from verl.utils.reward_score import math_dapo
    require(sha(math_dapo.__file__) == read(HERE / 'provenance/offline_eval.json')['verifier_sha256'], 'Verifier changed')
    prompts = validate_prompts(read(out / 'prompts.json'))
    manifest = read(out / 'manifest.json')
    require(manifest['protocol'] == read(HERE / 'provenance/offline_eval.json'), 'Protocol changed')
    tok = tokenizer()
    per_question, metrics = [], []
    for h in (2048, 8192):
        scored = {p['prompt_id']: [] for p in prompts}
        for shard in range(8):
            rows = [json.loads(line) for line in (out / f'h{h}/shard{shard}.jsonl').read_text().splitlines()]
            jobs = requests(prompts, shard)
            require(len(rows) == len(jobs), 'Incomplete or duplicated shard')
            for row, (p, r) in zip(rows, jobs, strict=True):
                validate_row(row, p, r, h, shard, manifest['checkpoint']['sha256'], tok)
                reward = math_dapo.compute_score(solution_str=row['response_text'], ground_truth=p['ground_truth'])
                require(reward['acc'] in (True, False, 0, 1), 'Invalid verifier result')
                scored[p['prompt_id']].append((int(reward['acc']), len(row['response_token_ids']),
                    row['finish_reason'] == 'length' or len(row['response_token_ids']) >= h))
        for benchmark in BENCHMARKS:
            selected = [p for p in prompts if p['benchmark'] == benchmark]
            arr = np.asarray([scored[p['prompt_id']] for p in selected], dtype=float)
            require(arr.shape == (len(selected), 32, 3), 'Coverage mismatch')
            avg, passed = arr[:, :, 0].mean(axis=1), arr[:, :, 0].max(axis=1)
            for p, a, b in zip(selected, avg, passed):
                per_question.append(dict(benchmark=benchmark, horizon=h, prompt_id=p['prompt_id'], avg32=float(a), pass32=float(b)))
            metrics.append(dict(benchmark=benchmark, horizon=h, questions=len(selected), samples=32,
                avg32=float(avg.mean()), avg32_ci95=interval(avg), pass32=float(passed.mean()), pass32_ci95=interval(passed),
                mean_tokens=float(arr[:, :, 1].mean()), p50_tokens=float(np.quantile(arr[:, :, 1], .5)),
                p90_tokens=float(np.quantile(arr[:, :, 1], .9)), truncation_fraction=float(arr[:, :, 2].mean())))
    write(out / 'scores.json', dict(manifest=manifest, per_question=per_question, metrics=metrics))
    import csv
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def compare():
    root = Path(env('EVAL_OUTPUT'))
    a, b = (read(root / arm / 'scores.json') for arm in ('dapo', 'ncbr'))
    require(a['manifest']['step'] == b['manifest']['step'], 'Comparison requires the same explicit checkpoint step')
    require(a['manifest']['protocol'] == b['manifest']['protocol'] == read(HERE / 'provenance/offline_eval.json'), 'Protocol mismatch')
    rows = []
    for h in (2048, 8192):
        for benchmark in BENCHMARKS:
            cells = [[p for p in s['per_question'] if p['horizon'] == h and p['benchmark'] == benchmark] for s in (a, b)]
            expected = list(range(512, 542)) if benchmark == 'AIME2024' else list(range(542, 572)) if benchmark == 'AIME2025' else list(range(572, 612))
            require(all([p['prompt_id'] for p in cell] == expected for cell in cells), 'Paired coverage mismatch')
            row = dict(benchmark=benchmark, horizon=h)
            for metric in ('avg32', 'pass32'):
                delta = [100*(y[metric]-x[metric]) for x, y in zip(*cells, strict=True)]
                row[metric + '_delta_pp'] = sum(delta)/len(delta)
                row[metric + '_delta_ci95_pp'] = interval(delta)
            rows.append(row)
    write(root / 'comparison.json', rows)
    lines = ['# 离线评测：NCBR − DAPO', '', '| 数据集 | H | Avg@32 差值 (pp), 95% CI | pass@32 差值 (pp), 95% CI |', '|---|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['benchmark']} | {r['horizon']} | {r['avg32_delta_pp']:.3f}, {r['avg32_delta_ci95_pp']} | {r['pass32_delta_pp']:.3f}, {r['pass32_delta_ci95_pp']} |")
    (root / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    print(root / 'REPORT_ZH.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare-data')
    sub.add_parser('compare')
    for command in ('check', 'run', 'score'):
        p = sub.add_parser(command)
        p.add_argument('arm', choices=('dapo', 'ncbr', 'all'))
    p = sub.add_parser('_worker')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--horizon', type=int, choices=(2048, 8192), required=True)
    p.add_argument('--shard', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    if args.command == 'prepare-data':
        prepare_data()
    elif args.command == 'compare':
        compare()
    elif args.command == '_worker':
        worker(args)
    else:
        for arm in ('dapo', 'ncbr') if args.arm == 'all' else (args.arm,):
            if args.command == 'run':
                launch(arm)
            elif args.command == 'score':
                score(Path(env('EVAL_OUTPUT')) / arm)
            else:
                check_inputs()
                print(json.dumps(checkpoint(env(arm.upper() + '_HF')), indent=2))
        if args.command == 'run' and args.arm == 'all':
            compare()


if __name__ == '__main__':
    main()
