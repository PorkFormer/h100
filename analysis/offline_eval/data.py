"""Download pinned public inputs and reconstruct the historical prompts."""
import json
import math
import os
from pathlib import Path
import urllib.request

from protocol import (canonical, lock, require, selected, sha, tokenizer, validate_prompts, write)


def download(data_dir, names, offline=False):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    spec = lock()
    # AMC23 derives its exact wrapper from the pinned AIME2024 source.
    needed = selected(set(names) | ({'AIME2024'} if 'AMC23' in names else set()))
    for name in needed:
        entry = spec['datasets'][name]
        path = data_dir / entry['file']
        if not path.exists():
            require(not offline, 'Offline input missing: ' + str(path))
            url = f"https://huggingface.co/datasets/{spec['dataset_repo']}/resolve/{spec['dataset_revision']}/{entry['file']}"
            tmp = path.with_name(path.name + f'.{os.getpid()}.part')
            try:
                with urllib.request.urlopen(url, timeout=90) as response, tmp.open('wb') as output:
                    while chunk := response.read(1 << 20):
                        output.write(chunk)
                require(sha(tmp) == entry['sha256'], 'Downloaded input hash mismatch: ' + name)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        require(sha(path) == entry['sha256'], 'Input hash mismatch: ' + str(path))
    return data_dir


def prepare(data_dir, tokenizer_dir, names, offline=False):
    import pyarrow.parquet as pq
    names = selected(names)
    directory = download(data_dir, names, offline)
    for name, expected in lock()['tokenizer_files'].items():
        path = Path(tokenizer_dir) / name
        require(path.is_file() and sha(path) == expected,
                'Historical tokenizer asset mismatch: ' + name + '; use --tokenizer with the frozen Base tokenizer')
    tok = tokenizer(tokenizer_dir)
    spec = lock()['datasets']
    prompts = []
    for name in names:
        rows = pq.read_table(directory / spec[name]['file']).to_pylist()
        messages_answers = []
        if name == 'AMC23':
            first = pq.read_table(directory / spec['AIME2024']['file']).to_pylist()[0]
            content, problem = first['prompt'][0]['content'], first['extra_info']['raw_problem']
            require(content.count(problem) == 1, 'Cannot derive historical AMC wrapper')
            prefix, suffix = content.split(problem)
            for row in rows:
                answer = row['answer']
                require(not isinstance(answer, bool) and math.isfinite(float(answer)) and float(answer).is_integer(), 'Invalid AMC answer')
                messages_answers.append(([dict(role='user', content=prefix + row['question'] + suffix)], str(int(float(answer)))))
        else:
            seen = {}
            for row in rows:
                messages, answer = row['prompt'], str(row['reward_model']['ground_truth'])
                key = canonical(messages)
                if key in seen:
                    require(seen[key] == answer, 'Conflicting duplicate answer')
                    require(name == 'AIME2024', 'Unexpected duplicate question: ' + name)
                    continue
                seen[key] = answer
                messages_answers.append((messages, answer))
        for index, (messages, answer) in enumerate(messages_answers):
            prompts.append(dict(benchmark=name, prompt_id=spec[name]['first_id'] + index,
                                canonical_prompt=canonical(messages), ground_truth=answer,
                                rendered_prompt=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                                prompt_token_ids=tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)))
    return validate_prompts(prompts, names)


def tokenizer_identity(directory):
    directory = Path(directory).resolve(strict=True)
    names = ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'chat_template.jinja', 'special_tokens_map.json', 'added_tokens.json')
    require((directory / 'tokenizer.json').is_file(), 'Tokenizer missing; supply --tokenizer with the historical Base tokenizer')
    return dict(path=str(directory), files={n: sha(directory / n) for n in names if (directory / n).is_file()})


def checkpoint(directory):
    from safetensors import safe_open
    from protocol import digest, read
    directory = Path(directory).resolve(strict=True)
    cfg = read(directory / 'config.json')
    require(cfg.get('model_type') == 'qwen3', 'This frozen protocol supports Qwen3 checkpoints')
    require(cfg.get('eos_token_id') in (151643, [151643]), 'Checkpoint EOS differs from frozen Base protocol')
    # Model dimensions belong to the evaluated model, not to the reference tokenizer.
    for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size'):
        require(isinstance(cfg.get(key), int) and cfg[key] > 0, 'Invalid model config: ' + key)
    rope = cfg.get('rope_theta') or (cfg.get('rope_parameters') or {}).get('rope_theta')
    require(isinstance(rope, (int, float)) and rope > 0, 'Invalid rope_theta')
    files = sorted(directory.glob('*.safetensors'))
    require(bool(files), 'Expected full HF safetensors checkpoint; export FSDP first')
    index = directory / 'model.safetensors.index.json'
    mapping = read(index)['weight_map'] if index.exists() else None
    require(mapping is not None or len(files) == 1, 'Multiple shards require an index')
    require(mapping is None or set(mapping.values()) == {p.name for p in files}, 'Shard coverage mismatch')
    keys, shapes = {}, {}
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as weights:
            for key in weights.keys():
                require(key not in keys, 'Duplicate tensor: ' + key)
                keys[key] = path.name
                shapes[key] = weights.get_slice(key).get_shape()
    require(bool(keys) and (mapping is None or keys == mapping), 'Tensor index mismatch')
    require(shapes.get('model.embed_tokens.weight') == [cfg['vocab_size'], cfg['hidden_size']], 'Embedding/config mismatch')
    for layer in range(cfg['num_hidden_layers']):
        require(any(k.startswith(f'model.layers.{layer}.') for k in keys), 'Missing transformer layer')
    hashes = {}
    generation = directory / 'generation_config.json'
    for path in files + [directory / 'config.json'] + ([index] if index.exists() else []) + ([generation] if generation.exists() else []):
        before = path.stat()
        hashes[path.name] = sha(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Checkpoint changed during hashing')
    return dict(path=str(directory), sha256=digest(hashes), files=hashes)
