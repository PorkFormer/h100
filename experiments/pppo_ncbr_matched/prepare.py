"""Prepare the immutable PPPO split and full resolved, field-audited recipe."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/workspace/rl/h100')
RUN = ROOT / 'outputs/pppo_ncbr_matched_16xa10040g_s300'
HERE = Path(__file__).resolve().parent
MODEL = Path('/workspace/models/Qwen3-4B-Base')
SOURCE = Path('/workspace/rl/data/dapo_math_17k_train.parquet')
SOURCE_SHA = '816b8778751076fdbc882ce65045aa80bb59db1618ce36e8308903db90ae92f4'
BASELINE = ROOT / 'outputs/qwen3_4b_base_boundary_return_b256_g768_m16_n8_h2048_l8192_s300_v1/provenance/resolved_config.yaml'
CONFIG = REPO / 'verl/examples/pppo/configs/pppo_ncbr_matched_16xa10040g.yaml'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def code_fingerprint():
    digest=hashlib.sha256()
    for root in [REPO/'verl/verl',HERE,REPO/'verl/tests/experimental/pppo']:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py','.yaml'):
                digest.update(str(path.relative_to(REPO)).encode())
                digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def identity(prompt):
    # Normalize whitespace for splitting only; keep original training messages.
    text = json.dumps([{'role': m['role'], 'content': ' '.join(m['content'].split())}
                       for m in prompt], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def prepare_split():
    from verl.utils import hf_tokenizer
    from verl.utils.tokenizer import normalize_token_ids
    directory = RUN / 'data'
    manifest_path = directory / 'split_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest['files'].items():
            if sha(directory / name) != digest:
                raise ValueError(f'Existing split changed: {name}')
        if sha(SOURCE) != manifest['source_sha256']:
            raise ValueError('Source changed')
        return manifest
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Historical source Parquet mismatch')
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = hf_tokenizer(str(MODEL))
    table = pq.read_table(SOURCE)
    groups, row_keys = {}, []
    for chunk in table.to_batches(max_chunksize=16384):
        for row in chunk.to_pylist():
            key = identity(row['prompt'])
            index = len(row_keys)
            row_keys.append(key)
            if key not in groups:
                ids = normalize_token_ids(tokenizer.apply_chat_template(row['prompt'], tokenize=True,
                                                                         add_generation_prompt=True))
                groups[key] = dict(rows=[], first=index, prompt_tokens=len(ids),
                                   truths=set())
            group = groups[key]
            group['truths'].add(str(row['reward_model']['ground_truth']))
            group['rows'].append(index)
    test_keys = set()
    test_hashes = {}
    for path in [Path('/workspace/rl/data/aime-2024-verl.parquet'),
                 Path('/workspace/rl/data/aime-2025-verl.parquet')]:
        test_hashes[str(path)] = sha(path)
        for prompt in pq.read_table(path, columns=['prompt'])['prompt'].to_pylist():
            test_keys.add(identity(prompt))
    # The frozen historical offline prompts include AMC23 in its actual template.
    frozen = ROOT / 'analysis/ncbr_offline_eval/templates/prompts.parquet'
    if not frozen.exists():
        raise FileNotFoundError('Frozen three-benchmark prompts required for holdout overlap check')
    test_hashes[str(frozen)] = sha(frozen)
    for row in pq.read_table(frozen).to_pylist():
        if 'canonical_prompt' in row:
            test_keys.add(identity(json.loads(row['canonical_prompt'])))
        elif 'prompt' in row:
            prompt = row['prompt']
            test_keys.add(identity(json.loads(prompt) if isinstance(prompt, str) else prompt))
        else:
            raise ValueError(f'Cannot identify frozen test prompt fields: {list(row)}')
    eligible = [key for key, value in groups.items() if value['prompt_tokens'] <= 1024
                and len(value['truths']) == 1 and key not in test_keys]
    chosen = sorted(eligible, key=lambda key: hashlib.sha256(f'pppo-holdout-v1\0{42}\0{key}'.encode()).hexdigest())[:256]
    if len(chosen) != 256:
        raise ValueError('Insufficient isolated holdout questions')
    selected = set(chosen)
    train_indices = [i for i, key in enumerate(row_keys) if key not in selected]
    # Do not deduplicate or reorder training; RLDataSet applies the historical prompt filter.
    pq.write_table(table.take(pa.array(train_indices)), directory / 'train.parquet', compression='zstd')
    pq.write_table(table.take(pa.array([groups[key]['first'] for key in chosen])),
                   directory / 'holdout.parquet', compression='zstd')
    np.save(directory / 'train_source_rows.npy', np.asarray(train_indices, dtype=np.int64))
    effective_rows = sum(len(v['rows']) for k, v in groups.items() if k not in selected and v['prompt_tokens'] <= 1024)
    manifest = dict(version=1, seed=42, identity='SHA256 whitespace-normalized role/content messages',
                    source=str(SOURCE), source_sha256=SOURCE_SHA, source_rows=len(row_keys),
                    unique_questions=len(groups), training_rows=len(train_indices),
                    ambiguous_source_questions={k: sorted(v['truths']) for k,v in groups.items() if len(v['truths']) != 1},
                    ambiguous_question_policy='Preserve historical training rows; exclude ambiguous identities from holdout eligibility',
                    effective_training_rows=effective_rows, holdout_questions=256,
                    holdout=[dict(question_hash=k, source_rows=groups[k]['rows'],
                                  prompt_tokens=groups[k]['prompt_tokens']) for k in chosen],
                    final_test_files=test_hashes, exact_normalized_test_overlap=0,
                    historical_ncbr_excluded_holdout=False,
                    files={name: sha(directory / name) for name in
                           ['train.parquet', 'holdout.parquet', 'train_source_rows.npy']})
    write_json(manifest_path, manifest)
    return manifest


def flatten(value, prefix=''):
    result = {}
    if isinstance(value, dict) and value:
        for key, child in value.items():
            result.update(flatten(child, f'{prefix}.{key}' if prefix else key))
    else:
        result[prefix] = value
    return result


def prepare_config(split):
    baseline = OmegaConf.load(BASELINE)
    config = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
    with initialize_config_dir(config_dir=str(REPO / 'verl/verl/trainer/config'), version_base=None):
        defaults = compose(config_name='pppo_trainer')
    changes = {}

    def set_value(key, value, category, reason):
        OmegaConf.update(config, key, value, merge=False, force_add=True)
        changes[key] = dict(category=category, reason=reason)

    algorithm = 'intentional_algorithm_differences'
    protocol = 'environment_or_protocol_differences'
    set_value('algorithm.pppo', OmegaConf.to_container(defaults.algorithm.pppo), algorithm,
              'PPPO accumulated binary reward, prefix objective, deterministic independent continuations and disclosed plateau completion')
    for key, value in {
        'data.max_response_length':8192, 'actor_rollout_ref.rollout.response_length':8192,
        'actor_rollout_ref.actor.policy_loss.loss_mode':'pppo', 'actor_rollout_ref.actor.clip_ratio_c':None,
        'actor_rollout_ref.rollout.boundary_return.mode':'off',
        'reward.reward_kwargs.overlong_buffer_cfg.enable':False,
        'reward.reward_kwargs.max_resp_len':8192,
        'algorithm.filter_groups.metric':'pppo_filter_reward',
    }.items():
        set_value(key,value,algorithm,'PPPO + DAPO Dynamic Sampling: original 8K responses, accumulated binary correctness, no NCBR shaping or dual clip')
    for key, value in {
        'algorithm.filter_groups.enable':True, 'algorithm.filter_groups.max_num_gen_batches':10,
        'data.train_batch_size':256, 'data.gen_batch_size':768,
        'actor_rollout_ref.rollout.n':8, 'trainer.nnodes':2, 'trainer.n_gpus_per_node':8,
        'actor_rollout_ref.actor.ppo_mini_batch_size':16, 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':1,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':2,
        'trainer.total_training_steps':300, 'trainer.save_freq':50, 'trainer.test_freq':10,
    }.items():
        set_value(key,value,protocol,'Required NCBR-matched invariant')
    set_value('actor_rollout_ref.actor.fsdp_config.pppo_fixed_linear_tile',128,protocol,
              'BF16 shape-stable 128-token linear execution fixes full/cropped projection and update discrepancies; parameters and objective unchanged')
    set_value('actor_rollout_ref.actor.fsdp_config.pppo_bf16_full_reduction',True,protocol,
              'Disable BF16 reduced-precision intermediate reduction; BF16 operands and FP32 master dtype unchanged')
    set_value('algorithm.pppo.holdout_hash',split['files']['holdout.parquet'],protocol,'Freeze isolated validation split')
    set_value('data.train_files',str(RUN/'data/train.parquet'),protocol,'Exclude every copy of holdout questions; preserve remaining repeats and order')
    set_value('data.val_files',[str(RUN/'data/holdout.parquet')],protocol,'Independent 256-question holdout; never tune eta on final tests')
    set_value('trainer.val_before_train',True,protocol,'Step zero holdout baseline required by disclosed plateau rule')
    for key,value in {'n':1,'temperature':0.0,'do_sample':False}.items():
        set_value('actor_rollout_ref.rollout.val_kwargs.'+key,value,protocol,'Fixed greedy holdout validation every ten steps')
    batches=split['effective_training_rows']//768
    if batches < 1:
        raise ValueError('No full candidate batch')
    set_value('trainer.total_epochs',math.ceil(300*10/batches),protocol,'Enough data epochs for worst-case ten candidate batches per outer step; stop remains 300')
    for key,value in {
        'trainer.resume_mode':'disable','trainer.resume_from_path':None,
        'trainer.default_local_dir':str(RUN/'checkpoints'),
        'trainer.experiment_name':RUN.name,
        'trainer.nondeterminism_diagnostics.enabled':False,
        'ray_kwargs.ray_init.address':'10.8.191.127:6395',
        'ray_kwargs.ray_init._node_ip_address':'10.8.191.127',
    }.items():
        set_value(key,value,protocol,'Fresh isolated PPPO experiment; retain original cluster topology')
    environment = {
        'PYTHONPATH':str(REPO/'verl'),'PYTHONDONTWRITEBYTECODE':'1',
        'XDG_CACHE_HOME':'/tmp/pppo-matched-cache','FLASHINFER_WORKSPACE_BASE':'/tmp/pppo-matched-cache/flashinfer',
        'GLOO_SOCKET_IFNAME':'ens201',
        'NCCL_SOCKET_IFNAME':'=ens101,ens102,ens103,ens104,ens105,ens106,ens107,ens108',
        'NCCL_IB_HCA':'=mlx5_101:1,mlx5_102:1,mlx5_103:1,mlx5_104:1,mlx5_105:1,mlx5_106:1,mlx5_107:1,mlx5_108:1',
        'NCCL_IB_DISABLE':'0','NCCL_SOCKET_FAMILY':'AF_INET','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1',
        'FLASH_ATTENTION_DETERMINISTIC':'1',
        'WANDB_PROJECT':'a100_verl','WANDB_ENTITY':'cartman','WANDB_MODE':'offline',
        'WANDB_NAME':RUN.name,'WANDB_RUN_ID':'pppo_matched_16a100_s300','WANDB_RESUME':'never',
        'WANDB_DIR':str(RUN/'wandb'),'TOKENIZERS_PARALLELISM':'false',
    }
    set_value('ray_kwargs.ray_init.runtime_env.env_vars',environment,protocol,
              'Isolated code/cache/log identity and inherited NCBR networking; W&B offline because online disclosure was rejected by automatic approval review')
    set_value('ray_kwargs.ray_init.runtime_env.env_vars.FLASH_ATTENTION_DETERMINISTIC','1',protocol,
              'Deterministic FlashAttention backward removes same-view gradient/update noise in BF16 gate')
    set_value('reward.reward_model.rollout.name','vllm',protocol,
              'Resolve inherited missing engine name in disabled reward-model branch; reward model remains disabled')
    OmegaConf.resolve(config)
    before, after = flatten(OmegaConf.to_container(baseline,resolve=True)), flatten(OmegaConf.to_container(config,resolve=True))
    differences=[]
    for key in sorted(set(before)|set(after)):
        if key in before and key in after and before[key]==after[key]: continue
        owners=[p for p in changes if key==p or key.startswith(p+'.')]
        if not owners: raise ValueError('Unapproved config difference: '+key)
        owner=max(owners,key=len)
        differences.append(dict(field=key,before=before.get(key),after=after.get(key),
                                before_present=key in before,after_present=key in after,**changes[owner]))
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    text=OmegaConf.to_yaml(config,resolve=True)
    if '${' in text or '???' in text or '__MODEL_PATH__' in text:
        raise ValueError('Unresolved formal YAML')
    CONFIG.write_text(text)
    write_json(HERE/'config_diff.json',dict(baseline_path=str(BASELINE),baseline_sha256=sha(BASELINE),
               config_sha256=sha(CONFIG),differences=differences,unapproved_differences=[]))
    (HERE/'reference').mkdir(exist_ok=True)
    (HERE/'reference/ncbr_historical_resolved.yaml').write_text(BASELINE.read_text())
    write_json(HERE/'split_summary.json',{k:v for k,v in split.items() if k!='holdout'})
    print('Frozen config:',CONFIG,'sha256:',sha(CONFIG),flush=True)


if __name__=='__main__':
    prepare_config(prepare_split())
