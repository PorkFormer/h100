"""Frozen 8K/32-sample PPPO Step20 and historical Step300 MATH500 extension.

No training controls or background monitoring. All generation is independent of training.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
HERE = REPO / 'experiments/pppo_ncbr_matched'
sys.path.insert(0, str(HERE))
import historical_eval as historical
validate_historical_prompts = historical.validate_prompts
ROOT = Path('/workspace/rl/h100')
RUN = ROOT / 'outputs/pppo_ncbr_matched_16xa10040g_s300_ckpt1_v2'
OUT = RUN / 'offline_eval_step20_h8192_math500_v1'
MODEL = Path('/workspace/models/Qwen3-4B-Base')
MATH = ROOT / 'datasets/math500/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.parquet'
MATH_SHA = '60270068a9b1735a15602b13a10bce4b5995efd2724be94c58f0552cabdb560c'
BENCHMARKS = ('AIME2024', 'AIME2025', 'AMC23', 'MATH500')
ARMS = {
    'pppo_s20': (20, RUN / 'checkpoints/global_step_20/actor/huggingface'),
    'ncbr_s300': (300, ROOT / 'outputs/qwen3_4b_base_boundary_return_b256_g768_m16_n8_h2048_l8192_s300_v1/checkpoints/global_step_300/actor/huggingface'),
    **{f'dapo{h}_s300': (300, Path(f'/workspace/train_rl/qwen_3_4b/dapo/runs/qwen3_4b_base_dapo_b256_g768_m16_n8_r{h}_s600_v1/checkpoints/global_step_300/actor/huggingface')) for h in (2048,4096,8192)},
}
read, write, sha, require = historical.read, historical.write, historical.sha, historical.require
os.environ.setdefault('MODEL_PATH',str(MODEL))


def protocol():
    return dict(historical=read(HERE/'provenance/offline_eval.json'),horizon=8192,samples=32,
        math500=dict(revision='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',sha256=MATH_SHA,id_start=612,count=500),
        extension_source_sha256=sha(Path(__file__)),historical_evaluator_sha256=sha(Path(historical.__file__)),
        sampling=historical.sampling(8192,42),engine=dict(dtype='bfloat16',tp=1,context=9216,
        gpu_memory_utilization=.72,max_num_seqs=8,max_num_batched_tokens=32768,shards=8,batch=8),
        verifier='Unchanged math_dapo.compute_score default args, full generated response as input; normalized string equivalence, not general symbolic algebra')


def verify_sources():
    from verl.utils.reward_score import math_dapo
    require(sha(math_dapo.__file__)==protocol()['historical']['verifier_sha256'],'Historical verifier changed')
    require(sha(MATH)==MATH_SHA,'MATH500 source changed')


def norm(text):
    return ' '.join(text.split())


def prepare():
    import pyarrow.parquet as pq
    verify_sources()
    require(not (OUT/'prompts.json').exists(),'Prepared input exists; do not overwrite')
    frozen=ROOT/'analysis/ncbr_offline_eval/templates/prompts.parquet'
    require(sha(frozen)==protocol()['historical']['historical_prompts_sha256'],'Historical prompts changed')
    prompts=[{k:(list(p[k]) if k=='prompt_token_ids' else p[k]) for k in historical.FIELDS} for p in pq.read_table(frozen).to_pylist()]
    validate_historical_prompts(prompts)
    tok=historical.tokenizer()
    first=pq.read_table('/workspace/rl/data/aime-2024-verl.parquet').slice(0,1).to_pylist()[0]
    content=first['prompt'][0]['content'];question=first['extra_info']['raw_problem']
    require(content.count(question)==1,'Ambiguous wrapper')
    prefix,suffix=content.split(question)
    rows=pq.read_table(MATH).to_pylist()
    require(len(rows)==500 and len({r['unique_id'] for r in rows})==500 and len({norm(r['problem']) for r in rows})==500,'MATH500 duplicate or missing question')
    for i,row in enumerate(rows):
        require(isinstance(row['answer'],str) and row['answer'].strip(),'Missing MATH answer')
        messages=[dict(role='user',content=prefix+row['problem']+suffix)]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
        require(len(ids)<=1024,'MATH500 prompt exceeds historical context allowance; do not truncate/filter')
        prompts.append(dict(prompt_id=612+i,benchmark='MATH500',canonical_prompt=historical.canonical(messages),
            rendered_prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
            prompt_token_ids=ids,ground_truth=row['answer']))
    # String-level overlap audit, not a semantic-leakage guarantee.
    math_ids={norm(r['problem']):r['unique_id'] for r in rows};overlaps={}
    for name,path in [('pppo_train',ROOT/'outputs/pppo_ncbr_matched_16xa10040g_s300/data/train.parquet'),
                      ('pppo_holdout',ROOT/'outputs/pppo_ncbr_matched_16xa10040g_s300/data/holdout.parquet'),
                      ('historical_train',Path('/workspace/rl/data/dapo_math_17k_train.parquet'))]:
        found=Counter()
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32768,columns=['extra_info']):
            for record in batch.to_pylist():
                key=norm(record['extra_info'].get('raw_problem',''))
                if key in math_ids:found[math_ids[key]]+=1
        overlaps[name]=dict(source=str(path),matching_question_ids=dict(found),matching_unique=len(found))
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'prompts.json',prompts)
    write(OUT/'protocol.json',protocol())
    write(OUT/'input_manifest.json',dict(semantic_sha256=historical.semantic(prompts),
        historical100_sha256=historical.semantic(prompts[:100]),math500_sha256=historical.semantic(prompts[100:]),
        max_prompt_tokens=max(len(p['prompt_token_ids']) for p in prompts),questions=600,
        math_source=str(MATH),math_sha256=MATH_SHA,overlap_audit=overlaps,
        limitations='Normalized exact raw-problem matches only. Full official test retained even if overlaps exist; disclose counts.'))
    print(json.dumps(read(OUT/'input_manifest.json'),indent=2),flush=True)


def inputs(arm,diagnostic=False):
    prompts=read(OUT/'prompts.json');m=read(OUT/'input_manifest.json')
    require(historical.semantic(prompts)==m['semantic_sha256'],'Frozen prompts changed')
    validate_historical_prompts(prompts[:100])
    require(len(prompts)==600 and [p['prompt_id'] for p in prompts]==list(range(512,1112)),'Question IDs changed')
    require(protocol()==read(OUT/'protocol.json'),'Evaluation code/protocol changed since preparation')
    if diagnostic:
        return prompts[:4]+prompts[100:104]
    return prompts if arm=='pppo_s20' else prompts[100:]


def validate_selection(prompts,arm,diagnostic=False):
    require(prompts==inputs(arm,diagnostic),'Prompt subset changed')
    return prompts


def idle_head():
    import socket
    require(socket.gethostname().split('.')[0]=='p-kt-dgx-a100-03','Run evaluation on authorized head')
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[list(map(int,l.split(','))) for l in output.splitlines()]
    require(len(rows)==8 and all(mem<=100 and util==0 for mem,util in rows),'Head GPUs not idle; never compete with training')


def worker(out,shard):
    manifest=read(out/'manifest.json')
    verify_sources()
    require(manifest['protocol']==protocol(),'Manifest protocol mismatch')
    historical.validate_prompts=lambda ps:validate_selection(ps,manifest['arm'],manifest['diagnostic'])
    historical.worker(argparse.Namespace(output=out,horizon=8192,shard=shard))


def score(out):
    import numpy as np
    from verl.utils.reward_score import math_dapo
    verify_sources();manifest=read(out/'manifest.json');require(manifest['protocol']==protocol(),'Protocol mismatch')
    prompts=validate_selection(read(out/'prompts.json'),manifest['arm'],manifest['diagnostic'])
    tok=historical.tokenizer();values={p['prompt_id']:[] for p in prompts}
    def timeout(_signum,_frame):raise TimeoutError('Verifier exceeded 10 seconds; do not assign reward zero')
    prior=signal.signal(signal.SIGALRM,timeout)
    try:
        for shard in range(8):
            jobs=historical.requests(prompts,shard)
            with (out/f'h8192/shard{shard}.jsonl').open() as stream:
                for (p,r),line in zip(jobs,stream,strict=True):
                    row=json.loads(line);historical.validate_row(row,p,r,8192,shard,manifest['checkpoint']['sha256'],tok)
                    signal.setitimer(signal.ITIMER_REAL,10)
                    try:reward=math_dapo.compute_score(solution_str=row['response_text'],ground_truth=p['ground_truth'])
                    finally:signal.setitimer(signal.ITIMER_REAL,0)
                    require(reward['acc'] in (True,False,0,1),'Invalid verifier result')
                    values[p['prompt_id']].append((int(reward['acc']),len(row['response_token_ids']),int(row['finish_reason']=='length' or len(row['response_token_ids'])==8192)))
    finally:
        signal.signal(signal.SIGALRM,prior)
    metrics=[];questions=[]
    for name in BENCHMARKS:
        selected=[p for p in prompts if p['benchmark']==name]
        if not selected:continue
        arr=np.asarray([values[p['prompt_id']] for p in selected],dtype=float)
        require(arr.shape==(len(selected),32,3),'Coverage mismatch')
        avg=arr[:,:,0].mean(axis=1);passed=arr[:,:,0].max(axis=1);length=arr[:,:,1].mean(axis=1)
        se=lambda a:float(np.std(a,ddof=1)/np.sqrt(len(a))) if len(a)>1 else None
        for p,a,b,l in zip(selected,avg,passed,length,strict=True):
            questions.append(dict(benchmark=name,prompt_id=p['prompt_id'],avg32=float(a),pass1=float(a),pass32=float(b),mean_tokens=float(l)))
        metrics.append(dict(benchmark=name,questions=len(selected),samples=32,pass1=float(avg.mean()),avg32=float(avg.mean()),
            pass1_se=se(avg),avg32_se=se(avg),avg32_ci95=historical.interval(avg),pass32=float(passed.mean()),
            pass32_se=se(passed),pass32_ci95=historical.interval(passed),mean_tokens=float(length.mean()),
            mean_tokens_se=se(length),mean_tokens_ci95=historical.interval(length),p50_tokens=float(np.quantile(arr[:,:,1],.5)),
            p90_tokens=float(np.quantile(arr[:,:,1],.9)),truncation_fraction=float(arr[:,:,2].mean())))
    result=dict(manifest=manifest,per_question=questions,metrics=metrics,definitions=dict(
        pass1='Unbiased per-question c/32, averaged over questions; equals Avg@32',pass32='Fraction of questions with at least one correct sample',
        ci='10000 question bootstrap resamples, seed20260812; not training-seed uncertainty'))
    write(out/'scores.json',result)
    with (out/'metrics.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    return result


def preflight():
    verify_sources();inputs('pppo_s20')
    tok=historical.tokenizer()
    for p in read(OUT/'prompts.json'):
        require(tok.apply_chat_template(json.loads(p['canonical_prompt']),tokenize=True,add_generation_prompt=True,return_dict=False)==p['prompt_token_ids'],'Tokenization drift')
    records={}
    for arm,(step,path) in ARMS.items():
        if arm=='pppo_s20' and not path.exists():records[arm]=dict(status='WAITING_FOR_STEP20',step=20);continue
        records[arm]=dict(status='PASS',step=step,checkpoint=historical.checkpoint(path))
    write(OUT/'checkpoint_preflight.json',records);print(json.dumps(records,indent=2),flush=True)


def run_arm(arm,diagnostic=False):
    require(read(RUN/'pause_step20_receipt.json')['status']=='PAUSED_AT_STEP20','Must pause and validate Step20 before evaluation')
    idle_head();verify_sources();prompts=inputs(arm,diagnostic)
    step,path=ARMS[arm];cp=historical.checkpoint(path)
    expected=read(OUT/'checkpoint_preflight.json')[arm]
    require(expected['status']=='PASS' and expected['checkpoint']['sha256']==cp['sha256'],'Preflight checkpoint identity mismatch')
    out=OUT/('diagnostic' if diagnostic else arm);out.mkdir(exist_ok=False);(out/'h8192').mkdir()
    write(out/'prompts.json',prompts)
    write(out/'manifest.json',dict(arm=arm,step=step,checkpoint=cp,protocol=protocol(),horizons=[8192],
        diagnostic=diagnostic,prompts_sha256=historical.semantic(prompts),expected_responses=32*len(prompts),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
    children=[];start=time.time();state=dict(status='RUNNING',started_at=start,pid=os.getpid(),arm=arm)
    write(out/'status.json',state)
    try:
        for shard in range(8):
            with (out/f'shard{shard}.log').open('x') as log:
                children.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--out',str(out),'--shard',str(shard)],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(shard)},stdout=log,stderr=subprocess.STDOUT))
        # Wait for the execution children; this is not a background training monitor.
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('GPU worker failed; stop this arm and preserve all artifacts')
            time.sleep(1)
        require(all(c.returncode==0 for c in children),'GPU worker failed')
        state['generation_ended_at']=time.time()
        require(historical.checkpoint(path)['sha256']==cp['sha256'],'Model changed during evaluation')
        result=score(out);state.update(status='PASS',responses=sum(m['questions']*32 for m in result['metrics']))
    except BaseException as error:
        state.update(status='FAIL',error=repr(error));raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state['ended_at']=time.time();state['wall_seconds']=state['ended_at']-start
        state['allocated_gpu_hours']=8*state['wall_seconds']/3600
        state['cost_scope']='8 reserved head GPUs for generation and CPU scoring; other node idle is reported separately, not counted as active evaluation workers'
        write(out/'status.json',state)
    print(arm,'PASS',flush=True)


def report():
    results={arm:read(OUT/arm/'scores.json') for arm in ARMS}
    require(all(read(OUT/arm/'status.json')['status']=='PASS' for arm in ARMS),'Some evaluation arms incomplete')
    lines=['# PPPO Step20 and historical Step300: H_eval=8192','',
        'Every question has 32 samples. Pass@1 is c/32 and equals Avg@32; Pass@32 is at least one success. All CIs resample questions, not training seeds.',
        'PPPO remains paused. Different training steps/budgets are disclosed; this is not claimed to be exactly equal compute.',
        'Historical three-benchmark results are not silently relabeled or regenerated in this extension.','',
        '| Model | Step | Benchmark | Pass@1 / Avg@32 | SE | Pass@32 | SE | Mean tokens |','|---|---:|---|---:|---:|---:|---:|---:|']
    differences=[]
    for arm,res in results.items():
        for m in res['metrics']:lines.append(f"| {arm} | {res['manifest']['step']} | {m['benchmark']} | {m['avg32']:.6f} | {m['avg32_se']:.6f} | {m['pass32']:.6f} | {m['pass32_se']:.6f} | {m['mean_tokens']:.2f} |")
    pp={p['prompt_id']:p for p in results['pppo_s20']['per_question'] if p['benchmark']=='MATH500'}
    for arm,res in results.items():
        if arm=='pppo_s20':continue
        qs=res['per_question'];require([p['prompt_id'] for p in qs]==list(range(612,1112)),'MATH paired IDs mismatch')
        for metric in ('avg32','pass32'):
            ds=[100*(pp[q['prompt_id']][metric]-q[metric]) for q in qs]
            differences.append(dict(comparison='PPPO20 minus '+arm,benchmark='MATH500',metric=metric,delta_pp=sum(ds)/len(ds),ci95_pp=historical.interval(ds)))
    write(OUT/'comparison.json',dict(math500_paired_differences=differences,input_manifest=read(OUT/'input_manifest.json')))
    write(OUT/'cost_report.json',dict(evaluation={a:read(OUT/a/'status.json') for a in ARMS},
        diagnostic=read(OUT/'diagnostic/status.json'),training_pause=read(RUN/'pause_step20_receipt.json'),
        baseline_cost_source=str(ROOT/'analysis/qwen3_4b_ncbr_baseline_s300_cost_20260920/summary.csv'),
        warning='NCBR 628.65 process-lineage hours and 614.38 step-plus-testing hours have different scopes; do not interchange'))
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


def run_all():
    import fcntl
    with (OUT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import prepare as ctx
        ctx.RUN=RUN
        sys.path.insert(0,str(REPO/'verl/examples/pppo'))
        from continue_ncbr_matched import notify
        try:
            run_arm('pppo_s20',diagnostic=True)
            for arm in ARMS:
                run_arm(arm)
                notify('offline20_'+arm,'8K离线评测完成：'+arm+'；PPPO继续保持暂停')
            report();notify('offline20_complete','PPPO Step20四套题库及四组历史MATH500扩展评测全部完成；训练保持暂停')
        except BaseException as error:
            notify('offline20_failure','Step20离线评测失败，保留现场：'+repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','run-all','worker','score','report'])
    p.add_argument('--out',type=Path);p.add_argument('--shard',type=int,choices=range(8));a=p.parse_args()
    if a.action=='worker':worker(a.out,a.shard)
    elif a.action=='score':score(a.out)
    else:{'prepare':prepare,'preflight':preflight,'run-all':run_all,'report':report}[a.action]()
