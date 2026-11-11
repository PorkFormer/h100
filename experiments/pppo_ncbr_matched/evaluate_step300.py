"""Only the formal step-300 model; frozen historical 8K, 32-sample protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare import MODEL, ROOT, RUN, HERE, sha, write_json
import historical_eval as historical


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    output=RUN/'offline_eval_step300_h8192'
    os.environ['MODEL_PATH']=str(MODEL)
    if args.worker is not None:
        historical.worker(argparse.Namespace(output=output,horizon=8192,shard=args.worker))
        return
    status=json.loads((RUN/'training_status.json').read_text())
    if status['state']!='TRAINING_COMPLETE' or status['last_successful_step']!=300:
        raise SystemExit('Formal training must finish before downstream evaluation')
    from launch import validate_checkpoints
    validate_checkpoints(RUN,[50,100,150,200,250,300])
    from verl.utils.reward_score import math_dapo
    protocol=json.loads((HERE/'provenance/offline_eval.json').read_text())
    if sha(math_dapo.__file__)!=protocol['verifier_sha256']:raise ValueError('Historical verifier changed')
    source=ROOT/'analysis/ncbr_offline_eval/templates/prompts.parquet'
    if sha(source)!=protocol['historical_prompts_sha256']:raise ValueError('Historical prompts changed')
    import pyarrow.parquet as pq
    prompts=pq.read_table(source).to_pylist()
    prompts=[{k:p[k] if k!='prompt_token_ids' else list(p[k]) for k in historical.FIELDS} for p in prompts]
    historical.validate_prompts(prompts)
    cp=historical.checkpoint(RUN/'checkpoints/global_step_300/actor/huggingface')
    query=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[list(map(int,row.split(','))) for row in query.splitlines()]
    if len(rows)!=8 or any(used>100 or util!=0 for used,util in rows):raise RuntimeError('Head GPUs are not idle')
    output.mkdir(exist_ok=False)
    (output/'h8192').mkdir()
    write_json(output/'prompts.json',prompts)
    write_json(output/'manifest.json',dict(arm='pppo',step=300,checkpoint=cp,horizons=[8192],protocol=protocol))
    started=time.time()
    children=[]
    try:
        for shard in range(8):
            with (output/f'shard{shard}.log').open('x') as log:
                children.append(subprocess.Popen([sys.executable,__file__,'--worker',str(shard)],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(shard)},stdout=log,stderr=subprocess.STDOUT))
        for child in children:
            if child.wait()!=0:raise RuntimeError('Evaluation worker failed; preserved output requires a new attempt directory')
        if historical.checkpoint(cp['path'])['sha256']!=cp['sha256']:raise ValueError('Checkpoint changed during evaluation')
        historical.score(output)
        import numpy as np
        results=json.loads((output/'scores.json').read_text())
        for row in results['metrics']:
            questions=[q for q in results['per_question'] if q['benchmark']==row['benchmark']]
            row['pass1']=row['avg32']
            row['pass1_standard_error']=float(np.std([q['avg32'] for q in questions],ddof=1)/np.sqrt(len(questions)))
            row['avg32_standard_error']=row['pass1_standard_error']
            row['pass32_standard_error']=float(np.std([q['pass32'] for q in questions],ddof=1)/np.sqrt(len(questions)))
        results['wall_seconds']=time.time()-started
        results['allocated_gpu_hours']=8*results['wall_seconds']/3600
        write_json(output/'scores.json',results)
        lines=['# Formal Step 300 independent evaluation','',
               'Pass@1 is the unbiased c/32 estimate, averaged over questions; Avg@32 is the same per-question mean.',
               'Pass@32 is the fraction of questions with at least one correct response. Standard errors use questions as independent units.',
               'Training seed uncertainty is not estimated from these within-checkpoint samples.','',
               '| Benchmark | Pass@1 / Avg@32 | SE | Pass@32 | SE | Mean tokens |',
               '|---|---:|---:|---:|---:|---:|']
        for r in results['metrics']:
            lines.append(f"| {r['benchmark']} | {r['pass1']:.6f} | {r['pass1_standard_error']:.6f} | {r['pass32']:.6f} | {r['pass32_standard_error']:.6f} | {r['mean_tokens']:.2f} |")
        (RUN/'evaluation_report.md').write_text('\n'.join(lines)+'\n')
        status['state']='COMPLETE'
        status['evaluation_completed_at']=time.time()
        write_json(RUN/'training_status.json',status)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait()


if __name__=='__main__':main()
