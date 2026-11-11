"""Gate receipts and cost summaries derived solely from completed real runs."""
import argparse
import json
import math
import subprocess
import shutil
import time

from prepare import CONFIG, HERE, REPO, RUN, code_fingerprint, sha, write_json
from launch import completed_steps, validate_checkpoints


def gate3():
    roots=[RUN/'acceptance_runs'/name for name in ('smoke-g1','smoke-g8')]
    for root,last in zip(roots,(1,4),strict=True):
        status=json.loads((root/'training_status.json').read_text())
        if status['state']!='STAGE_COMPLETE' or status['last_successful_step']!=last:
            raise ValueError('Smoke incomplete')
        validate_checkpoints(root,[last])
        files=sorted((root/'optimizer_audit').glob('optimizer-rank-*.jsonl'))
        if len(files)!=16:raise ValueError('Missing rank optimizer audit')
        for path in files:
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            if len(rows)!=last*2:raise ValueError('Incomplete smoke optimizer updates')
            for i,row in enumerate(rows,1):
                if (not row['finite_parameters'] or not math.isfinite(row['grad_norm'])
                    or row['sampled_parameter_l2_change']<=0 or row['optimizer_steps_min']!=i
                    or row['optimizer_steps_max']!=i):raise ValueError('Invalid per-rank update evidence')
    attempts=list((roots[1]/'attempts').glob('*/training_status.json'))
    if len(attempts)<2 or len({json.loads(p.read_text())['job_id'] for p in attempts})<2:
        raise ValueError('Fresh-process resume needs two distinct real Ray jobs')
    return dict(runs=list(map(str,roots)),fresh_process_resume_step=4)


def gate4():
    root=RUN/'acceptance_runs/profile'
    status=json.loads((root/'training_status.json').read_text())
    rows=completed_steps(root)
    if status['state']!='STAGE_COMPLETE' or len(rows)!=1:raise ValueError('Full profiling step incomplete')
    m=rows[0]['metrics']
    batches=int(m['train/num_gen_batches'])
    expected={'pppo/candidate_original_rows':6144*batches,'pppo/continuation_rows':49152*batches,
              'pppo/actor_rows':2048,'train/retained_prompt_groups':256,
              'actor/pppo_optimizer_updates_total':16,'actor/pppo_optimizer_updates_this_outer':16}
    for key,value in expected.items():
        if m[key]!=value:raise ValueError(f'Profiling count mismatch {key}')
    validate_checkpoints(root,[1])
    return dict(metrics=m,timing_seconds=rows[0]['timing_seconds'],
                allocated_gpu_hours=16*(status['ended_at']-status['started_at'])/3600,
                scope='16 allocated GPUs for whole profiling job, including setup and validation; stage timings may overlap')


def reports():
    lines=['# PPPO + DAPO Dynamic Sampling (NCBR-Matched)','',
           'PASS means the stated gate ran successfully. NOT RUN is not a pass.','',
           '| Gate | Status | Evidence |','|---|---|---|']
    for gate in range(5):
        path=RUN/'acceptance'/f'gate{gate}'/'result.json'
        status=json.loads(path.read_text())['status'] if path.exists() else 'NOT RUN'
        lines.append(f'| {gate} | {status} | {path} |')
    (RUN/'validation_report.md').write_text('\n'.join(lines)+'\n')
    if not (RUN/'training_status.json').exists():
        write_json(RUN/'training_status.json',dict(state='NOT RUN',job_id=None,last_successful_step=0,
                   reason='Gate 0–4 must all pass; formal training has not started'))
    if not (RUN/'checkpoint_manifest.json').exists():
        write_json(RUN/'checkpoint_manifest.json',dict(status='NOT RUN',checkpoints=[]))
    for name in ['evaluation_report.md','comparison_with_ncbr.md']:
        if not (RUN/name).exists():
            (RUN/name).write_text('# '+name.removesuffix('.md')+'\n\nNOT RUN: formal Step 300 is required.\n')
    costs={}
    for root in [RUN,*sorted((RUN/'acceptance_runs').glob('*'))]:
        rows=completed_steps(root)
        if not rows:continue
        stages={}
        totals={}
        for row in rows:
            for key,value in row['timing_seconds'].items():stages[key]=stages.get(key,0.)+value
            for key in ['pppo/original_tokens','pppo/tail_tokens','pppo/continuation_prefill_tokens',
                        'pppo/optimized_tokens','pppo/actor_input_tokens','train/filtered_prompt_groups']:
                totals[key]=totals.get(key,0)+row['metrics'].get(key,0)
        status=json.loads((root/'training_status.json').read_text())
        elapsed=status.get('ended_at',time.time())-status['started_at']
        costs[root.name]=dict(stage_wall_seconds=stages,tokens_and_groups=totals,
                             allocated_gpu_hours=16*elapsed/3600,job_wall_seconds=elapsed,
                             online_holdout_wall_seconds=stages.get('testing',0),
                             successful_outer_steps=len(rows),timings_may_overlap=True)
    diagnostics={}
    for path in sorted((RUN/'acceptance').glob('gate2*/result.json')):
        result=json.loads(path.read_text());diagnostics[str(path.parent)]=dict(status=result['status'],
                   wall_seconds=result.get('wall_seconds'),allocated_gpu_hours=result.get('allocated_gpu_hours'))
    write_json(RUN/'cost_report.json',dict(status='MEASURED' if costs else 'NOT RUN',runs=costs,
               diagnostic_runs=diagnostics,
               gpu_hours_rule='allocated GPU count times measured job wall time / 3600; online holdout and offline evaluation reported separately',
               warning='optimized-token fraction is not a GPU-hours saving'))
    reports_dir=HERE/'reports';reports_dir.mkdir(exist_ok=True)
    for name in ('validation_report.md','training_status.json','cost_report.json','checkpoint_manifest.json',
                 'evaluation_report.md','comparison_with_ncbr.md'):
        shutil.copy2(RUN/name,reports_dir/name)
    if (RUN/'training_metrics.jsonl').exists():
        shutil.copy2(RUN/'training_metrics.jsonl',reports_dir/'training_metrics.jsonl')
    else:
        (reports_dir/'training_metrics.jsonl').write_text('')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['gate3','gate4','reports']);args=parser.parse_args()
    if args.action=='reports':reports()
    else:
        gate=int(args.action[-1]);directory=RUN/'acceptance'/args.action;directory.mkdir(parents=True,exist_ok=True)
        report=dict(gate=gate,status='FAIL',timestamp=time.time(),code_fingerprint=code_fingerprint(),
                    config_sha256=sha(CONFIG),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        try:
            report['evidence']=gate3() if gate==3 else gate4();report['status']='PASS'
        except BaseException as error:report['error']=repr(error);raise
        finally:write_json(directory/'result.json',report);reports()
