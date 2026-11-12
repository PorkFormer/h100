"""One-shot checkpoint budget report; never schedules, monitors, deletes or stops training."""
import argparse
import json
from pathlib import Path

TARGET = 628.649734220604


def report(root):
    status=json.loads((root/'training_status.json').read_text())
    rows=[json.loads(line) for line in (root/'training_metrics.jsonl').read_text().splitlines()]
    records=[];step_sum=0.
    for row in rows:
        step=row['step'];step_sum+=row['timing_seconds']['step']
        path=root/'checkpoints'/f'global_step_{step}'/'pppo_state.json'
        if not path.exists():raise ValueError(f'Missing expected every-step checkpoint {step}')
        state=json.loads(path.read_text())
        if state['status']!='committed' or state['step']!=step:raise ValueError('Uncommitted checkpoint')
        for name,size in state['files'].items():
            if (path.parent/name).stat().st_size!=size:raise ValueError('Checkpoint size mismatch')
        elapsed=path.stat().st_mtime-status['started_at']
        if elapsed<=0:raise ValueError('Checkpoint timestamps predate the fresh run')
        records.append(dict(step=step,path=str(path.parent),allocated_process_gpu_hours=16*elapsed/3600,
                            step_timer_gpu_hours=16*step_sum/3600))
    below=[x for x in records if x['allocated_process_gpu_hours']<=TARGET]
    above=[x for x in records if x['allocated_process_gpu_hours']>TARGET]
    return dict(target_gpu_hours=TARGET,source='/workspace/rl/h100/analysis/ncbr_boundary_return_s300_seed1_20260828/gpu_cost_summary.json',
        basis='16 GPUs times fresh driver-start to committed-sidecar mtime; includes startup, online validation and checkpoint saves',
        limitations='Filesystem timestamps give checkpoint boundary estimates. NCBR full-lineage trainer elapsed includes process end overhead; audit exact timer endpoints before claiming exact equal cost. Step-only timer is a separate lower-scope statistic, not interchangeable.',
        discarded_old_run_cost='Excluded from this model lineage, must be reported separately as experiment expenditure',
        last_at_or_below_budget=below[-1] if below else None,first_above_budget=above[0] if above else None,
        budget_bracket_complete=bool(below and above),checkpoints=records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    print(json.dumps(report(args.run),indent=2))
