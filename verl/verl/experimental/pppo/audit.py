"""Opt-in per-rank optimizer evidence; never used by other algorithms."""
import json
from pathlib import Path
import torch


def snapshot(engine):
    values=[]
    for group in engine.optimizer.param_groups:
        for parameter in group['params']:
            local=parameter.detach()
            if hasattr(local,'to_local'): local=local.to_local()
            if local.numel():
                # Several positions per tensor, not just the embedding's first unused row.
                flat=local.reshape(-1)
                count=min(16,flat.numel())
                # Float32 rounds large last indices up, potentially to numel.
                indices=torch.arange(count,device=flat.device,dtype=torch.int64)
                indices=indices*(flat.numel()-1)//max(count-1,1)
                values.append(flat[indices].float().cpu())
    return torch.cat(values) if values else torch.empty(0)


def record(engine, directory, step, mini, before, output):
    from verl.utils import tensordict_utils as tu
    metrics = tu.get(output, "metrics")
    after=snapshot(engine)
    steps=[float(state['step']) for state in engine.optimizer.state.values() if 'step' in state]
    rank=torch.distributed.get_rank()
    path=Path(directory)
    path.mkdir(parents=True,exist_ok=True)
    record={'grad_norm': float(metrics['grad_norm']), 'loss': float(metrics['loss']), 'rank':rank,'outer_step':step,'mini':mini,'optimizer_steps_min':min(steps) if steps else None,
            'optimizer_steps_max':max(steps) if steps else None,'finite_parameters':bool(torch.isfinite(after).all()),
            'sampled_parameter_l2_change':float(torch.linalg.vector_norm(after-before)),
            'sampled_parameter_count':after.numel(),'engine':type(engine).__name__,'module':__file__}
    with (path/f'optimizer-rank-{rank}.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
