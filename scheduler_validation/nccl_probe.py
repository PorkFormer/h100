import datetime, json, os
import torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=60))
x=torch.tensor([rank+1.],device=f'cuda:{rank}')
dist.all_reduce(x)
torch.cuda.synchronize()
assert x.item()==36
print(json.dumps(dict(rank=rank,uuid=str(torch.cuda.get_device_properties(rank).uuid),result=x.item())),flush=True)
dist.destroy_process_group()
