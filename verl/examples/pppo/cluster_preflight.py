"""Read-only, zero-GPU Ray inspection. This is not an NCCL or training gate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def inspect_node(repo):
    def command(args):
        return subprocess.check_output(args, text=True, timeout=20).strip()
    paths = [str(Path(repo) / 'verl/verl/experimental/pppo/trainer.py'),
             '/workspace/models/Qwen3-4B-Base/config.json',
             '/workspace/models/Qwen3-4B-Base/tokenizer_config.json',
             '/workspace/rl/data/dapo_math_17k_train.parquet']
    hashes = {}
    for name in paths:
        digest = hashlib.sha256()
        with open(name, 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    return dict(host=socket.gethostname(),
                versions={p: importlib.metadata.version(p) for p in
                          ('torch', 'vllm', 'ray', 'transformers', 'flash-attn')},
                gpus=command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits']),
                compute_processes=command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                           '--format=csv,noheader,nounits']),
                hashes=hashes, disk=command(['df', '-B1', repo, '/tmp', '/dev/shm']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    ray.init(address='10.8.191.127:6395', log_to_driver=False)
    try:
        nodes = [n for n in ray.nodes() if n['Alive']]
        remote = ray.remote(num_cpus=0)(inspect_node)
        probes = ray.get([remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n['NodeID'], soft=False))
                         .remote(str(repo)) for n in nodes], timeout=120)
        errors = []
        if {n['NodeManagerAddress'] for n in nodes} != {'10.8.191.127', '10.8.191.131'}:
            errors.append('Expected exactly the two historical NCBR nodes')
        for node, probe in zip(nodes, probes, strict=True):
            if node['Resources'].get('GPU') != 8:
                errors.append('Expected eight Ray GPUs per node')
            rows = probe['gpus'].splitlines()
            if len(rows) != 8:
                errors.append('Expected eight physical GPUs per node')
            for row in rows:
                _, _, name, used, total, utilization = map(str.strip, row.split(','))
                if name != 'NVIDIA A100-SXM4-40GB' or int(total) != 40960 or int(used) > 100 or int(utilization) != 0:
                    errors.append(f"{probe['host']}: GPU unavailable: {row}")
            if probe['compute_processes']:
                errors.append(f"{probe['host']}: active GPU processes")
        if ray.available_resources().get('GPU') != 16:
            errors.append('All sixteen Ray GPUs must be available')
        if probes and any(p['hashes'] != probes[0]['hashes'] or p['versions'] != probes[0]['versions'] for p in probes):
            errors.append('Node hashes or versions differ')
        report = dict(timestamp=time.time(), status='FAIL' if errors else 'PASS',
                      scope='cluster membership, physical idle GPUs, selected shared files and package versions only',
                      allocation='existing platform allocation confirmed by user',
                      head='10.8.191.127:6395', worker_management='user', errors=errors,
                      nodes=[dict(ip=n['NodeManagerAddress'], resources=n['Resources']) for n in nodes], probes=probes,
                      not_tested=['NCCL collectives', 'FSDP', 'model weight hashes', 'training capacity'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        os.replace(temporary, args.output)
        print(json.dumps(report, indent=2))
        if errors:
            raise SystemExit(1)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
