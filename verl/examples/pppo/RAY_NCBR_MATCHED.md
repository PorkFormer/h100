# Existing NCBR two-node allocation

Head: p-kt-dgx-a100-03 / 10.8.191.127:6395, 240 CPUs and 8 A100-SXM4-40GB.
Worker: p-kt-dgx-a100-07 / 10.8.191.131, 256 CPUs and 8 A100-SXM4-40GB.
The user owns worker startup. No script here stops Ray or other workloads.

`bash ray_head_ncbr_matched.sh` retains a working head. For a new head it
uses the historical 200000000000-byte object store and stays foregrounded
with `--block`; use a persistent session. A live but unresponsive head is
an error, never an instruction to run `ray stop`.

For the user to run on the allocated worker **only if it has not already joined**:

```bash
export RAY_USAGE_STATS_ENABLED=0
export RAY_TMPDIR=/tmp/rw16g
ray start --address=10.8.191.127:6395 --node-ip-address=10.8.191.131 \
  --num-cpus=256 --num-gpus=8 --resources='{"pppo_worker":1}' \
  --object-store-memory=200000000000 \
  --object-manager-port=7011 --node-manager-port=7012 \
  --min-worker-port=22000 --max-worker-port=22999 \
  --dashboard-agent-listen-port=52365 --dashboard-agent-grpc-port=52366 \
  --runtime-env-agent-port=52367 --metrics-export-port=52368 \
  --disable-usage-stats --block
```

Use Ray 2.54.1 and the existing matching container/runtime on both nodes.
The 22000–22999 range follows the later validated deployment; the old
100-port range exhausted Ray CPU worker ports. `/tmp/rh16g` and `/tmp/rw16g`
avoid AF_UNIX path length failures. Network access stays on the private network.

Run `cluster_preflight.py --output <new-run>/cluster_preflight.json` from the
head. This uses zero-GPU inspection tasks and never reserves model GPUs.
Membership PASS is not a GPU algorithm acceptance result.

On 2026-11-11 the existing cluster already contained both nodes with 16 idle
GPUs, custom resource labels dapo_head/dapo_worker, and a 1 GiB head object
store versus the historical 200 GB. These labels do not affect the shared
actor resource pool. The existing cluster was retained without restarting;
object-store capacity remains subject to the full-scale profiling gate.
