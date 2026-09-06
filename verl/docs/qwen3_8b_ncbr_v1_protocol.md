# Qwen3-8B-Base DAPO / NCBR-v1 正式对照实验协议

本协议从已完成的 Qwen3-1.7B 分支 HEAD
`a66e4d56655c5c6dd2af62f5ffefaf0d75a2c309` 派生。需求文本记录的
`a66e4d56655c6dd2af62f5ffefaf0d75a2c309` 不是本仓库中的 Git 对象；两者不得混同。
8B 分支为 `exp/qwen3-8b-ncbr-v1`，算法实现（continuation、reward replacement、
`boundary_acc`、prefix-only actor tensors）不作修改。

模型固定为 `/workspace/models/Qwen3-8B-Base`、`Qwen/Qwen3-8B-Base` revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`。每个 GPU stage 使用 A/B 两节点各
8 张 A100-40GB；node-A 与 node-B manifest 必须分别重算所有模型文件、revision
metadata 和 train/AIME2024/AIME2025 SHA256，再经 `compare_node_manifests.py`
逐项比较。AIME2024 与 AIME2025 永远分别记录。

## 冻结训练语义

共同配置为 B256/G768/M16/N8、prompt 1024、H2048、`max_model_len=9216`；AdamW
LR 1e-6 constant、warmup 0、weight decay 0.01、grad clip 1.0、PPO epoch 1；
GRPO std normalization、dynamic sampling、`metric=acc`、最多 10 generation
batches；clip low/high 0.20/0.28、dual clip 3.0、token-mean；KL、entropy、所有
其他 intervention 均关闭。采样为 temperature 1、top-p 1、top-k -1；全部 seed
为 42。actor param offload=false、optimizer offload=true、ref param offload=true，
profiling 不得更改。

NCBR 两臂都显式携带 L8192、correctness key `acc`、threshold 0.5、task score key
`score`、concurrency 128、request batch 512、timeout 600 秒、long-reward chunk 256、
seed 42、strict=true。唯一算法差异为 `boundary_return.mode: off -> replace`。

正式 W&B/输出名称为：

- `qwen3_8b_base_dapo_ctx9216_b256_g768_m16_n8_h2048_s300_seed42_v1`
- `qwen3_8b_base_ncbr_v1_b256_g768_m16_n8_h2048_l8192_s300_seed42_v1`

两者均执行 Step 0 validation、每 10 step validation、每 50 step 完整 checkpoint，
共 300 optimizer steps。

## Profiling 与 smoke

候选固定如下，且 offload 仍使用共同配置：

| 候选 | TP | actor micro | rollout/ref logprob micro | vLLM util | max seqs | batched tokens |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 4 | 1 | 1 | 0.40 | 128 | 16384 |
| P1 | 4 | 1 | 2 | 0.50 | 256 | 32768 |
| P_SAFE | 8 | 1 | 1 | 0.35 | 64 | 8192 |

每个候选从 Base 独立运行 5 个有效 optimizer steps；Step 1 warmup，比较 Step 2–5。
先运行 P0。P0 仅在 peak NVML memory ≤36 GiB 且无 OOM、worker loss、preemption
时允许 P1；P1 要求 peak ≤38.5 GiB、无 OOM/preemption/deadlock，且 Step 2–5
median total time ≤P0 的 110%，否则冻结 P0。仅 P0 失败时运行 P_SAFE；P_SAFE
失败即停止，不改训练超参。`select_qwen3_8b_candidate.py` 不读取 reward/accuracy。

每份输入 profile 必须包含 16 卡 1 秒采样、allocated/reserved/NVML peaks、逐 step
rollout/actor/total time、正常生成 token/s、candidate batches、16 卡利用率分布、
OOM/worker loss/preemption/deadlock、vLLM scheduling 和 Ray worker status。

冻结候选随后从 Base 独立启动 3-step NCBR smoke。`validate_qwen3_8b_smoke.py`
要求 continuation request 非零、三步完成、无 OOM/deadlock/timeout、long verifier
行数匹配、`prefix_penalty_drift_max=0`、无 NaN/Inf、boundary correction 生效、
actor tail token count 为零且 teardown PASS。自然 continuation 为零时输出
`mechanism_coverage_insufficient` 并阻断正式就绪，不调整参数。

## 启动门禁与硬暂停

`run_qwen3_8b_profile_fsdp.sh` 是分阶段 launcher；两个正式入口分别为
`run_qwen3_8b_baseline_s300.sh` 与 `run_qwen3_8b_ncbr_v1_s300.sh`。正式入口默认
退出 3，只有新的明确授权把 `NCBR_AUTHORIZE_S300` 设置为
`AUTHORIZE_QWEN3_8B_S300` 后才继续。NCBR 入口还必须验证 Baseline 已完成 Step 300、
完整 checkpoint 和 teardown 的 PASS receipt，因此顺序不可颠倒。

正式启动前必须在 `ulimit -n 524288` 下重新证明两节点身份/版本一致、16 GPU 全空闲、
NVML/进程/Ray 一致、node affinity、共享哈希与 W&B 可用。不得停止来源不明的 Ray；
只有确认两节点均无 daemon 且端口空闲才可新建专用集群，否则只能在完整空闲证明后复用。
任何缺项均 fail closed。

两个正式命令先以 Hydra `--cfg job` 生成完整 resolved OmegaConf。运行
`resolved_config_diff.py` 后，白名单仅包括 experiment/W&B identity、输出/checkpoint/
cache/receipt 路径及 `boundary_return.mode`；任何其他差异写 FAIL receipt。常规 formal
launcher 会在训练模块 exec 前重跑该 gate。配置解析模式必须显式设置
`NCBR_CONFIG_RESOLUTION_ONLY=1` 且参数严格包含 `--cfg job`，不会训练。

本轮结束条件是实现、CPU/命令测试、实际可用时的 profiling 与 smoke、配置冻结和报告。
无论 readiness 为 PASS 或 FAIL，报告后均硬暂停，不自动启动 S300。
