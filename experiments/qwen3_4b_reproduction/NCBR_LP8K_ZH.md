# NCBR H2K/L8K：完整回答使用 DAPO 8K 长度惩罚

新实验名为 `ncbr_lp8k`，从 Qwen3-4B-Base 独立训练300步，目标为单节点8×A100 80GB。它对齐原NCBR设置，仅改变长度惩罚的参数及作用长度；不把actor训练长度从2K扩大到8K。

## 奖励含义

历史参考为 `qwen3_4b_base_dapo_b256_g768_m16_n8_r8192_s600_v1`：`max_resp_len=8192`、`overlong_buffer_cfg.len=1638`、`penalty_factor=1.0`，enable=true、log=false。原NCBR的短长度惩罚是2048/410，新实验使用8192/1638，并明确对**完整prefix+tail回答**计算。

```text
P(t) = min(-(t - 6554) / 1638, 0)
触顶样本新增奖励 = 长任务分数 - 短任务分数 + P(完整长度) - P(短长度)
```

`t`是不含prompt和padding的实际回答token数，遵循DAPO有效回答mask计数，包括其中的EOS。6554及以下不扣分，6555扣1/1638，7373扣0.5，8192扣1。原始短回答最多2048，因此新配置下P(短长度)=0；减去P(短长度)仍显式保留在实现中，避免重复惩罚。

达到H上限的样本自然续写到EOS或总长度L=8192，完整奖励回传到**原2K前缀最后一个有效token**。未达到H的样本不续写，保留短回答奖励。任务分数、正确性acc和正确性动态过滤含义不变，其他奖励项保持不变；tail不参与actor反向传播。长奖励张量本身不整体替换短奖励，避免重复引入其他奖励项。

只改8192/1638数值而不启用新模式，会导致短回答永远不触发惩罚，不能实现本实验。因此配置额外设置：

```yaml
actor_rollout_ref.rollout.boundary_return.length_penalty_mode: full_response
```

其默认值为 `prefix`，原 `ncbr`、`dapo`、`dapo4k` 行为不变；原 `all` 仍只运行DAPO 2K和原NCBR。新模式要求registered DAPO reward manager、合法惩罚参数及max_resp_len=L；不适用于naive/custom reward manager。shadow模式计算诊断但不改变训练奖励。

## 配置与环境

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-4B-Base，沿用原固定revision及资产哈希 |
| H / L / 上下文 | 2048 / 8192 / 9216 |
| train batch / gen batch / mini-batch / n | 256 / 768 / 16 / 8 |
| 训练拓扑 | 单节点8×A100 80GB，rollout TP2、训练SP1 |
| 学习率及优化器 | 沿用原NCBR，AdamW 1e-6、constant |
| scheduler / continuation并发 | fixed_wave / 128，batch512 |
| 步数 / 验证 | 300步；初始及每10步，greedy、n=1、短回答上限2048 |
| 保存 | 50/100/150/200/250/300步，完整训练状态及HF权重 |
| W&B | offline，不需要登录 |

指定容器为 `docker.1ms.run/verlai/verl:vllm018.dev1`。先按 [环境说明](ENVIRONMENT.md) 安装/核验固定依赖，再按 [路径与下载说明](PATHS_AND_DOWNLOAD_ZH.md) 在新机器手动下载模型和数据并配置路径。Base链接：https://huggingface.co/Qwen/Qwen3-4B-Base 。本包不包含权重、数据或历史结果，不自动下载训练资产。

完整源码包内的 `verl/` 包含新奖励选项，不能用旧压缩包中的运行时代码替代。源码清单记录原始baseline及本次逐文件修改哈希；原始清单保存在 `provenance/source_baseline.json`。配置变化及历史8K参数来源在 `provenance/ncbr_lp8k_alignment.json`。

## 手动路径配置及启动

在容器内进入解压后的源码包目录，编辑 `settings.env`（仓库内对应 `experiments/qwen3_4b_reproduction`）。下面路径只是例子，必须改成新机器实际可见路径；容器中填写挂载后的容器路径：

```bash
MODEL_PATH="/data/models/Qwen3-4B-Base"
TRAIN_FILE="/data/dapo_math_17k_train.parquet"
VAL_FILE_1="/data/aime-2024-verl.parquet"
VAL_FILE_2="/data/aime-2025-verl.parquet"
OUTPUT_ROOT="/data/runs"
CACHE_ROOT="/tmp/q4repro-cache"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=8
RUN_TAG="ncbr_lp8k_v1"
# 使用已通过环境检查的Python时填写，否则默认包内.venv/bin/python：
# PYTHON_BIN="/path/to/python"
# 源码默认自动定位；手动配置需指向含setup.py的外层verl：
# VERL_ROOT="/workspace/qwen3_4b_reproduction/verl"
```

其余settings保留原值，包括RAY_CPUS=32、RAY_OBJECT_STORE_GB=16。只使用模型原始Base初始化；不要设置历史checkpoint或改resume配置。正式检查要求输出盘约460GB可用，另留模型、数据、缓存空间；六份完整checkpoint不会自动删除。建议主机RAM512GiB、至少128GiB可用，共享内存64GiB。

CPU配置检查（不进行GPU测试、不开Ray、不训练）：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="/path/to/package/verl" \
  /path/to/python /path/to/package/run.py config ncbr_lp8k
```

该直接Python命令需要先把settings变量导出，最方便的方式是先 `set -a; source settings.env; set +a`，再执行上述命令。也可使用 `bash run_ncbr_lp8k.sh config`，仅配置校验，不执行GPU资源预检。

**以下命令供用户之后在目标机器准备运行时执行。本次交付不会执行它们：**

```bash
bash run_ncbr_lp8k.sh environment
bash run_ncbr_lp8k.sh check  # 包含GPU资源检查
bash run_ncbr_lp8k.sh smoke  # 可选，真实GPU两步冒烟
bash run_ncbr_lp8k.sh        # 正式从Base训练300步
```

等价正式命令：`bash run.sh train ncbr_lp8k`。使用配置副本可在命令前设置 `SETTINGS_FILE=/path/to/settings.env`。

正式目录是 `OUTPUT_ROOT/ncbr_lp8k_s300_<RUN_TAG>/`，冒烟是 `ncbr_lp8k_smoke_s2_<RUN_TAG>/`。已有目录拒绝覆盖；重跑换RUN_TAG，正式训练不从冒烟继续。脚本前台运行，建议使用持久终端。不得与其他训练抢占GPU。

## 输出与检查

常规日志、resolved_config、历史差异、完整checkpoint结构沿用原入口。新增长度惩罚统计为：

- `boundary_return/short_length_penalty_mean`
- `boundary_return/full_length_penalty_mean`
- `boundary_return/length_penalty_delta_mean`
- `boundary_return/full_length_penalty_fraction`

以上按每个trainer step的全部候选行加权汇总；不等同于过滤后最终actor batch的均值。任务分数差仍使用原 `task_score_delta` 统计，不将长度扣分混入任务正确性。逐行诊断标签分别保存短、完整长度惩罚及差值。

触顶样本缺失续写、prefix与actor不一致、完整长度超过L、无效惩罚参数或非有限奖励均失败，不静默跳过。奖励残差校验允许且仅允许显式长度惩罚差，其余shaping保持不变。

每个global_step目录保留data.pt、全部8个rank模型/optimizer/extra_state及HF导出。若未来续训，需完整目录和相同拓扑；当前启动器只提供独立训练。评测仍使用统一离线评测工具，与原实验采用相同预算及数据，不对评测正确率额外施加训练长度惩罚。

本次按用户要求仅执行CPU测试和文件检查；没有执行GPU预检、GPU冒烟、正式训练或目标容器GPU验证。CPU通过不代表显存或训练吞吐已验证。
