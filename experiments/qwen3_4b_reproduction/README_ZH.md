# Qwen3-4B：DAPO 2K 与 NCBR H=2K、L=8K 独立复现

本包在 **单机 8×80GB NVIDIA GPU** 上，从同一份 **Qwen3-4B-Base** 分别训练 DAPO 和 NCBR，各 **300 个 trainer step**。两组独立初始化、独立输出；`train all` 顺序执行，DAPO 失败时不会继续 NCBR。默认本地 JSONL＋离线 W&B，无需登录。

源码来自完整工作区 `b1835ea0fb388eb6e4d29a58ab18b408cae7b34d`，不是上游默认 verl，也不是精简 submission 包。包内 `verl/` 包含训练所需完整运行时代码及相关测试。包不含数据、模型、历史 checkpoint、凭据或其他实验输出。

**新机器首先阅读 [手动下载与路径配置](PATHS_AND_DOWNLOAD_ZH.md)**：自行选择新机器存储目录，下载源码包、Base 模型和三份原始预处理数据，设置 Docker 挂载，再手工填写 `settings.env`。文中的绝对路径均为示例，不要求与旧机器一致；启动器不会自动下载资产或沿用旧机器目录。

## 1. 准备机器与解压

推荐 Linux x86_64、8 张 A100/H100 80GB 或同等兼容 NVIDIA GPU、32 个以上逻辑 CPU、512GiB 系统内存、至少 1TB 可用磁盘。默认检查至少 128GiB **空闲**系统内存，这是启动下限，不是峰值内存保证。GPU 必须支持 BF16；脚本拒绝不足 80GB 级别显存的卡。

指定容器镜像为 **`docker.1ms.run/verlai/verl:vllm018.dev1`**。宿主机准备兼容 CUDA 12.9 的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit；Python、CUDA toolkit（`nvcc`）及 C++ 编译器在容器内检查。默认 Ray object store 为 16GiB，启动检查要求 `/dev/shm` 至少有 20GiB 空闲，启动容器时设置 `--shm-size=64g`。完整的镜像拉取、目录挂载和启动命令见 [ENVIRONMENT.md](ENVIRONMENT.md)。

```bash
sha256sum -c qwen3_4b_reproduction.tar.gz.sha256
tar -xzf qwen3_4b_reproduction.tar.gz
cd qwen3_4b_reproduction
sha256sum -c SHA256SUMS
```

首次验证压缩包完整性后再编辑 `settings.env`；编辑后整个包的 SHA256SUMS 会合理地提示该文件变化，启动器只锁定实验配置与运行时源码，不锁定用户机器设置。

## 2. 在新机器手动下载 Base 和数据

按 [下载与路径指南](PATHS_AND_DOWNLOAD_ZH.md) 在新机器执行下载：源码来自远程仓库的交付包；Base 来自 `Qwen/Qwen3-4B-Base`，固定 revision；三份历史预处理 Parquet 的下载地址由你手动提供。当前没有已核实的这三份文件的公开下载 URL，不能用公开原始题库替代。该流程不要求访问或复用旧机器的绝对路径。

必须保留原始文件内容，包括题目、答案、chat template、样本重复和顺序。**不要使用 Instruct 模型、旧训练导出的 HF 权重、去重后的 17k 数据或重新转换的数据。**

| 文件 | 原始行数 | 说明 |
|---|---:|---|
| dapo_math_17k_train.parquet | 1,791,700 | 文件名是 17k，但已保存重复样本 |
| aime-2024-verl.parquet | 960 | 原历史验证集有重复样本 |
| aime-2025-verl.parquet | 30 | 保留原文件 |

`provenance/assets.json` 保存三份数据及全部 Base 文件的 SHA256、文件大小和数据行数。数据及模型配置/tokenizer 配置哈希与历史记录一致；完整模型权重哈希是在本次打包时对现存 Base 权重计算的，并非历史时刻留下的权重哈希。`check` 会逐个核验。

## 3. 安装环境并设置路径

先按 [容器环境说明](ENVIRONMENT.md) 启动指定镜像，并在容器内检查自带环境：

```bash
PYTHON_BIN="$(command -v python3)" bash run.sh environment
```

检查通过时，将该 Python 的绝对路径写入 `settings.env` 的 `PYTHON_BIN`，跳过下面的重新安装。镜像标签不能替代版本检查；本次尚未实际拉取或验证该镜像。

若需要建立独立环境，在容器内安装 uv（已有 uv 可跳过），然后运行环境脚本，并保持 `settings.env` 中的 `PYTHON_BIN` 注释：

```bash
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash setup_env.sh
```

脚本创建包内 `.venv`，不会覆盖已有虚拟环境。安装依赖需要网络和足够的构建空间。FlashAttention 首次编译可能较久，可用 `MAX_JOBS=4 bash setup_env.sh` 限制编译并发。

核心版本为 Python 3.12、torch 2.10.0+cu129、vLLM 0.18.0、Ray 2.54.1、Transformers 5.3.0、FlashAttention 2.8.3、FlashInfer 0.6.6。**不要执行 `pip install -e 'verl[vllm]'` 或直接安装未锁定的 requirements**。历史环境存在两个已知依赖声明冲突，安装方法及验证边界见 [ENVIRONMENT.md](ENVIRONMENT.md)。

在新机器手动编辑 `settings.env`。容器运行时填写**容器内路径**，直接运行时填写新机器本地路径。以下是独立源码包挂载到 `/workspace/qwen3_4b_reproduction`、资产挂载到 `/data` 的示例；其他位置按实际情况改写：

```bash
VERL_ROOT="/workspace/qwen3_4b_reproduction/verl"
MODEL_PATH="/data/models/Qwen3-4B-Base"
TRAIN_FILE="/data/dapo_math_17k_train.parquet"
VAL_FILE_1="/data/aime-2024-verl.parquet"
VAL_FILE_2="/data/aime-2025-verl.parquet"
OUTPUT_ROOT="/data/runs"
CACHE_ROOT="/tmp/q4repro-cache"
# .venv 用户保持 PYTHON_BIN 注释；镜像环境用户填写 command -v python3 的实际输出。
# PYTHON_BIN="/workspace/qwen3_4b_reproduction/.venv/bin/python"
```

源码根目录应含 `setup.py` 和内层 `verl/`，不能误填内层 Python 包路径。源码包与 Git checkout 的目录层级不同；宿主机/容器路径对照表、Python 选择、输出位置和文件检查命令统一见 [手动下载与路径配置](PATHS_AND_DOWNLOAD_ZH.md)。

路径允许空格，shell 中需保持双引号。其余默认值已适配 8 卡；不要为提速改 batch、n、H、L、学习率或 reward。`CUDA_VISIBLE_DEVICES` 接受物理卡编号或完整 GPU UUID；`GPUS_PER_NODE` 必须与之相符。启动器固定单节点，TP=2，要求 GPU 数可整除 128 条响应的全局 mini-batch。其他合法卡数仅通过算术检查，并不表示已完成该拓扑的训练验证。

`RUN_TAG` 决定新实验目录，默认 `repro_v1`。`PROJECT_NAME` 仅影响记录名称。需要使用外部已配置环境时，在设置文件指定 `PYTHON_BIN`；仍会严格检查锁定版本。不要在设置文件写密码或 API key。

## 4. 检查、冒烟与正式启动

```bash
# 仅 CPU 配置检查：不读取模型/数据，不连接 Ray、不训练。
bash run.sh config all

# 检查环境真实导入、完整资产哈希、GPU 空闲、内存/共享内存/磁盘。
bash run.sh check all

# 两组各完整 batch 训练 2 步，分别保存/验证 step 2。
bash run.sh smoke all

# 两组独立从 Base 开始，各跑 300 步；按 DAPO → NCBR 顺序执行。
bash run.sh train all
```

也可单独启动 `bash run.sh train dapo` 或 `bash run.sh train ncbr`。建议放在 `tmux`/`screen` 持久终端中前台运行。脚本不建立 systemd 服务，不保证重启机器后自动恢复。

`check` 检查的是正式运行目录与正式磁盘预算；`smoke` 使用独立目录与较小的 checkpoint 磁盘预算，实验 batch、n 和长度全部保持原值。冒烟结果不会用作正式初始化；正式运行重新从 Base 开始。

第一阶段可能长时间进行 179 万行数据的 tokenizer/长度过滤与初始验证，这是原数据规模所致。查看日志，不要为了缩短等待擅自抽样或改变数据。脚本不会自动降低 batch、截断长度、重试训练或停止其他 GPU 进程。

预检查要求目标 GPU 利用率为 0、显存占用不超过 1GiB、没有可见计算进程；启动前再次检查，并用每卡本地锁防止本脚本重复启动。外部程序不受该锁控制，机器应为该任务独占。使用本地 Ray，忽略继承的历史集群地址与网卡绑定，绝不执行全局 `ray stop`。

## 5. 配置对齐表

| 设置 | DAPO 2K | NCBR H=2K L=8K |
|---|---|---|
| 历史参考 | qwen3_4b_base_dapo_b256_g768_m16_n8_r2048_s600_v1 | qwen3_4b_base_boundary_return_b256_g768_m16_n8_h2048_l8192_s300_v1 |
| 本次 trainer step | 300（历史前半程） | 300（不读取旧 step-20） |
| prompt 上限 / 初始响应 H | 1024 / 2048 | 同左 |
| 总响应 L / vLLM context | 无续写 / 3072 | 8192 / 9216 |
| 训练 / 生成 prompt batch | 256 / 768 | 同左 |
| rollout n / PPO mini-batch | 8 / 16 个 prompt group | 同左 |
| 动态筛选 | acc、最多 10 个 generation batch | 配置仍为 acc；replace 路径实际使用 boundary_acc |
| Advantage / loss | GRPO、组内标准差归一化 / token-mean | 同左 |
| Optimizer | AdamW、lr=1e-6、weight decay=0.01、constant、无 warmup | 同左 |
| PPO | epoch=1、clip low/high=0.20/0.28、clip c=3、grad clip=1 | 同左 |
| KL / entropy loss / critic | 全部关闭 | 同左 |
| 训练 sampling | temperature=1、top_p=1、top_k=-1、正常 EOS | 同左 |
| 超长惩罚 | H=2048、buffer=410、penalty=1 | 保留短回答的非任务奖励项 |
| 验证 | 初始＋每 10 步、greedy、n=1、H=2048 | 同左；验证不启用 NCBR 续写 |
| 保存 | 每 100 步 | 每 50 步 |
| NCBR | off | replace、fixed_wave、并发 128、batch 512、超时 600s、reward chunk 256 |

NCBR 的 L=8192 是**包含初始 H 的总回答长度**，不是再生成 8192 个 token。对触及 H 上限的回答，用同一版本策略继续自然生成；完整回答只用于奖励判定。通过任务 score 的差量修正边界奖励，保留其他 reward 项；actor response/mask 仍是原始 H-token 数据，尾部不参与策略梯度。

每个 trainer step 使用 256×8=2048 条保留响应；PPO mini-batch 是 16×8=128 条响应，1 epoch 对应 16 次内部 optimizer update。单机 8 卡、SP=1、micro-batch/GPU=1 时，每个 mini-batch 每卡累积 16 次。不能把 `ppo_mini_batch_size=16` 误改成 128。

`reference/` 是经过旧 reward 接口迁移、路径脱敏后的历史有效配置；`configs/` 是当前接口下的冻结运行配置。`provenance/alignment.json` 列出每个历史字段的变化原因及当前接口新增字段；每次启动还会输出本次实际路径和拓扑对应的差异。新增的 Probe/Readiness/BSSF/OBCF、clip schedule、transfer queue 等均关闭；本次复现不启用其他实验方法。

执行参数保留 FSDP1、BF16 mixed precision、FP32 model dtype、TP=2、SP=1；actor/log-prob micro-batch 每卡 1/2；optimizer offload=true、param offload=false；vLLM memory utilization=0.50、max sequences=256、batched tokens=32768；动态 micro-batch=false，gradient checkpointing=true，enforce eager=false。

随机性逐字段保留：FSDP seed=42、actor data_loader_seed=42、NCBR continuation seed=42、data.seed=null。普通 vLLM rollout 保留历史副本编号相关的默认 seed；续写 seed 还依赖轨迹 UID。卡数、软件路径及运行时非确定性变化后，不承诺逐 token 或逐参数完全相同。

## 6. 输出、完成判定与失败排查

输出位于 `$OUTPUT_ROOT/{dapo,ncbr}_{s300,smoke_s2}_$RUN_TAG/`：

- `train.log`：完整 driver/Ray 日志；`metrics.jsonl`：逐步结构化指标。
- `wandb/`：本地离线记录；需要联网同步时，自己登录后手动执行 `wandb sync <offline-run目录>`。
- `provenance/`：resolved config、历史差异、启动参数、环境、资产和 GPU 检查记录。
- `checkpoints/global_step_*/actor/huggingface/`：HF 导出；同级保存分片模型、optimizer、RNG/extra state，step 目录保存 dataloader state。
- `status.json`：running / failed / complete。只有训练进程成功、指标步数与验证齐全、预期 checkpoint 结构及 HF 权重可读时才标记 complete。

正式 DAPO 应保存 100/200/300；NCBR 保存 50/100/150/200/250/300。历史单份完整 checkpoint 约 66GB，两组九份约 600GB；默认全部保留，不自动删除。另为日志、离线 W&B、下载和编译缓存预留空间。

| 问题 | 处理 |
|---|---|
| 模型/数据 SHA256 不符 | 重新复制原始文件；不要关闭校验或换成 Instruct |
| 依赖版本不符或导入失败 | 使用独立 `.venv`；检查 CUDA、torch、flash-attn 的 ABI；不要覆盖别的训练环境 |
| GPU busy / 显存不足 | 等待独占空闲 GPU，核对 80GB 规格；失败目录保留，不自动调整实验参数 |
| Ray object store / shm 错误 | 检查容器共享内存；默认至少 20GiB 空闲，建议 64GiB |
| 已有输出目录 | 修改 RUN_TAG 创建新实验，不能覆盖；失败的目录保留用于排查 |
| 进程退出但 status=failed | 看 train.log 和 status.json；可能未达到 300 步、缺少验证或 checkpoint 不完整 |

本启动器专用于**独立从 Base 重训**，不提供自动续训。若需要恢复中断训练，必须另行显式核验本次运行的完整 checkpoint、同一拓扑及 dataloader/optimizer 状态；不能加载历史 16 卡 checkpoint 到本 8 卡启动器冒充独立复现。

## 7. 可重复验收与边界

CPU 测试（不会启动 Ray 或训练）：

```bash
VERL_ROOT="$PWD/verl" .venv/bin/python verify_cpu.py
```

包内验证脚本检查两个冻结行为参考结果，并运行 NCBR、DAPO 动态采样和新启动器契约测试。仓库内使用时省略 VERL_ROOT，自动找到相邻源码。

已完成的检查与未完成项见 [VALIDATION.md](VALIDATION.md)。**当前交付没有在目标 8×80GB 机器执行训练、没有跑 300 步、也没有实测目标机峰值显存或最终分数。** 目标机执行 `smoke all` 后再启动正式训练；CPU 通过或 Ray 成功启动都不能代替 GPU 验收。
