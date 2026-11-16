# DAPO 4K：单节点 8×A100 80GB，独立训练300步

本入口对齐历史 `qwen3_4b_base_dapo_b256_g768_m16_n8_r4096_s600_v1`，从原始 **Qwen3-4B-Base** 重新开始，训练到 **300个trainer step**。不会加载历史checkpoint；不要使用现有训练权重替代Base。

## 配置对齐

| 项目 | 本入口 |
|---|---|
| 训练 / 生成全局batch | 256 / 768 |
| PPO mini-batch / 每题rollout | 16 / 8 |
| prompt / response / 模型上下文 | 1024 / 4096 / 5120 |
| 超长惩罚 | enable=true，buffer=819，penalty_factor=1，max_resp_len=4096 |
| 优化器 | AdamW，lr=1e-6，constant，无warmup，weight_decay=0.01 |
| 算法 | GRPO优势估计及组内标准差归一化、DAPO动态过滤、token-mean损失；保持历史配置 |
| 并行 | FSDP1，rollout TP=2，SP=1；单节点8卡 |
| 每卡micro-batch | actor=1，rollout/ref log-prob=2；动态micro-batch关闭 |
| 内存 | optimizer offload=true，actor parameter offload=false，gradient checkpointing=true |
| vLLM | memory utilization=0.50，max_num_seqs=256，max_num_batched_tokens=32768 |
| 验证 | 开训前及每10步；原始两份AIME验证文件；greedy、n=1、回答上限4096 |
| checkpoint | step100、200、300；完整训练状态与HF导出全部保留 |
| NCBR / 其他实验功能 | 关闭 |

与历史主动变更：**2节点16卡→1节点8卡、600步→300步**，另调整路径、实验标识及本地日志。8卡保持全局batch和mini-batch，梯度累积由框架按实际world size计算；不把batch减半。学习率恒定，缩短总步数不压缩衰减曲线。GPU数、运行时和归约顺序变化仍可能带来数值差异，不承诺逐位复现历史轨迹。

旧 `reward_model.reward_kwargs` 已迁入当前 `reward.reward_kwargs`，保留历史4K惩罚参数。历史配置归一化记录及原文件SHA256见 `provenance/dapo4k_historical.json`；每次启动记录实际参数差异。当前入口复用冻结源码和依赖，不能改用上游默认verl。原 `dapo` 仍是2K；原 `all` 仍只包含2K与NCBR。**4K请显式使用 `dapo4k` 或本页专用脚本。**

## 1. 下载、容器和依赖

先按 [新机器手动下载与路径配置](PATHS_AND_DOWNLOAD_ZH.md) 下载更新后的源码包并核验SHA256，手动取得原始模型及三份预处理Parquet。包不包含模型和数据，不自动下载。

- Base模型：https://huggingface.co/Qwen/Qwen3-4B-Base ，固定revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`。
- 原始文件：`dapo_math_17k_train.parquet`、`aime-2024-verl.parquet`、`aime-2025-verl.parquet`；具体哈希见下载指南。训练集仍需自行取得原始文件，不能换同名其他版本。合并后的评测协议已提供AIME两份验证数据的固定公开版本，见下方手动下载命令。
- 容器：`docker.1ms.run/verlai/verl:vllm018.dev1`。完整挂载、驱动要求和安装步骤见 [ENVIRONMENT.md](ENVIRONMENT.md)。镜像标签不是环境一致性的证明。
- 需要8张空闲A100 80GB，主机建议512GiB内存（检查至少128GiB可用）、共享内存64GiB、输出盘至少250GB可用；模型、数据、编译缓存另预留空间。两步冒烟额外保留独立checkpoint，其磁盘预算约110GB；做完冒烟后，正式启动仍要求250GB可用。

AIME验证文件可在新机器手动下载到选择的数据目录（之后仍由训练预检核验原始SHA256）：

```bash
DATA_DIR="/mnt/nvme/qwen3-repro/data"  # 改为新机器实际目录
mkdir -p "$DATA_DIR"
for FILE in aime-2024-verl.parquet aime-2025-verl.parquet; do
  curl --fail --location --retry 3 \
    "https://huggingface.co/datasets/Ottoman6324/rlvr/resolve/9d19f9b64ac9da472281617f0818a532185fe86a/$FILE" \
    --output "$DATA_DIR/$FILE"
done
```

下面是容器挂载示例，先手动修改宿主机目录。源码目录应直接包含 `run.sh` 和包内 `verl/`：

```bash
HOST_CODE_DIR="/mnt/nvme/qwen3-repro/qwen3_4b_reproduction"
HOST_DATA_DIR="/mnt/nvme/qwen3-repro/data"
IMAGE="docker.1ms.run/verlai/verl:vllm018.dev1"
mkdir -p "$HOST_DATA_DIR/runs"
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .}}' > "$HOST_DATA_DIR/runs/dapo4k-container-image.json"
docker run --gpus all -it --name qwen3-dapo4k \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  --ulimit nofile=524288:524288 \
  --mount "type=bind,source=$HOST_CODE_DIR,target=/workspace/qwen3_4b_reproduction" \
  --mount "type=bind,source=$HOST_DATA_DIR,target=/data" \
  --workdir /workspace/qwen3_4b_reproduction --entrypoint /bin/bash "$IMAGE"
```

以下命令在容器内、源码包目录执行：

```bash
nvidia-smi
# 首先保持settings.env里的PYTHON_BIN注释，检查容器自带环境。
PYTHON_BIN="$(command -v python3)" bash run_dapo4k.sh environment
```

若通过，把实际Python绝对路径写入 `settings.env` 的 `PYTHON_BIN`。若未通过，按环境文档准备uv/CUDA编译工具并执行 `bash setup_env.sh`，然后保持 `PYTHON_BIN` 注释，使用包内 `.venv/bin/python`。不要直接升级容器全局依赖。检查锁定Python3.12、torch2.10.0+cu129、vLLM0.18.0等版本及真实导入；仅看pip列表不够。

## 2. 在新机器手动配置路径

编辑 `settings.env`。下例均为上面挂载后的**容器内路径**，不是要求新机器沿用旧机器目录：

```bash
MODEL_PATH="/data/models/Qwen3-4B-Base"
TRAIN_FILE="/data/dapo_math_17k_train.parquet"
VAL_FILE_1="/data/aime-2024-verl.parquet"
VAL_FILE_2="/data/aime-2025-verl.parquet"
OUTPUT_ROOT="/data/runs"
CACHE_ROOT="/tmp/q4repro-cache"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=8
RAY_CPUS=32
RAY_OBJECT_STORE_GB=16
RUN_TAG="dapo4k_v1"
PROJECT_NAME="qwen3_4b_reproduction"
# 独立包自动定位以下源码根；手动设置时指向含setup.py的外层verl。
# VERL_ROOT="/workspace/qwen3_4b_reproduction/verl"
# 仅复用已通过检查的环境时填写，否则使用默认.venv：
# PYTHON_BIN="/usr/bin/python3"
```

仓库内直接运行时，先 `cd experiments/qwen3_4b_reproduction`，默认源码自动定位到仓库 `verl/`。若用配置副本，每条命令前加 `SETTINGS_FILE="/path/to/my-settings.env"`。路径中有空格须保持引号。

W&B固定offline，**不需要登录或API key**，训练指标还写入本地JSONL。GPU必须由本任务独占；启动器检查8卡可见性、显存和空闲状态，使用本地Ray，不需要手动组建两节点集群或连接旧Ray地址。

## 3. 检查、冒烟和一键正式启动

```bash
bash run_dapo4k.sh environment  # 源码及依赖检查
bash run_dapo4k.sh config       # CPU配置校验；不启动Ray或训练
bash run_dapo4k.sh check        # 模型/数据哈希、磁盘/内存、GPU预检
bash run_dapo4k.sh smoke        # 可选：真实GPU两步冒烟，保留完整batch及4K长度
bash run_dapo4k.sh              # 一键从Base正式训练到300步，前台运行
```

等价正式入口：`bash run.sh train dapo4k`。建议在宿主机tmux等持久终端中启动容器，保留挂载目录；不要关闭前台会话导致训练中断。

冒烟输出 `OUTPUT_ROOT/dapo4k_smoke_s2_<RUN_TAG>/`；正式输出 `OUTPUT_ROOT/dapo4k_s300_<RUN_TAG>/`。二者完全独立，正式训练不读取冒烟权重。已有输出目录拒绝覆盖，重跑时修改 `RUN_TAG`；不自动恢复失败训练，也不自动删checkpoint。

若OOM，检查80GB卡是否空闲、容器内GPU是否可见、依赖是否匹配；不要自行减小全局batch或采样数后仍称为同配置复现。检查失败时先根据提示修复环境或路径。

## 4. 输出与完成标准

- `train.log`、`metrics.jsonl`、`wandb/`：日志与离线指标。
- `provenance/resolved_config.yaml`、`historical_diff.json`、`preflight.json`：实际配置、对齐差异及预检证据。
- `checkpoints/global_step_{100,200,300}/`：`data.pt`及actor全部8个rank的model、optim、extra_state文件，另含 `actor/huggingface/`。
- `status.json`：只有训练正常退出、1–300步指标、初始/每10步验证以及全部预期checkpoint检查通过，才标记complete。

保留整个global_step目录才具备未来完整恢复状态的基础；仅HF权重不能恢复optimizer和数据进度。此入口不提供续训模式。100/200/300导出可用于独立离线评测；训练内greedy 4K验证与离线独立2K/8K采样是不同协议，详见 [OFFLINE_EVAL_ZH.md](OFFLINE_EVAL_ZH.md)，比较时另设对应checkpoint路径和输出目录。

本次交付执行CPU配置/回归验证及交付包校验；不代表已在目标容器完成GPU冒烟或300步训练。用户在目标机器成功运行smoke后再启动正式训练。
