# 新机器：手动下载与路径配置

所有下载、目录选择和配置都由你在**新机器**上手动完成。本文路径是示例，不要求新机器存在旧机器的 `/workspace/rl`、`/workspace/models` 或 `/data`。启动器不自动下载模型或数据，也不会探测旧机器目录。

先选宿主机上的实际存储目录，再设置 Docker 挂载，最后在 `settings.env` 填写**容器内可见路径**。宿主机路径和容器路径不必相同。

## 1. 在新机器下载源码包

以下命令在**新机器宿主机**执行。先把第一行改成你有权限、容量足够的实际目录；例如 `/mnt/nvme/qwen3-repro`，并非固定要求。

```bash
WORK_ROOT="/mnt/nvme/qwen3-repro"  # 必须按新机器实际磁盘修改
HOST_REPO_DIR="$WORK_ROOT/h100"
HOST_CODE_DIR="$WORK_ROOT/qwen3_4b_reproduction"
HOST_DATA_DIR="$WORK_ROOT/data"
mkdir -p "$WORK_ROOT" "$HOST_DATA_DIR/models" "$HOST_DATA_DIR/runs"

git clone --depth 1 --single-branch --branch feat/ncbr \
  https://github.com/PorkFormer/h100.git "$HOST_REPO_DIR"
git -C "$HOST_REPO_DIR" rev-parse HEAD

cd "$HOST_REPO_DIR/delivery"
sha256sum -c qwen3_4b_reproduction.tar.gz.sha256
tar -xzf qwen3_4b_reproduction.tar.gz -C "$WORK_ROOT"
cd "$HOST_CODE_DIR"
sha256sum -c SHA256SUMS
```

也可以从仓库 `delivery/` 手动下载压缩包和 `.sha256` 文件再解压。GitHub 不可用时，可将 clone 地址换为 `https://gitee.com/a3083578728/h100.git`，分支仍为 `feat/ncbr`。下载到已存在的目录时先检查内容，不要覆盖已有训练结果。

后续以**解压后的独立包**为主：`HOST_CODE_DIR` 内应直接包含 `run.sh`、`settings.env`、`setup_env.sh` 和 `verl/`；不是指 `h100/` 仓库根目录。

## 2. 在新机器手动下载模型与数据

下面示例仍在同一个宿主机 shell 中执行；如果换了终端，重新设置第 1 节中的目录变量。

### Base 模型

下载的是 `Qwen/Qwen3-4B-Base`，不是 Instruct 模型，也不是某次训练的 checkpoint。当前 Base 本地下载记录中的 revision 为 `906bfd4b4dc7f14ee4320094d8b41684abff8539`，下面固定该 revision；最终以包内资产哈希检查为准。

示例使用 uv 的隔离下载工具，不安装进训练 Python。若下载环境尚无 uv，先安装并加入 PATH（已有 uv 时跳过）：

```bash
curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

```bash
HOST_MODEL_DIR="$HOST_DATA_DIR/models/Qwen3-4B-Base"
uvx --from huggingface-hub==1.8.0 hf download Qwen/Qwen3-4B-Base \
  --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --local-dir "$HOST_MODEL_DIR"
```

也可以在容器启动后使用容器里的 `hf download`，此时 `--local-dir` 必须改为对应的**容器内模型目录**。模型目录应直接包含 `config.json`、tokenizer 文件、`model.safetensors.index.json` 和全部权重分片，不要把 MODEL_PATH 指到上一级 models 目录。

### 三份原始预处理 Parquet

这三个文件**未包含在源码包中，目前没有已核实的公开下载 URL**。需要你获得原始预处理文件的可访问下载地址（例如自己的文件服务或对象存储地址），在新机器手动填写并下载。不能拿公开 DAPO/AIME 原始题库直接替代，也不能仅凭文件名一致判断内容一致。

```bash
# 手动输入你实际取得的三个原始文件下载地址。
read -r -p '原始训练 Parquet 下载 URL: ' TRAIN_URL
read -r -p '原始 AIME2024 Parquet 下载 URL: ' VAL24_URL
read -r -p '原始 AIME2025 Parquet 下载 URL: ' VAL25_URL

curl --fail --location --retry 3 "$TRAIN_URL" \
  --output "$HOST_DATA_DIR/dapo_math_17k_train.parquet"
curl --fail --location --retry 3 "$VAL24_URL" \
  --output "$HOST_DATA_DIR/aime-2024-verl.parquet"
curl --fail --location --retry 3 "$VAL25_URL" \
  --output "$HOST_DATA_DIR/aime-2025-verl.parquet"
```

下载后保留原始内容，包括题目、答案、样本重复和顺序。模型和数据完整哈希由 `provenance/assets.json` 固定，运行 `check all` 时会核验：

| 文件 | 行数 | SHA256 |
|---|---:|---|
| dapo_math_17k_train.parquet | 1,791,700 | `816b8778751076fdbc882ce65045aa80bb59db1618ce36e8308903db90ae92f4` |
| aime-2024-verl.parquet | 960 | `ef67aff23b592844f2af5ecd290ef57fe447fd8158ac7534ff9dbac217d6730f` |
| aime-2025-verl.parquet | 30 | `7dc018ef3092aa894e285158eb0de7b6b065dddc2b197244b4e57189212ed93a` |

路径与文件名可以自行改变，内容哈希不能改变。若下载内容不同，先确认版本和文件来源，不要修改哈希清单绕过检查。

## 3. 宿主机路径与容器路径对照

按 [ENVIRONMENT.md](ENVIRONMENT.md) 的 Docker 命令，将 `HOST_CODE_DIR` 挂载到 `/workspace/qwen3_4b_reproduction`，将 `HOST_DATA_DIR` 挂载到 `/data`。以下是上述示例对应关系：

| 用途 | 新机器宿主机示例路径 | 容器内路径 / 设置 |
|---|---|---|
| 独立包、启动命令工作目录 | `/mnt/nvme/qwen3-repro/qwen3_4b_reproduction` | `/workspace/qwen3_4b_reproduction` |
| 配置文件 | 上述包目录下 `settings.env` | `/workspace/qwen3_4b_reproduction/settings.env` |
| verl 源码根目录 | 上述包目录下 `verl` | `VERL_ROOT=/workspace/qwen3_4b_reproduction/verl` |
| Base 模型 | `/mnt/nvme/qwen3-repro/data/models/Qwen3-4B-Base` | `MODEL_PATH=/data/models/Qwen3-4B-Base` |
| 训练数据 | `/mnt/nvme/qwen3-repro/data/dapo_math_17k_train.parquet` | `TRAIN_FILE=/data/dapo_math_17k_train.parquet` |
| AIME2024 验证数据 | `/mnt/nvme/qwen3-repro/data/aime-2024-verl.parquet` | `VAL_FILE_1=/data/aime-2024-verl.parquet` |
| AIME2025 验证数据 | `/mnt/nvme/qwen3-repro/data/aime-2025-verl.parquet` | `VAL_FILE_2=/data/aime-2025-verl.parquet` |
| 训练输出与 checkpoint | `/mnt/nvme/qwen3-repro/data/runs` | `OUTPUT_ROOT=/data/runs` |
| 独立虚拟环境 Python | 上述包目录下 `.venv/bin/python` | `/workspace/qwen3_4b_reproduction/.venv/bin/python` |
| 编译缓存 | 默认不挂载到宿主机 | `CACHE_ROOT=/tmp/q4repro-cache` |

`VERL_ROOT` 指向含 **setup.py 和内部 verl/ Python 包**的目录。独立包的正确层级是 `/workspace/qwen3_4b_reproduction/verl`，不要误填 `/workspace/qwen3_4b_reproduction/verl/verl`，也不要填镜像中另一份预装源码。

如果宿主机换成 `/home/user/disk/experiment`，只需修改 Docker 的挂载源路径；只要挂载目标仍是上述容器路径，`settings.env` 可以不变。若连挂载目标也改为 `/opt/repro` 和 `/training-data`，就必须将 settings 中对应路径一起改成 `/opt/repro/verl`、`/training-data/models/...` 等。

直接使用 Git checkout 而不解压独立包时，必须把**整个 h100 仓库**挂载进容器。例如挂载到 `/workspace/h100` 后，启动目录为 `/workspace/h100/experiments/qwen3_4b_reproduction`，`VERL_ROOT=/workspace/h100/verl`，默认 `.venv` 位于启动目录下。仅挂载 experiments 子目录会缺少相邻源码。不要混用 Git checkout 和独立包的路径层级。

## 4. 手动填写 settings.env

在容器内编辑 `/workspace/qwen3_4b_reproduction/settings.env`。下面是对应上述挂载方式的完整**路径部分**，将它替换进现有文件，保留原 GPU、Ray、RUN_TAG 等设置：

```bash
VERL_ROOT="/workspace/qwen3_4b_reproduction/verl"
MODEL_PATH="/data/models/Qwen3-4B-Base"
TRAIN_FILE="/data/dapo_math_17k_train.parquet"
VAL_FILE_1="/data/aime-2024-verl.parquet"
VAL_FILE_2="/data/aime-2025-verl.parquet"
OUTPUT_ROOT="/data/runs"
CACHE_ROOT="/tmp/q4repro-cache"

# 独立 .venv：保持注释，run.sh 自动选择包目录下的 .venv/bin/python。
# PYTHON_BIN="/workspace/qwen3_4b_reproduction/.venv/bin/python"
```

若复用镜像自带环境，先在容器里运行 `command -v python3`，并按环境文档通过 `environment` 检查，然后将该命令输出的**实际绝对路径**填入 `PYTHON_BIN` 并取消注释。不要猜测一定是 `/usr/bin/python3` 或 `/usr/local/bin/python3`。

`VERL_ROOT` 可以像上面显式填写；标准目录布局下也可以保持注释，由启动器自动定位。`CACHE_ROOT` 必须可写；希望持久保留编译缓存时可手动改成 `/data/cache/q4repro`。Ray socket 仍由启动器使用独立的短路径 `/tmp/q4r-*`，无需自己配置。

`RUN_TAG=repro_v1` 时，正式输出为 `$OUTPUT_ROOT/dapo_s300_repro_v1/` 和 `$OUTPUT_ROOT/ncbr_s300_repro_v1/`。权重位于各目录的 `checkpoints/global_step_<step>/actor/huggingface/`；同目录有日志和离线 W&B。不要将 OUTPUT_ROOT 填为 MODEL_PATH。

不使用容器、直接在新机器运行时，settings 中全部填**新机器本地绝对路径**，例如 `VERL_ROOT=/mnt/nvme/qwen3-repro/qwen3_4b_reproduction/verl`。无须创建任何示例中的容器目录。

## 5. 检查手动配置

完成下载、容器启动、环境选择和 settings 编辑后，在容器内执行：

```bash
set -euo pipefail
cd /workspace/qwen3_4b_reproduction

# 只展开并检查路径；settings.env 是你手工维护的可信 shell 配置。
set -a
source settings.env
set +a
test -f "${VERL_ROOT:-$PWD/verl}/setup.py"
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$TRAIN_FILE"
test -f "$VAL_FILE_1"
test -f "$VAL_FILE_2"
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"

bash run.sh config all
bash run.sh check all
bash run.sh smoke all
bash run.sh train all
```

请逐项确认命令成功后再继续；`test` 失败表示文件位置或挂载配置有误。`config` 仅校验配置；`check` 才会核验完整文件内容、依赖与 GPU；最后两条命令会实际训练。上述设置和下载均不会自动同步旧机器上的路径或训练状态。
