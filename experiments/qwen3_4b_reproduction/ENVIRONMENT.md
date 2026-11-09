# 环境锁定与安装说明

首次在新机器运行，先按 [手动下载与路径配置](PATHS_AND_DOWNLOAD_ZH.md) 自行下载源码、模型和原始预处理数据，并选择自己的存储目录。本文所有宿主机路径只是示例；`settings.env` 必须手动填写新机器运行环境中实际可见的路径。

## 指定的容器镜像

本次复现使用用户指定的容器基础环境：

```text
docker.1ms.run/verlai/verl:vllm018.dev1
```

宿主机需要兼容 CUDA 12.9 的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。Python、CUDA toolkit、编译器及训练依赖在容器内检查；无需为容器训练另外在宿主机安装 Python 训练环境。这里记录的是指定镜像地址，尚未实际拉取该镜像或验证其内部包版本，不能仅凭 `vllm018.dev1` 标签认定它与锁定环境完全一致。

### 在宿主机启动容器

先在新机器手动下载并解压源码包、下载模型及三份原始数据。把下面 `HOST_CODE_DIR` 和 `HOST_DATA_DIR` 改成**新机器宿主机上的实际绝对路径**。`HOST_CODE_DIR` 应直接包含 run.sh、settings.env 和 verl/。下面选择固定的容器内挂载目标，方便填写 settings；这些目标路径不代表旧机器的目录要求。

```bash
set -euo pipefail
IMAGE="docker.1ms.run/verlai/verl:vllm018.dev1"
HOST_CODE_DIR="/mnt/nvme/qwen3-repro/qwen3_4b_reproduction"  # 手动改为实际源码包目录
HOST_DATA_DIR="/mnt/nvme/qwen3-repro/data"                 # 手动改为实际资产目录
test -f "$HOST_CODE_DIR/run.sh"
test -d "$HOST_CODE_DIR/verl"
mkdir -p "$HOST_DATA_DIR/runs"
docker pull "$IMAGE"

# tag 可能变化；保留本次实际镜像 ID 和 RepoDigests，便于后续核对。
docker image inspect "$IMAGE" --format '{{json .}}' \
  > "$HOST_DATA_DIR/runs/container-image.json"

docker run --gpus all -it \
  --name qwen3-4b-repro \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=524288:524288 \
  --mount "type=bind,source=$HOST_CODE_DIR,target=/workspace/qwen3_4b_reproduction" \
  --mount "type=bind,source=$HOST_DATA_DIR,target=/data" \
  --workdir /workspace/qwen3_4b_reproduction \
  --entrypoint /bin/bash \
  "$IMAGE"
```

代码目录、数据和训练输出均通过 bind mount 保存在宿主机。64GiB 共享内存用于满足默认 16GiB Ray object store 的空间要求。此命令面向单机，不需要 host networking 或 privileged 模式。所有训练命令都在容器内部执行；在宿主机持久终端中运行上述命令，避免关闭交互终端导致任务中断。容器停止后可用 `docker start -ai qwen3-4b-repro` 重新进入，但不会自动恢复训练。

### 在容器内选择 Python 环境

```bash
nvidia-smi
python3 --version
command -v python3
nvcc --version

# 先检查镜像自带 Python：不启动 Ray、不加载训练权重、不进行训练。
PYTHON_BIN="$(command -v python3)" bash run.sh environment
```

首次检查前，`settings.env` 中的 `PYTHON_BIN` 应保持注释，避免覆盖上面传入的解释器。`environment` 会检查锁定依赖版本、实际训练模块导入和本包源码，不会因为镜像内预装了另一份 verl 就使用它。

- **检查通过：**将 `command -v python3` 输出的绝对路径填入 `settings.env` 的 `PYTHON_BIN`，复用镜像环境，跳过 `setup_env.sh`。启动器通过 `PYTHONPATH` 加载本包源码，无需覆盖镜像预装的 verl。
- **检查未通过：**按下文锁定安装流程，在容器内运行 `bash setup_env.sh` 创建独立 `.venv`。先确认容器内有 uv、CUDA 12.9 toolkit 和 C++ 编译器；不要直接升级或覆盖镜像的全局依赖。安装完成后保持 `PYTHON_BIN` 注释，使启动器使用包内 `.venv/bin/python`。

无论选择哪条 Python 路径，都要在 `settings.env` 手动填写源码、模型、三份数据和输出路径。按上述挂载示例：`VERL_ROOT=/workspace/qwen3_4b_reproduction/verl`，模型为 `/data/models/Qwen3-4B-Base`，三份数据为 `/data/*.parquet`，输出为 `/data/runs`。完整变量表及修改示例见 [路径配置指南](PATHS_AND_DOWNLOAD_ZH.md)，配置后依次执行：

```bash
bash run.sh config all
bash run.sh check all
bash run.sh smoke all
bash run.sh train all
```

镜像环境检查通过不等于 GPU 冒烟通过；两组各 2 步的实际 GPU 测试仍需在目标机器执行。

## 锁定依赖与独立环境安装

`requirements.lock` 由原运行环境的训练依赖递归闭包生成，包含 Ray default、vLLM 指定的 extras 及 CPU 测试依赖。所有条目固定版本；未纳入 Megatron、Mindspeed、其他项目环境或操作系统 apt 包。torch/torchvision/torchaudio 从 CUDA 12.9 wheel 源单独安装，FlashAttention 单独构建。`provenance/dependencies.json` 记录完整版本与依赖声明例外。

本机的 cryptography 41.0.7 来自 Debian 系统包，其 metadata 不包含 PyPI wheel 对 CFFI 的声明。因此另补充锁定 `cffi==1.17.1`、`pycparser==2.22`，供新建隔离环境安装。它们不改变训练算法；环境检查对新虚拟环境要求这两个版本，对原系统解释器允许已有系统 CFFI backend，并实际执行一次本地密码库运算验证二进制可用。总计 224 个历史依赖版本＋2 个可移植安装补充项。

历史实际环境存在两个已核实的 metadata 冲突：

| 声明方 | 声明要求 | 历史实际使用 |
|---|---|---|
| vLLM 0.18.0 | transformers >=4.56,<5 | Transformers 5.3.0 |
| opencv-python-headless 4.13.0.92 | Python≥3.9 时 numpy≥2 | NumPy 1.26.4 |

因此，不能要求普通 resolver 在这些声明下同时满足历史版本。安装脚本对**完整锁定依赖闭包**使用 `uv pip install --no-deps`，再逐项验证实际版本与训练模块导入，不偷偷升级 NumPy 或降级 Transformers。文本数学训练不使用 OpenCV 图像路径；这些例外不是对所有多模态用途的兼容性保证。`pip check` 会报告上述冲突，不能把它描述为零冲突环境。

安装前需有 Python3、uv、NVIDIA 驱动、CUDA 12.9 toolkit、C/C++ 编译器。建议 Ubuntu 24.04；Python 使用 `uv venv --python 3.12`，torch 为 2.10.0+cu129。安装脚本不会自动修改宿主机驱动，不会在已有 `.venv` 上覆盖安装。

所有源依赖均由固定版本的索引 wheel/sdist 获取；当前交付不是含 wheel 的离线包。安装材料需要访问 PyPI、PyTorch wheel 源，以及构建工具可能使用的下载源。PyTorch 下载体积较大，FlashAttention 可回退到本机编译，因此必须有 toolkit。驱动与 CUDA toolkit 的安装方式取决于目标宿主机，应由机器管理员预先配置。

源码通过 `uv pip install --no-deps --no-build-isolation -e ./verl` 安装。不要用源项目的 `[vllm]` extra，它声明的版本范围与历史 0.18.0 环境不同。

运行时缓存放在 `CACHE_ROOT` 下，默认 `/tmp/q4repro-cache`；Ray session 使用独立短目录 `/tmp/q4r-*`。缓存及 session 日志不会自动清理，可在确认对应任务已退出后手动清理。完整环境检查会显示已知 metadata 冲突，同时验证固定包版本、真实导入和源码哈希。

本次交付在现有历史依赖环境做了导入及 CPU 行为验证；没有重新下载并建立一个完全干净的目标机器环境。最终安装兼容性由目标机的 `setup_env.sh`、`check all` 和 `smoke all` 分阶段确认。
