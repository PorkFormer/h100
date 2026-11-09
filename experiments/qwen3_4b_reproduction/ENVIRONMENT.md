# 环境锁定与安装说明

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
