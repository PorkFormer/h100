# 从 checkpoint 离线评测到训练动态图表

本教程使用仓库根目录的 `evaluate.sh` 和 `plot_training.sh`。绘图只读取明确指定的 JSONL，不采集训练数据、不运行模型、不修改训练进程。离线评测是独立 GPU 工作负载，必须另行启动。本文示例路径需要替换为自己的实际资产路径。

## 1. 环境准备

普通服务器可在本机运行 CPU 绘图；离线生成需要兼容的 NVIDIA GPU。HPC4 登录节点只编辑、检查和提交作业，CPU 渲染、重评分、权重合并也必须在调度分配内执行。计算节点无外网，提前在登录节点安装依赖、准备模型、tokenizer 和固定数据。

绘图不需要 torch、verl、Ray、W&B 或 GPU，使用独立环境：

```bash
uv venv --python 3.12 /data/envs/training-plots
uv pip install --python /data/envs/training-plots/bin/python \
  -r analysis/training_plots/requirements.txt
export PLOT_PYTHON=/data/envs/training-plots/bin/python
```

可在有网机器下载 wheel 并离线安装：`uv pip install --python ... --no-index --find-links /data/wheels -r analysis/training_plots/requirements.txt`。HTML 内嵌 Plotly JS，直接双击打开，无需 CDN、服务端或网络；PNG/PDF 由 Matplotlib 导出，不需要 Chrome/Kaleido。静态图保留英文原始指标名，避免依赖中文字体。

评测使用项目固定版本环境，见[统一评测说明](../analysis/offline_eval/README_ZH.md)与[环境搭建](../experiments/qwen3_4b_reproduction/ENVIRONMENT.md)。HPC4 的资产路径见项目环境手册：Apptainer 必须有 squashfuse PATH 前缀与 `/fs1` 挂载；节点必须支持 `/dev/fuse`。绘图建议直接使用共享 Python 环境，不需要容器。安装/下载所需代理由用户环境配置；离线绘图本身不使用代理。

## 2. 准备 checkpoint、固定数据和 tokenizer

选择包含 `config.json`、完整 safetensors（分片时含完整 index）的 HF 目录。真实训练 step 应从训练日志与 checkpoint 回执交叉确认，不能从目录名推断。只有 FSDP 分片时，用**训练时同版本**的原生工具合并到新目录，保留原分片：

```bash
PYTHONPATH=/path/to/h100/verl python -m verl.model_merger merge \
  --backend fsdp --local_dir /data/checkpoint/actor --target_dir /data/checkpoint-hf
```

该操作可能占用大量 CPU/内存，HPC4 使用 CPU 调度作业。HF 导出只用于推理，不等于保留优化器/RNG/数据状态的完整续训 checkpoint。

评测协议冻结在 `analysis/offline_eval/protocol.lock.json`；数据为 `Ottoman6324/rlvr` 固定提交 `9d19f9b64ac9da472281617f0818a532185fe86a`，tokenizer 必须匹配冻结哈希。默认源数据缓存 `~/.cache/ncbr-offline-eval/data`，使用 `--data-dir` 显式指定共享缓存。普通服务器 `run` 可下载缺失的小型输入；HPC4 `submit` 在登录节点准备并校验输入，然后计算节点离线运行。模型由使用者预先准备；不自动下载权重。

## 3. 手动启动、检查和完成判定

普通 GPU 服务器：

```bash
bash evaluate.sh run --checkpoint /data/checkpoint-hf \
  --tokenizer /data/Qwen3-4B-Base --model-id ncbr_s300 --step 300 \
  --data-dir /data/eval-cache --output /data/results/ncbr_s300 \
  --gpus 2 --datasets AIME2024 AIME2025 AMC23 --horizons 2048 8192
```

默认四套题（另含 MATH500）、每题32条、H8192；可选择固定预算 2048/8192/16384。预算各自独立生成，不截取长回答冒充短预算。采样/评分协议不可任意改写。

HPC4 登录节点提交（资源按实际分区调整）：

```bash
bash evaluate.sh submit --checkpoint /fs1/path/checkpoint-hf \
  --tokenizer /fs1/path/Qwen3-4B-Base --model-id ncbr_s300 --step 300 \
  --data-dir /fs1/path/eval-cache --output /fs1/path/results/ncbr_s300 \
  --partition gpu1 --gpus 8 --cpus-per-gpu 8 --mem 128G --time 12:00:00
bash evaluate.sh status --output /fs1/path/results/ncbr_s300
```

`--dry-run`只准备脚本；`SUBMITTED`只代表提交。结合 `squeue -j JOB_ID`、`sacct -j JOB_ID`、`<output>.submission/` 日志判断排队/运行/退出。结果目录的 `COMPLETE` 必须与 `scoring_receipt.json` 和 `driver_exit.json` 一致；检查每个分片已生成/预期数量。仅作业退出0不足以证明训练能力提高。

失败恢复：数据/tokenizer 哈希不符时修复输入，不跳过校验；显存不足时检查资源与固定协议兼容性，不偷偷减少采样数；FUSE/容器失败保留错误并换合适环境；生成失败用新输出目录重跑，不覆盖旧轨迹；完整生成后的评分失败可独立重评分。

## 4. 重评分与配对比较

CPU 作业中执行，无需模型权重或 GPU：

```bash
bash evaluate.sh score --output /data/results/ncbr_s300
bash evaluate.sh compare --left /data/results/dapo_s300 \
  --right /data/results/ncbr_s300 --output /data/results/comparison
```

HPC4 前置 `yhrun -N 1 -p debug -n 1 -c 4`，并确保评测 Python/容器参数可用。比较要求真实 step、题目、预算和协议一致，结果为右侧减左侧的逐题配对 bootstrap。Avg@32 是所有题每题32次的平均正确率（pass@1 估计），pass@32 是每题32次中至少一次正确的题目比例；95% 区间按题重采样10000次、seed 20260812，不能按轨迹独立重采样。不同硬件的数值差异仍须记录。

保留 `manifest.json`、`metrics.csv`、`scoring_receipt.json`、`REPORT_ZH.md` 与原始轨迹。评分异常不能当作答错；重评分会撤下旧成功回执，失败后不得继续引用旧分数。

## 5. 原生训练日志

本工具消费已有原生 file logger：

```bash
mkdir -p /data/run/logs
export VERL_FILE_LOGGER_PATH=/data/run/logs/metrics.jsonl
# 在既有训练启动命令中设置 Hydra 参数：
# trainer.logger='[console,file]'
```

这只是配置说明；绘图命令不会启动训练。`VERL_FILE_LOGGER_PATH` 的父目录须存在。当前原生 FileLogger 以 `wb` 打开文件，重启可能截断旧日志；每次训练尝试使用新路径并保存配置最可靠。格式每行 `{"step":1,"data":{"critic/rewards/mean":-0.5,"timing_s/step":100}}`。如果已有日志没开 file logger，本工具不会从终端文本推测缺失指标。

指标由原生 logger/算法是否记录决定：

| 组 | 内容与口径 |
|---|---|
| 1 训练奖励 | `critic/rewards/*`、score；保留训练批次的奖励，不转成准确率 |
| 2 训练验证 | `val-core/*`、`val-aux/*`；结合配置确认 greedy，区别于离线 Avg@32/pass@32 |
| 3 回答长度 | 保留训练批次的 response/prompt 长度与截断比例，不当作筛选前候选长度 |
| 4 策略变化 | entropy、PPO old/new 近似 KL；不冒充参考模型 KL |
| 5 更新稳定性 | 梯度范数、loss、clip、LR、advantages/returns |
| 6 动态采样 | 生成/过滤/保留组数、候选轨迹数、原生 accept rate；不能用最终保留/生成替代 |
| 7 NCBR触发 | 候选数、hit cap、有效长评分数、续写请求、long cap |
| 8 续写收益 | recovered/regressed、四类短长转换、task score delta；显示计数和有分母的条件比例 |
| 9 组解锁 | short locked/all wrong、unlocked/newly locked 计数与比例 |
| 10 额外成本 | 续写输入/新增 tokens、尾部长度、原生分阶段耗时；阶段可能嵌套 |
| 11 速度 | step/阶段耗时、throughput、记录中的训练版本/step |
| 12 资源与异常 | 聚合内存、MFU、负载长度、超时、漂移、中断与后端哨兵值 |

恢复率分母由短错→长错、短错→长对计数相加；回退率用短对两类计数。旧日志的 newly locked 分母缺失时仅以显式 generated prompt groups 减 short locked 推得。没有有效分母时显示不可用，同时保留原始0值于CSV。后端 queue/prefill/decode 细分的0值在无法确认采集有效时不画成零耗时；负抢占计数视为后端哨兵值。内存是原生 worker 聚合值，不能声称逐卡显存；MFU 不等于 GPU 利用率。

## 6. 历史绘图与动态更新

普通 CPU 服务器：

```bash
bash plot_training.sh render \
  --run DAPO=/data/dapo/metrics.jsonl --run NCBR=/data/ncbr/metrics.jsonl \
  --output /data/plots
bash plot_training.sh watch --run NCBR=/data/ncbr/metrics.jsonl \
  --output /data/live-plots --interval 30
```

HPC4 使用 CPU 分配，无 GPU 申请：

```bash
export PLOT_PYTHON=/fs1/path/env/bin/python
# 在仓库根目录运行；debug 最长30分钟，长期监控需合适时限的CPU分区。
yhrun -N 1 -p debug -n 1 -c 4 bash plot_training.sh render \
  --run DAPO=/fs1/path/dapo/metrics.jsonl --run NCBR=/fs1/path/ncbr/metrics.jsonl \
  --output /fs1/path/plots
```

同理把 `render` 改为 `watch`。也可把完整命令写入 shell 脚本，用 `yhbatch -N 1 -p debug -n 1 -c 4 --time=00:30:00 script.sh` 提交。不要在登录节点直接渲染。`bash plot_training.sh --help` 是轻量参数检查。

默认横轴训练 step；`--axis recorded-time` 为累计**已记录**的 `timing_s/step` 秒数，不含初始化、排队和漏记时间，跨记录段重置。无耗时的验证点位于当时累计值。`--smooth` 额外画10个连续 step 滑动均值；缺失点/step缺口会打断窗口，原始曲线始终保留。

浏览器打开 `index.html`：缩放、悬停、点图例选择实验/记录段。watch 每30秒读取新增字节并原子更新文件，页面按相同周期刷新；刷新会重置缩放/图例选择并恢复滚动位置。HTML 不依赖网络。复制到另一台机器的文件不会自动同步，需自己同步文件或在共享路径查看。

同 step 按字段合并，冲突保留最后值并在 audit 中记录；较小 step、文件替换/截断形成新段，曲线不会跨段连线。不完整末行暂存直到出现换行；完整非法行记录解析错误。watch 只监控明确给定的文件，启动时必须存在；暂时消失时记录事件等待恢复。停止 watch 不触碰训练。Ctrl-C/SIGTERM 正常退出会导出静态图并撤下刷新脚本；SIGKILL/调度器强杀来不及导出时重新执行 `render`。渲染耗时加在轮询周期上，不保证实时延迟精确为30秒。

截断前数据在当前 watcher 内存及输出 CSV 中保留；重启 watcher 后只读当前输入文件，无法恢复已删除日志。轮询间发生且字节完全恢复的瞬时改写无法检测；reader 用 inode、长度与末4096字节锚点识别常见替换/截断。应保留每次尝试的独立日志文件。

缺失指标画断点或不可用面板，不补零、不插值。验证点稀疏时不会画连续验证趋势。没有配置证明时一律称“未记录”，不猜测“未启用”。验证源名保持原样；确认数据来源后才提供显示映射，例如 JSON 文件 `{"math_dapo":"verified_dataset_name"}`，加 `--source-names /data/source_names.json`。此映射仅影响交互图标题，CSV保留原名，绝不默认认作 AIME2024。

结果目录：

- `index.html`：12组离线交互图，含审计折叠区；watch期间不反复导出静态图。
- `01_reward.png/.pdf` 至 `12_resources.png/.pdf`：各组静态导出，render或正常停止时生成。
- `metrics.csv`：实验、源文件、segment、step、累计记录时间、原始字段、原值、展示值、单位、状态；空展示值不是0。
- `metric_dictionary.json`：指标分组、单位、分母与口径。
- `audit.json`：输入字节数与SHA256、历史文件代次哈希、残行/解析错误/冲突、代码哈希、Git提交与软件版本、参数、导出列表。

文件逐个原子替换，最后发布审计；整个目录不是跨文件事务。需冻结可分享快照时停止 watch 后再复制。输出目录与输入目录必须不同。

## 7. 可复现与验收

保存模型身份、真实训练step、训练配置（含验证采样配置）、源码提交、logger路径、输入哈希、评测manifest、图表audit与依赖版本。分享时注明奖励口径、验证方法、预算、采样次数、统计单位，以及是否跨训练重启段。日志和CSV可能含本机路径，分享前另存脱敏副本；不要将权重、题目、轨迹或原始训练日志提交源码仓库。

CPU验收：

```bash
# HPC4 在CPU调度分配内运行：
python -m pytest -q analysis/training_plots/test_plots.py
bash plot_training.sh --help
bash evaluate.sh --help
```

测试覆盖增量/残行/冲突/坏行/回退/截断/替换、分母与原生接受率、缺失和累计时间、独立HTML与PNG/PDF导出。实际日志验收结果见[脱敏验收记录](../analysis/training_plots/VALIDATION.md)。
