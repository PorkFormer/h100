# 独立离线评测：AIME24、AIME25、AMC23

本入口用于训练完成后的独立评测，与训练中的 greedy 验证分开。默认比较 **DAPO step300 和 NCBR step300**；也可手动指定其他相同步数的完整 checkpoint。不会自动挑选最好 checkpoint，不依赖训练进程、Ray 集群、W&B 登录或在线模型服务。

## 1. 环境与路径

使用 [ENVIRONMENT.md](ENVIRONMENT.md) 的容器 `docker.1ms.run/verlai/verl:vllm018.dev1` 和已安装的固定 Python 环境。仍需先运行 `setup_env.sh`；容器标签本身不能代替依赖版本核验。历史环境为 Python3.12、vLLM0.18.0、torch2.10.0+cu129、transformers5.3.0。默认单机8卡，每个进程 TP=1。训练结束并释放显存后再评测。

在**新机器自行下载并填写实际路径**，容器中填写挂载后的路径。示例 `/data` 不是强制目录。源码定位规则与训练一致：独立包用包内 `verl/`，仓库用 `../../verl`，可手动设置 `VERL_ROOT`。

| 配置文件 | 变量 | 指向内容 |
|---|---|---|
| settings.env | MODEL_PATH | 固定版本 Qwen3-4B-Base 目录，始终用其 tokenizer |
| settings.env | VAL_FILE_1 / VAL_FILE_2 | 原始 aime-2024-verl.parquet / aime-2025-verl.parquet |
| settings.env | PYTHON_BIN / VERL_ROOT | 已搭建环境的 Python / 含 setup.py 的 verl 源码根目录 |
| eval_settings.env | AMC23_FILE | 手动下载的固定版本 AMC23 原始 Parquet |
| eval_settings.env | EVAL_DATA | 准备脚本将生成的 prompts.json，首次必须不存在 |
| eval_settings.env | DAPO_HF / NCBR_HF | 各自完整 HF safetensors checkpoint 目录 |
| eval_settings.env | EVAL_OUTPUT | 本次评测输出根目录；已有 dapo/ncbr 子目录拒绝覆盖 |
| eval_settings.env | EVAL_STEP | 两个 checkpoint 的实际训练步数，默认300，必须手动核实 |
| eval_settings.env | EVAL_GPUS | 本进程可用的物理 GPU 编号或 UUID，默认0到7 |

`SETTINGS_FILE` 和 `EVAL_SETTINGS_FILE` 可以选择配置文件副本。生成过程不会下载模型或数据。纯重评分只需原始 Base tokenizer、已保存的生成结果和同版本 verifier，无需 GPU 或 checkpoint 权重。

## 2. 手动下载与准备数据

Base 下载链接：https://huggingface.co/Qwen/Qwen3-4B-Base 。固定 revision、AIME 原始文件下载说明见 [PATHS_AND_DOWNLOAD_ZH.md](PATHS_AND_DOWNLOAD_ZH.md)。AIME 原始预处理文件尚无已核实的公开 URL，需自行取得原始文件；不能直接换另一套 prompt 模板。

AMC23 数据集：https://huggingface.co/datasets/math-ai/amc23 。在新机器手动执行，目标目录按实际情况修改：

```bash
AMC_DOWNLOAD="/data/amc23-test.parquet"
mkdir -p "$(dirname "$AMC_DOWNLOAD")"
curl --fail --location --retry 3 \
  'https://huggingface.co/datasets/math-ai/amc23/resolve/80815d37005feb82cd7f8fbc6901d5d3eff43057/test-00000-of-00001.parquet' \
  --output "$AMC_DOWNLOAD"
echo "b696e87ba47be4e879a60fd4ef1d4aa522ba78c5ae013c6aff5cc9788a397c5e  $AMC_DOWNLOAD" | sha256sum -c -
```

将此路径写入 `eval_settings.env` 的 `AMC23_FILE`。准备命令在包目录运行：

```bash
bash evaluate.sh prepare-data
```

准备时 AIME2024 的960行按原始消息首次出现顺序去重为30题，AIME2025为30题，AMC23保留固定40题顺序。**不会改动训练/验证 Parquet**。AMC prompt 包装逐字沿用历史 AIME 模板，包括模板里的字面量 `\n`；无额外 system/thinking 提示。题目 ID 为512–611。最终对100题的题目、答案、顺序、渲染文本及 token IDs 做整体语义哈希核验，不要求重新写出的 Parquet 字节一致。哈希清单在 `provenance/offline_eval.json`，源码包不携带题目或答案。

## 3. 选定训练 checkpoint

本训练启动器保存的 `checkpoints/global_step_300/actor/huggingface/` 若已包含完整 `model*.safetensors`，直接把该目录填入 `DAPO_HF` / `NCBR_HF`。不要填 Base 模型路径、global_step 父目录或只有 config/tokenizer 的目录。

若使用其他兼容训练入口，只保存了 FSDP actor 分片，则在相同 Python 环境、相同源码下先手动导出。以下路径都要替换为新机器实际路径，DAPO和NCBR分别执行：

```bash
# PYTHON_BIN、VERL_ROOT 与 settings.env 一致；训练停止且 checkpoint 写入完成后执行。
PYTHON_BIN="/path/to/reproduction/.venv/bin/python"
VERL_ROOT="/path/to/reproduction/verl"
ACTOR_DIR="/data/runs/dapo/checkpoints/global_step_300/actor"
HF_DIR="/data/checkpoints/dapo-step300-hf"  # 选择新的目录
PYTHONPATH="$VERL_ROOT" "$PYTHON_BIN" -m verl.model_merger merge \
  --backend fsdp --local_dir "$ACTOR_DIR" --target_dir "$HF_DIR"
```

导出是 CPU 内存密集操作，应预留足够 RAM 和额外约8–16GB/模型的权重存储（取决于导出dtype；本训练FP32导出约16GB）。评测检查架构、safetensors 可读性、索引覆盖并记录权重 SHA256；不从目录名字推断真实 step，请自己确认 `EVAL_STEP`。

## 4. 一键运行与重评分

先编辑两份 env 配置，再运行：

```bash
bash evaluate.sh check all     # CPU 检查数据、源码、tokenizer、两份 HF 权重
bash evaluate.sh run all       # DAPO 两个长度 → NCBR 两个长度 → 对比报告
```

分别运行也可以：

```bash
bash evaluate.sh run dapo
bash evaluate.sh run ncbr
bash evaluate.sh compare
```

命令前台运行，退出码非零表示失败。每个模型、每个 horizon、每个 shard 有独立日志和 JSONL。默认8个独立 vLLM 进程各占1卡，2K结束释放进程后再启动8K；无需启动 Ray。少于8卡时只修改 `EVAL_GPUS`，脚本按波次执行固定的8个逻辑 shard，保持请求顺序和每批8条；硬件/调度变化仍可能导致数值差异，严格对齐推荐8卡及同型GPU。禁止与训练抢占同一批GPU。

失败时查看 `EVAL_OUTPUT/{dapo,ncbr}/status.json` 与 `h*/shard*.log`，修复后换新的 `EVAL_OUTPUT` 重跑。脚本不自动续跑或覆盖已有生成结果。正式生成不做缩小数据集的 smoke，避免把部分结果混成正式评测。

已有完整生成结果可以独立 CPU 重评分，不重新生成：

```bash
bash evaluate.sh score all
bash evaluate.sh compare
```

重评分覆盖派生的 `scores.json` / `metrics.csv`，保留原始 JSONL。遇到缺失、重复、错误种子、错误长度、文本/token不一致或 verifier 异常直接失败，不把异常算成答错；失败后不要使用先前留下的派生报告，修复并成功重评分后再比较。

## 5. 与历史离线评测对齐的协议

| 项目 | 固定设置 |
|---|---|
| 数据集 | AIME2024 30题、AIME2025 30题、AMC23 40题，分别报告 |
| 采样 | 每题32次，temperature=1、top_p=1、top_k=-1、repetition_penalty=1、ignore_eos=false |
| 长度 | 独立 H=2048 / H=8192；上下文4096 / 9216；输入≤1024 |
| 引擎 | BF16、TP1、seed42、显存利用率0.72、max_num_seqs8、max_num_batched_tokens32768、prefix caching/chunked prefill开启 |
| 请求分片 | `(prompt_order*32+rollout_index)%8`；每 shard 400条，固定批次8条 |
| 请求种子 | SHA256(`offline-answer-timing-v1\0{42}\0{prompt_id}\0{rollout_index}`) 前4字节 little-endian，再与0x7fffffff；模型/长度间使用相同种子 |
| 判分 | 原始 Base tokenizer 解码，skip_special_tokens=True；完整回答送入历史 `math_dapo.compute_score` 默认参数，取 `acc` |
| 总量 | 每个checkpoint每个长度3200条，共6400条；两个checkpoint共12800条 |

2K必须独立生成，不能将8K输出截断为2K充当正式结果。评测是普通采样，不启用训练时 NCBR boundary-return 机制；DAPO和NCBR使用完全相同推理预算。这里的上下文设置是历史**离线**协议，不是训练 rollout 的上下文设置。

每个模型输出6行数据集×长度指标：

- **Avg@32 / pass@1估计**：每题32次正确率，然后对题目平均。
- **pass@32**：每题32次至少一次正确的比例；不能与Avg@32混用。
- 平均、P50、P90输出token数；截断率（finish_reason=length或token数达到H）。
- 95% CI：以**题目**为重采样单位，10000次 bootstrap，seed=20260812，2.5%/97.5%分位数。

`comparison.json` / `REPORT_ZH.md` 给出同一步数的 **NCBR−DAPO** 差值（百分点）及逐题配对bootstrap CI。三套题库分别报告，不混为一个总分；这是固定训练种子的评测，不代表多训练种子置信区间。

## 6. 输出和验证边界

`EVAL_OUTPUT/dapo/`、`ncbr/` 各含 `manifest.json`（权重哈希、step、GPU与协议）、`prompts.json`、`status.json`、`h2048/`、`h8192/`、`scores.json`、`metrics.csv`。根目录有 `comparison.json` 和 `REPORT_ZH.md`。输出包含题目和生成文本，放在自己选择的数据目录，不提交到源码仓库。

本次交付已在当前机器用历史数据验证100题语义哈希、3200个固定采样种子，并运行CPU协议/覆盖/重评分测试。尚未在新机器容器上执行GPU生成，不能将脚本交付视为已获得复现分数。环境、训练与独立离线评测使用相同固定源码；GPU运行前按环境文档完成依赖检查。
