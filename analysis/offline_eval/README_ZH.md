# 一键离线评测

仓库根目录入口为 `bash evaluate.sh`。输入一个完整 HF checkpoint，即可完成固定数据准备、权重检查、独立生成、评分和中文报告。适用于当前 Qwen3-4B/8B 的 DAPO、NCBR 或其他兼容 checkpoint；评测只进行普通自回归采样，不调用训练时的 NCBR 续写奖励机制。

默认评测 **AIME2024 30题、AIME2025 30题、AMC23 40题、MATH500 500题**，每题32次、H8192，共19,200条回答。默认使用至多8张可见GPU，推荐8×A100/A800；其他支持冻结推理配置的型号、不同显存容量和少卡机器均可使用。这里没有型号、80GB、固定节点或64核门槛。

## 环境和模型

历史推理环境为 Python 3.12、vLLM 0.18.0、torch 2.10.0+cu129、transformers 5.3.0、numpy 1.26.4、pyarrow 23.0.1、safetensors 0.7.0。使用项目已有 `verl-vllm018-dev1.sif`，或按 `experiments/qwen3_4b_reproduction/ENVIRONMENT.md` 搭建环境。镜像标签不能代替实际版本检查：入口会检查实际软件版本。不自动安装或修改共享环境，无需 Ray 或 W&B。

指定 Python：`--python /path/to/python`。使用 Apptainer：`--container /path/to/image.sif --apptainer /path/to/apptainer --bind /data:/data`；容器模式的 `--python` 是容器内路径。`--squashfuse-bin /path/to/bin` 可配置宿主机所需工具。源码目录始终按当前位置定位，没有 `/workspace` 或旧机器 IP 依赖。

权重必须是包含 `config.json` 和完整 safetensors 的 HF 目录，不根据目录名猜测 step。使用 `--tokenizer` 指定历史 Base tokenizer，四份tokenizer资产哈希固定在 `protocol.lock.json` 中；可使用本项目已有的 Qwen3-4B-Base 或字节一致的8B tokenizer。若checkpoint本身具有这些冻结资产，可以省略该参数。模型结构从被测checkpoint验证，不拿4B模型维度限制8B。

只有FSDP分片时，先用同版本verl原生工具导出到**新目录**：

```bash
PYTHONPATH=/path/to/h100/verl python -m verl.model_merger merge \
  --backend fsdp --local_dir /path/to/actor --target_dir /path/to/new-hf
```

导出和完整权重检查可能使用较多CPU/内存；HPC4须在计算节点进行。

## 一键启动

普通GPU服务器，在仓库根目录执行：

```bash
bash evaluate.sh run \
  --checkpoint /data/ncbr/step300-hf --tokenizer /data/Qwen3-4B-Base \
  --model-id ncbr_s300 --step 300 --output /data/results/ncbr_s300
```

源数据缺失时，入口从公共 `Ottoman6324/rlvr` 的固定提交 `9d19f9b64ac9da472281617f0818a532185fe86a` 下载四个小型文件并检查SHA256，不下载模型或训练集。默认缓存位于 `~/.cache/ncbr-offline-eval/data`，可用 `--data-dir` 修改。需要代理时在启动前 `source ~/proxy_on.sh`；脚本不主动修改用户代理。

少卡及数据集/预算选择示例：

```bash
bash evaluate.sh run \
  --checkpoint /data/dapo/step300-hf --tokenizer /data/Qwen3-4B-Base \
  --model-id dapo_s300 --step 300 --output /data/results/dapo_selected \
  --gpus 2 --datasets AIME2024 MATH500 --horizons 2048 8192 16384
```

`--gpus`是期望并发卡数，实际取它、可见卡数和8的最小值。`--devices 0 2`表示**当前CUDA可见范围内**的逻辑设备0和2，不是宿主机物理卡号。调度器提供的 `CUDA_VISIBLE_DEVICES` 可以是数字或UUID；入口依据实际可见设备身份映射，并在worker启动vLLM前再次核对UUID。映射不一致时停止，不能访问分配外设备。

物理卡减少时，固定8个逻辑分片按波次执行，保持每个请求的种子、顺序与每批8条不变。其他硬件或驱动可能产生数值差异，不承诺逐token完全一致。显存不足时明确失败，不自动缩短回答、减少采样或改变dtype。

## HPC4

从登录节点一键提交：

```bash
bash evaluate.sh submit \
  --checkpoint /fs1/path/to/step300-hf --tokenizer /fs1/path/to/Qwen3-4B-Base \
  --model-id ncbr_s300 --step 300 --output /fs1/path/to/results \
  --partition gpu1 --gpus 8 --cpus-per-gpu 8 --mem 128G --time 12:00:00
```

登录节点只下载并校验小型原始输入、生成脚本和提交作业；完整权重检查、分词、生成与评分在分配内运行。计算节点使用 `--offline`，不会联网。单节点执行，未绑定具体节点或A100/A800型号。`gpu`、`gpu1`遵守各自CPU/GPU申请上限；其他分区由用户指定。`--dry-run`仅准备提交脚本，不提交；使用新输出路径避免重复准备。

HPC4预设只提供当前容器、Apptainer和squashfuse路径，可用上述参数覆盖。`EVAL_MANAGEMENT_PYTHON`指定登录节点的Python（需3.10+）；计算环境Python独立。缺少FUSE或容器工具时保留实际错误，选择具备容器条件的节点或使用已安装的Python环境，不以解包大镜像作为隐式兜底。

提交回执和日志位于 `<output>.submission/`，任务结果位于 `<output>/`。`SUBMITTED`只表示已提交，不表示运行或完成：

```bash
bash evaluate.sh status --output /fs1/path/to/results
```

状态包含阶段及各分片已生成/预期数量。CPU准备、检查或重评分可通过 `yhrun -N 1 -p debug -n 1 -c 4 ...` 在CPU分配内执行；本项目不在登录节点运行这些操作。

## 协议与可配置边界

协议定义在 `protocol.py`，资产清单在 `protocol.lock.json`。命令行不提供以下设置的覆盖选项；检测到对应 `EVAL_*` 环境覆盖时直接报出冲突。

| 项目 | 冻结值 |
|---|---|
| 采样 | 每题32次，temperature=1，top_p=1，top_k=-1，repetition_penalty=1，正常EOS |
| 请求种子 | 历史 `stable_sampling_seed(42,prompt_id,rollout_index)`；不受卡数和数据集子集影响 |
| 回答档位 | 2048/8192/16384分别独立生成；默认8192；上下文4096/9216/17408 |
| 输入 | ≤1024 token；固定原始文件、题序、ID、模板、答案、tokenizer和token IDs |
| 引擎 | vLLM0.18.0，BF16，TP1，seed42，显存比例0.72，max_num_seqs8，batched tokens32768 |
| 调度 | 8个逻辑分片，每批8请求，prefix caching/chunked prefill开启 |
| 评分 | 历史 `math_dapo.compute_score`，按最后300字符内的Answer规则判分 |
| 统计 | 按题目bootstrap10000次，seed20260812，95%区间 |

只开放物理卡数、可见设备子集、路径、CPU/内存/分区/时限等运行资源。TP、批次、显存比例、dtype等也按本次约定严格冻结。选择2K/8K/16K属于选择已有协议档位，不能任意填写新长度。高预算结果不得截取前缀冒充低预算评测。

AIME2024原始文件960行按首次出现顺序去重为30题；原三套题ID保留512–611，MATH500保留1000–1499。所有数据集都核对独立语义哈希，选择子集不会重新编号。AMC23模板来自固定AIME输入，MATH500直接使用已发布的verl消息，保留各自换行形式。

MATH500继续使用历史字符串归一化评分器，并非符号等价判分；不能在同一协议下悄悄切换math-verify。

推理明确使用vLLM默认生成配置和本协议的显式采样参数，不继承checkpoint里额外的stop、penalty或其他采样默认值；模型EOS须与历史Base协议一致（151643）。checkpoint的generation_config也记录哈希，但不能覆盖本评测配置。

## 完成、重评分与比较

所有分片正常退出、权重身份不变、预期轨迹完整且评分成功后才写入 `COMPLETE`。任何缺失、重复、错误种子、非法结束原因、文本/token不一致或评分异常都会使任务失败；评分异常不计为答错。

主要结果：`manifest.json`、`prompts.json`、`h*/shard*.jsonl`及日志/进度、`generation_complete.json`、`resource_release.json`、`driver_exit.json`、`trajectory_scores.parquet`、`metrics.csv`、`metrics.parquet`、`scoring_receipt.json`、`REPORT_ZH.md`。

报告逐数据集、逐预算列出Avg@32/pass@1估计、pass@32、95%区间、长度与截断率，不混合成一个总分。manifest记录权重、tokenizer、协议、执行代码、GPU及软件身份。

完整生成可以独立重评分，无需权重或GPU：

```bash
bash evaluate.sh score --output /data/results/ncbr_s300
bash evaluate.sh compare --left /data/results/dapo_s300 \
  --right /data/results/ncbr_s300 --output /data/results/comparison
```

重评分前会撤下旧的成功回执和公开派生报告，并归档到 `scoring_attempts/`；失败后不能误读旧分数。比较只接受状态成功、原始文件及报告哈希未变、step/数据集/协议/预算一致的结果，给出“右侧−左侧”逐题配对bootstrap差值。不同硬件的执行身份仍保留在各自manifest中。

原始生成目录不可覆盖，不自动续跑失败分片或调整参数。生成失败后检查日志，使用新输出目录；评分失败且原始生成完整时使用 `score`。进程清理仅针对本次启动的进程组，不使用全局Ray停止或终止其他GPU任务。

旧 `experiments/qwen3_4b_reproduction/evaluate.sh` 仍支持 `run dapo/ncbr/all` 等命令，但委托同一套实现；为兼容旧调用，其默认仍为原三个数据集和2K+8K。`EVAL_GPUS`改为可见范围内逻辑序号，留空自动选择。新格式结果的比较目录为 `<EVAL_OUTPUT>/comparison/`。旧格式历史结果使用 `replay.py`审计，不直接伪装成新生成。独立源码包内，新统一入口名为 `offline_evaluate.sh`，原 `evaluate.sh` 保留兼容接口。

## 验收与历史修复

本次只执行CPU测试和历史轨迹回放，**没有执行新入口的GPU生成验收**。实际通过项见 `VALIDATION.md`。

```bash
python -m pytest -q analysis/offline_eval/test_eval.py
python analysis/offline_eval/replay.py --help
```

回归覆盖：UUID被vLLM当作整数解析、4B/8B架构误限制、rope配置序列化差异、固定数据量导致扩展集漏跑、部分任务失败仍退出成功、重复启动、残缺/重复轨迹、评分异常、旧成功报告残留、固定种子及数据子集稳定性。

源码仓库不包含题目正文、权重、生成轨迹或运行日志。历史轨迹回放使用外部本地资产，只发布不含正文的验收摘要。
