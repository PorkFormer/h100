# NCBR 离线评测：手动启动

入口：`/workspace/rl/h100/analysis/ncbr_offline_eval/evaluate.py`。
适用于使用相同 Qwen3-4B-Base tokenizer 的完整 HF checkpoint。每次显式指定一个 checkpoint、模型标识和训练步数，生成一个独立结果目录。不会从路径猜参数，也不会自动寻找其他 checkpoint。

## 一条命令启动

先确认训练已正常结束、HF 权重保存完成。将下面的 checkpoint 路径替换为实际路径，output 必须是尚不存在的新目录：

```bash
python /workspace/rl/h100/analysis/ncbr_offline_eval/evaluate.py run \
  --checkpoint /绝对路径/global_step_300/actor/huggingface \
  --model-id my_ncbr_s300 \
  --step 300 \
  --label 'NCBR H1024 L4096' \
  --output /workspace/rl/h100/analysis/my_ncbr_s300_eval
```

命令先在 CPU 上校验 HF 配置、safetensors 分片索引/可读性并计算全部文件 SHA256，然后后台启动。即使退出终端，任务仍运行。此手动入口不等待训练 completion.json；选择已完成且不再写入的权重由启动者负责。模型载入时仍可能发现实际张量与架构不匹配，相关失败会保留日志。

默认硬绑定 `10.8.191.127` 八卡，连接现有 Ray `10.8.191.127:6397`。如需其他节点，可显式传 `--node IP --ray-address HEAD_IP:PORT`；目标必须是该集群中可预留八卡的节点。脚本不会创建、重启或改配置共享 Ray。当前 r5 已有独立自动等待任务，通常无需为相同权重另开一次手动评测。

## 先准备，再启动

把上述 `run` 换为 `prepare` 即只准备目录和权重清单，不连接 Ray、不启动 GPU：

```bash
python /workspace/rl/h100/analysis/ncbr_offline_eval/evaluate.py start \
  --output /workspace/rl/h100/analysis/my_ncbr_s300_eval

python /workspace/rl/h100/analysis/ncbr_offline_eval/evaluate.py status \
  --output /workspace/rl/h100/analysis/my_ncbr_s300_eval
```

`run`、`start` 可加 `--foreground`，等待后台任务结束并返回真实退出码；中断前台等待不会停止后台任务。`start` 只接受未启动的 PREPARED 目录，防止重复进程与覆盖旧结果。不同结果目录之间不会按 checkpoint 去重；请避免为同一权重重复提交。

## 固定评测方法

| 项目 | 配置 |
|---|---|
| 题目 | AIME2024 30、AIME2025 30、AMC23 40；复用冻结题序、prompt token IDs、答案 |
| 采样 | 每题 32 次；temperature=1、top_p=1、top_k=-1、repetition_penalty=1，正常 EOS |
| 种子 | `stable_sampling_seed(42, prompt_id, rollout_index)` |
| 回答预算 | 独立 H2048、H8192；各 3200 条，共 6400 条 |
| 推理 | vLLM 0.18.0、BF16、8 个 TP1 worker |
| 分片与批次 | `(题序 × 32 + rollout_index) % 8`；每分片 400 请求，按原顺序每批 8 条 |
| 上下文 | H2048 为 4096；H8192 为 9216 |
| 引擎 | 显存比例 0.72、max_num_seqs=8、max_num_batched_tokens=32768；prefix caching、chunked prefill 开启 |
| 评分 | 冻结 `math_dapo.compute_score`，按历史默认参数调用；不是单独的 boxed 答案判定 |
| 区间 | 按题目 bootstrap 10000 次，seed=20260812，95% CI |

训练回答预算不改变这两个离线预算；H2048 必须重新独立生成，不能截取 H8192 前缀代替。

pass@1（Avg@32）是先计算每题 32 次的正确率，再对题目取均值。pass@32 是 32 次中至少一次正确的题目比例。两个准确率均提供按题目重采样的区间。平均长度按生成 token 数计算，截断率按 `finish_reason=length` 或 token 数达到 horizon 计算。每个数据集、每个 horizon 单独报告，共六行；不混合数据集、不挑最佳 checkpoint、不生成历史对照、不发邮件。

## 运行和失败处理

- 启动前核验冻结代码、输入和权重；节点有占用则等待。获得八卡 Ray 预留后再次核对进程占用和 GPU UUID。
- 每个结果目录拥有独立 Ray namespace、短缓存路径、端口配置和日志。无强制 smoke、持续 120 秒空闲门、重复 NCCL 探针或心跳超时终止。
- 每五分钟记录生成数量、耗时和 GPU 状态。状态依次为 PREPARED、VERIFY_CHECKPOINT、WAIT_NODE、WAIT_RESERVATION、GENERATING、SCORING、COMPLETE；异常为 FAILED。
- 一个失败分片最多三次完整尝试；每次用新目录、相同参数/顺序/批界从头生成。只合并各分片一份成功结果，保留失败输出。
- 评分与生成分离。优先重新评分，最多三个评分进程尝试，不重新生成已完成轨迹。初始评分异常明确记录，不能填成答错。
- 最终验证每个 horizon 3200 条、每题 rollout_index=0–31、无重复/缺失，题目/种子/权重身份一致，并独立复算全部评分。
- 控制器整体失败后不会自动重启。保留原目录，排查日志后再决定恢复方式；不要对失败目录再次执行 start，也不要手动删除失败证据。

## 结果文件

- `formal/REPORT_ZH.md`：中文报告。
- `formal/metrics.csv`、`formal/metrics.parquet`：六单元指标。
- `formal/trajectory_scores.parquet`、`formal/scoring_receipt.json`、`formal/scoring_audit_*.json`：逐条评分与复核。
- `checkpoint_manifest.json`、`generation_config.json`、`run_config.json`、`freeze.json`、`SHA256SUMS.json`：身份、配置与文件哈希。
- `status.json`、`driver_exit.json`：计算状态和控制器真实退出码。
- `gpu_release.json`、`resource_release.json`：GPU 进程释放、Ray 预留移除。完成须同时核对生成、评分、正常退出及资源释放。

环境使用当前工作区已有 Python/vLLM/Ray 依赖；需要能访问共享路径、现有 Ray、nvidia-smi 和目标 GPU。`prepare` 不需要 GPU；正式启动需在具有这些权限的主机环境中执行。

## 仓库发布范围与本地数据依赖

远程仓库只包含启动器、运行模板、方法说明、配置、哈希元数据和 CPU 测试，不上传题目正文、答案、prompt token IDs、种子表、生成轨迹或运行日志。

运行时从 `input_provenance.json` 指定的本地已冻结目录 `analysis/qwen3_4b_ncbr_l4096_aligned_eval_20260913` 读取 `prompts.parquet`、`prompt_manifest.json`、`seeds.json`，先验证 SHA256，再复制到新结果目录。还需该文件列出的本地 tokenizer 和历史评分器。因此仅克隆仓库并不足以在另一台机器上直接运行；必须具备这些相同版本、相同路径的本地依赖。
