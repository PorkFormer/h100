# 离线评测入口

推荐使用仓库根目录 `bash evaluate.sh`，详见 [统一离线评测说明](../../analysis/offline_eval/README_ZH.md)。默认四个数据集、H8192，支持普通单机及HPC4。

本目录的 `evaluate.sh` 是兼容入口，读取原有 `settings.env` 和 `eval_settings.env`，仍支持：

```bash
bash evaluate.sh prepare-data
bash evaluate.sh check all
bash evaluate.sh run all
bash evaluate.sh score all
bash evaluate.sh compare
```

它调用同一套生成和评分实现；为保留旧调用语义，仍默认AIME2024、AIME2025、AMC23及独立2K/8K，不会额外启动MATH500。`EVAL_GPUS`留空自动使用至多8张可见卡，非空时填写可见范围内的逻辑序号，不再接受物理卡号或UUID列表。

数据源已核实为公共 `Ottoman6324/rlvr`；新入口会下载固定版本，旧入口继续接受手动填写的文件路径。输入必须与冻结SHA256一致。完整HF权重路径和真实step需显式填写。评分不需要模型权重或GPU。HPC4计算操作须在调度分配内执行。

生成结果使用统一格式，比较报告写入 `<EVAL_OUTPUT>/comparison/`。旧格式结果可用统一模块的 `replay.py`作历史回放，不直接以新入口重评分。新的源码打包会附带共享 `offline_eval/` 模块；不复用已发布旧压缩包中的评测实现。

本次没有执行新入口GPU生成；测试和历史回放证据见统一模块的 `VALIDATION.md`。
