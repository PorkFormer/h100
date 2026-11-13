# NCBR / DAPO experiments

- [训练动态图表与离线评测完整教程](docs/TRAINING_PLOTS_AND_EVAL_ZH.md)
- [训练复现说明](experiments/qwen3_4b_reproduction/README_ZH.md)
- [统一离线评测：配置、启动与协议](analysis/offline_eval/README_ZH.md)
- [离线评测验收记录](analysis/offline_eval/VALIDATION.md)

离线评测默认四个数据集、每题32次、8K回答预算，推荐8×A100/A800，也支持少卡和其他兼容GPU：

```bash
bash evaluate.sh run --checkpoint /data/checkpoint-hf \
  --tokenizer /data/Qwen3-4B-Base --model-id ncbr_s300 --step 300 \
  --output /data/results/ncbr_s300
```

HPC4登录节点将 `run` 换为 `submit`，可指定分区和资源；采样、推理及统计协议固定。数据集、预算可从已有固定选项中选择。

`analysis/ncbr_offline_eval/` 是保留哈希的旧环境历史评测包；新评测统一使用根目录入口。历史包保持原样，避免破坏原有证据校验。

本次新入口已完成CPU协议测试和历史轨迹回放；尚未执行新入口的GPU生成验收。
