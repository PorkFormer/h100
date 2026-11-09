# 交付验证记录

验证日期：2026-11-09。源码基线：`b1835ea0fb388eb6e4d29a58ab18b408cae7b34d`。

## 已验证

- 两套配置通过当前框架的 dataclass、拓扑及 NCBR preflight 检查；保持 DAPO context=3072、NCBR context=9216；正式步数均为 300。
- 历史 DAPO 的旧 reward_model 参数先迁移到 reward，再冻结有效配置；buffer=410、penalty=1、max_resp_len=2048 均保留。
- 当前真实环境的 224 个依赖版本与锁定清单一致；真实训练入口、vLLM、FlashAttention 导入通过。两个已知 metadata 冲突保留并显式记录。
- 16 个原始资产文件的 SHA256、三份数据行数和字段结构通过；原始 Base tokenizer 能应用原始训练样本的 chat template。
- checkpoint 验证器已读取历史 DAPO step-300 的真实 16 卡文件结构：49 个权重/optimizer/extra/HF 文件，HF 权重 398 个 tensor。此项验证的是检查器及历史文件，不是新训练结果；PyTorch 分片仅检查存在性/大小，HF safetensors 检查可读 header，不进行 optimizer 恢复或完整模型推理。
- 新启动器覆盖配置漂移、错误拓扑、资产缺失/损坏、输出重复、依赖缺失、GPU 忙碌/数量错误、非有限 loss、未完成步数、NCBR 未发生续写等错误路径。
- 完整 CPU 测试包含现有 218 项 NCBR/DAPO 测试及本交付的 29 项测试；两份冻结行为参考文件逐字节比较。最终结果见 `provenance/cpu_validation.json`。
- 搬运包在独立解压目录验证配置、真实环境导入、源码哈希及完整 CPU 回归；测试禁用 GPU，并禁止 Ray 初始化。

## 尚未验证

- 指定容器基础环境为 `docker.1ms.run/verlai/verl:vllm018.dev1`；本次文档更新没有拉取或运行该镜像，没有记录已验证的镜像 digest。目标机拉取后需保存 image inspect 结果，并运行容器内的环境检查。
- 没有在新机器重新联网安装全部依赖；下载可用性、CUDA 编译和目标驱动 ABI 由安装脚本在目标机验证。
- 没有在目标单机 8×80GB 上执行冒烟或正式训练，没有新 checkpoint、GPU 峰值显存、吞吐或分数结果。
- CPU 测试包含续写尾部不进入 actor、边界奖励、动态采样和关闭 NCBR 的行为检查，但不能证明不同 GPU 拓扑的训练轨迹或最终分数逐位相同。

目标机验收顺序：`config all` → `check all` → `smoke all` → `train all`。冒烟不改变 batch/n/H/L；正式训练不读取冒烟或历史 checkpoint。`status.json` 的 complete 需要实际完成目标步数、验证及预期 checkpoint 检查。

## 独立离线评测增补

完整 CPU 回归现为265项通过（原247项加18项离线评测测试），两份 frozen golden 仍逐字节一致。100道历史题目的语义哈希及3200个采样种子已逐一核对；源码和 Base tokenizer 的 CPU preflight 通过。测试覆盖固定分片顺序、采样参数、结果身份/长度/文本校验、完整6400条合成结果重评分、缺失结果拒绝、Avg@32与pass@32计算及同step比较限制。合成结果只验证脚本逻辑，不是模型成绩。记录见 `provenance/offline_eval_validation.json`。未执行GPU离线生成或新机器容器测试。
