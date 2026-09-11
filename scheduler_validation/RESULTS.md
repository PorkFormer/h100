# NCBR continuation 调度验证结果

已完成实现及规定验证。`work_conserving` 在本次固定环境中通过功能与六项训练集成验收，可作为显式选项使用；性能验收未通过，默认仍为 `fixed_wave`。

独立 checkout：`/tmp/ncbr_scheduler_20260911`，分支 `perf/continuation-scheduler`，基于 `c557509845f5aacbde9b990ba9157b66502a294b`。共享 `/workspace/rl/h100` 保持只读。实现提交 `51255ba`；失败日志计数最小修复 `b4077eb`。各实例实际执行源码哈希保留在原始证据中。

## 功能与语义

- 216 项 CPU 测试通过，包含原 190 项；另 4 项独立 runner/报告/进程所有权测试通过。先补测试再实现，首次失败日志保留。
- 两个 frozen characterization 与原 golden 字节完全一致，未修改 oracle/golden。
- 可控事件测试覆盖 FIFO 补位、跨队列批次补位、并发上限、release ACK 前不补位、乱序恢复；慢派发、生成错误、超时、取消、缺失/重复/版本错误、释放失败及重复取消中的清理也已覆盖。
- 槽位覆盖派发、生成和释放；失败后停止补充，等待派发句柄，执行 abort/drain/release 和本地收尾。释放只有单一所有者；远端清理未确认时阻止 actor 更新、发布和 sleep。
- 原 2048 条短轨迹/78 条 continuation，以及真实修正候选 batch 的 17 条 continuation，在 old C4/new C4/new C8 的 off/shadow/replace 下，请求内容与调用数、tokens/masks、raw/effective rewards、筛选顺序、actor tensors 和 advantage 精确一致。另绑定真实保留的 8 行 actor batch。关闭路径为零额外请求。
- 八卡确定性 actor 诊断中，新 C4、新 C8 分别与旧 C4 的全部 96 项 vanilla/GSPO 完整 loss inputs 和梯度记录精确一致，各变体内部重复也精确一致。真实 0→1 修正改变 4 行 advantage 和两个 loss 各 8 个梯度分片；shadow 与 off 精确一致。
- 每次 GPU 启动前执行八卡空闲/UUID/真实 CUDA 计算检查；首次八 rank NCCL all-reduce 通过。模型、依赖、seed、采样、verifier、奖励、动态筛选、loss、optimizer、actor H 前缀与生产确定性设置保持原协议。

## 真实生成计时：仅诊断，性能 FAIL

两次旧调度重复分别为 1470.42 / 1452.34 秒、196866 / 198676 tokens，2/78 请求输出不同。该失败足以阻断性能资格。首个旧重复包含首次负载均衡器惰性初始化，其计时不进入配对统计；所有配对实例均在相同预热前确认服务和负载均衡器就绪。

固定原始 78 条请求，H=2048、L=8192，每个实例重新启动八个服务并执行相同预热，六组按 A/B、B/A 交替运行。表内时间只包含 continuation，不含启动。所有配对请求内容相同，生成 tokens 存在差异，不能将下列耗时比解释为调度加速。

| 对照 | 六组耗时比均值 | 配对 95% CI | 可比工作量 | 性能验收 |
|---|---:|---|---|---|
| 旧 C4 / 新 C4 | 1.8270 | [1.7518, 1.8935] | 否 | FAIL |
| 新 C4 / 新 C8 | 1.8872 | [1.8438, 1.9376] | 否 | FAIL |

统计量为六个配对比值的算术均值；固定 seed 42，10000 次配对 bootstrap。要求工作量可比、均值 ≥1.05 且区间下界 >1；本次只满足数值阈值，未满足工作量门槛。C8 是容量变化的独立对照，不归因于调度算法。

### 旧 C4 与新 C4

| 配对 | A 秒 | B 秒 | A tokens | B tokens | 差异请求数 /78 |
|---|---:|---:|---:|---:|---:|
| 1 | 1456.67 | 824.99 | 196866 | 218586 | 31 |
| 2 | 1458.31 | 769.46 | 198505 | 195455 | 37 |
| 3 | 1463.28 | 792.01 | 198817 | 209603 | 36 |
| 4 | 1457.21 | 752.83 | 196866 | 198651 | 37 |
| 5 | 1468.80 | 881.66 | 198505 | 233026 | 33 |
| 6 | 1466.00 | 791.58 | 198505 | 208272 | 35 |

### 新 C4 与新 C8

| 配对 | A 秒 | B 秒 | A tokens | B tokens | 差异请求数 /78 |
|---|---:|---:|---:|---:|---:|
| 1 | 764.98 | 405.10 | 207617 | 198411 | 33 |
| 2 | 788.86 | 430.34 | 205954 | 207932 | 36 |
| 3 | 815.06 | 408.04 | 215281 | 195805 | 39 |
| 4 | 786.45 | 410.90 | 205138 | 199175 | 35 |
| 5 | 782.97 | 418.15 | 209069 | 206268 | 27 |
| 6 | 755.08 | 415.39 | 199570 | 207493 | 34 |

配对服务启动耗时 79.73–83.81 秒，逐实例启动、预热、continuation 和总进程耗时见 `results/timing_instances.json`。26 个真实计时实例（含旧重复）均正常结束并确认远端 drain/release。24 个配对实例的完整事件审计通过 FIFO、身份唯一、槽位上限、单次释放和全部释放检查；新 C4/C8 都观察到跨批次补位。

`evidence/event_report.json` 保存逐请求派发、生成、完成后等待 release、release、槽位总耗时及实际服务映射。`events.jsonl`/`metrics.json` 保存活动数轨迹和可补位却空闲的墙钟时间。补充汇总 `results/events.json` 的 idle slot seconds 是空槽位数量的时间积分，不是墙钟时间。旧 C4 的完成后等待 release 累计均值为 2900.44 请求秒；新 C4（12 个实例）为 0.05184、新 C8 为 0.05211 请求秒。这些是跨请求求和，不能当作墙钟节省或性能通过证据。

所有计时实例采样峰值均为每卡 14773 MiB，C8 未观察到额外峰值。采样周期约 2 秒，固定服务预分配会影响该指标；不是瞬时内存峰值或普遍的零额外显存结论。

## 四次更新训练集成

以下各实例均保留 1800 秒总预算、600 秒请求超时和四次更新目标，无缩减、补样或容差调整。每个实例八卡均完成四次非零梯度且权重改变的更新，版本 1–4 发布，版本 1–3 在下一轮实际服务。最终版本 4 的发布不冒充后续服务证据。

| 模式 | 并发 | 结果 | 总秒 | updates | 请求/release ACK | actor 前缀行 | 有效奖励修正行 | 最高单卡 MiB |
|---|---:|---|---:|---:|---|---:|---:|---:|
| DAPO shadow | 4 | PASS | 1387.28 | 4 | 99/99 | 32 | 0 | 16799 |
| DAPO replace | 4 | PASS | 1147.02 | 4 | 87/87 | 32 | 1 | 16799 |
| GSPO replace | 4 | PASS | 984.80 | 4 | 67/67 | 1024 | 3 | 22305 |
| DAPO shadow | 8 | PASS | 896.54 | 4 | 110/110 | 32 | 0 | 16799 |
| DAPO replace | 8 | PASS | 970.97 | 4 | 104/104 | 32 | 0 | 20629 |
| GSPO replace | 8 | PASS | 766.06 | 4 | 52/52 | 1024 | 1 | 22007 |

所有 actor 输入均维持 H=2048 前缀；奖励变化仅在末个有效 token，GPU UUID 集合与独立源码来源核验通过。C8 DAPO replace 的这一次实时运行未出现有效奖励修正；真实修正影响由冻结回放/梯度诊断独立覆盖。训练实例不构成配对性能实验，显存和耗时差异不能直接归因于并发。

## 首次失败、边界与交付

- 原始失败保留：`evidence/tests_before.log`、`evidence/release_receipt_before.log`。后者发现失败日志遗漏已由槽位所有者观察到的 release 错误数量；屏障原本已 fail closed。仅修复日志聚合计数，并重跑 216 测试与两个冻结特征，见 `audit_fix_manifest.json`。
- 首次 CUDA gate 因瞬时 4 MiB 占用拒绝启动，未重置或终止外部进程；待空闲后重做，原始 gate 保留。
- 首次旧重复 token 差异见 `evidence/old_repeat_first_difference.json`，后续所有原始 token 和差异均保留，未用不同工作量宣称加速。
- 超时：本次规定的 36 个 GPU 实例均未超时；六项训练均 PASS。远端异常注入由 CPU 可控故障测试覆盖，未新增真实 GPU 故障注入或硬件 reset 测试，也未验证四步之后的长期训练。
- 后加的进程所有权收尾 receipt 覆盖全部 24 个配对和六个训练实例；此前 NCCL、三个 actor 诊断和两次旧重复没有该格式的 receipt，依赖正常退出、下一次空闲 gate 和最终全卡空闲证据。不能将本地进程退出替代远端清理 ACK。
- 结束时八卡 0 MiB / 0% 占用、Recovery Action None，无计算进程。最终哈希复核通过 28 个冻结输入、10 个模型文件和所有记录依赖版本。

明确区分：**功能 PASS；本环境六项集成 PASS；性能 FAIL（DIAGNOSTIC_ONLY）；长期训练/真实故障注入未覆盖。** 默认保持 `fixed_wave`，不做生产默认切换。

启用配置见 `README.md`：`ncbr.enable=true`，`boundary_return.mode=shadow` 或 `replace`，`scheduler=work_conserving`，`max_concurrent_requests=4` 或单独验证过的 `8`，`request_batch_size=8`，`request_timeout_seconds=600`。

原始证据在本 checkout 的 `scheduler_validation/evidence/`，提交的分项摘要在 `results/`。`manifest.json` 绑定协议和输入；`source_manifest.json`/`audit_fix_manifest.json` 绑定实现及最小修复。全部写入结束后由 `finalize.py` 生成 `evidence/final_manifest.json`，对应最终绑定摘要在 `results/evidence_binding.json`。原始证据未删除、未覆盖首次失败。
