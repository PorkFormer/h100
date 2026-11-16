# 训练动态图表 CPU 验收

2026-11-13；HPC4 CPU `debug` 分区，单节点、请求4 CPU / 16 GB / 30分钟，未申请GPU。首轮作业3885177与最终复验3885202均为 `COMPLETED / 0:0`。未启动训练或模型评测。

## 自动测试

```bash
python -m pytest -q -p no:cacheprovider analysis/training_plots/test_plots.py
```

最终结果：**5 passed in 5.48s**。覆盖：

- JSONL增量、末行暂存与补全、同step字段合并/冲突、完整坏行报告；
- step回退、文件截断、inode替换、同长度改写的分段；
- 续写/解锁条件比例的分母、原生动态接受率、采集状态不明的后端零值与负哨兵值；
- 未记录值、缺失step断点、分段累计记录耗时、10步滑动均值；
- 内嵌Plotly的HTML、CSV、字典、审计、12组PNG与PDF格式检查；
- `timing_s/.../response_length` 的单位为tokens而不是秒。

另对实际 `bash plot_training.sh watch` 进程完成集成验收：创建日志→读取首条→追加半行→审计显示pending bytes→补全→读取第二条→SIGTERM→退出0→12组静态导出→HTML移除自动刷新。此过程使用合成数据，不触碰训练。

## 现有日志渲染

对两份外部现有日志同时运行 `render --smooth`，不提交日志本身：

| 标签 | 合并后记录数 | 记录段 | 末行残余 | 输入SHA256 |
|---|---:|---:|---:|---|
| DAPO | 100 | 1 | 0 | `40bc63cec73a08e7f53cad3a2caa3da685b8cebdca8019c88b7a5c2689ea1b1c` |
| NCBR | 101 | 1 | 0 | `aeda464efdeaada5578fcf2f09af467ae07d53cee9c4fbb302fab72035ab0026` |

记录数不是optimizer更新次数，不据此推断其他训练尝试的终点。两份输入无解析/冲突/重启事件。导出159个字段、31,959行标准化指标、12张PNG、12份PDF、1份自包含HTML及CSV/指标字典/输入审计。依赖版本：Plotly6.3.1、Matplotlib3.10.7；独立环境Python3.13.11。

实际检查了策略变化与续写收益静态图，奖励、原生比例、分母不可用断点保留原始含义。HTML无外部script引用，保留缩放/悬停/图例功能的Plotly配置；未执行自动化浏览器交互测试。

命令检查：`bash -n plot_training.sh`、`bash plot_training.sh --help`、`bash evaluate.sh --help`、`git diff --check`。教程参数对照当前离线评测入口检查，不重新启动GPU评测。

## 边界

- 缺失数据不会补零；无配置证据时不声称“未启用”。
- 只有轮询期间观察到的截断/替换才能保留前段；重新启动watch无法恢复已被覆盖的日志。
- 页面刷新重置缩放/图例状态；输出目录各文件分别原子替换，不提供多文件事务。
- 本验收证明读取、口径处理与导出可用，不证明模型能力提升。
- 原始训练日志、生成轨迹、模型、实际图表和机器路径均未纳入源码发布。
