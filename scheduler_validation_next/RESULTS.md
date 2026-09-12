# NCBR scheduler realistic-concurrency performance

本轮为 frozen Qwen3-1.7B-Base / 单节点八 A100 / 八 TP1 service 的 systems benchmark。默认保持 fixed_wave；不启动训练，不实现 capacity-aware admission。历史 natural-generation 结果仍为 DIAGNOSTIC_ONLY。

正式 fixed-workload 结果：C64: 1.9136x, 95% CI [1.9011, 1.9261], PASS；C128: 1.5183x, 95% CI [1.5096, 1.5270], PASS。
Capacity-aware decision: **NO-GO**。本轮 C128 仍最佳且稳定，未定位明显 saturation knee；真实调度收益成立，但尚无足够证据优先引入动态 admission。

## Frozen Workload

256 requests / 256 unique IDs；prompt tokens 40774；prefix tokens 524288；total prefill tokens 565062；decode tokens 572028。
Decode length min/P25/P50/P75/P90/P95/max = [15.0, 516.5, 1462.5, 3445.5, 6144.0, 6144.0, 6144.0]。
冻结 natural tails 中 finish_reason=length 的请求 27 条；这些仍为原预算下右删失观测，replay 只强制已观察到的长度，不把 cap-hit 解释为自然终止。
Manifest workload SHA256: `000dce9a4848680205605cf0251dff5ecc1e86e01bc5c2138d1902fc844c3ba6`。完整批次、真实 cap-hit 原始 IDs/natural tails/source mapping 保留在 `evidence/collect_v1`；确定性顺序前 256 条入选，完整收集条数 279。

Replay 使用原输入 IDs、seed、temperature=1/top_p=1/top_k=-1；每条 max_tokens=冻结 natural decoded length 且 ignore_eos=true。生成内容无需逐 token 一致。本实验不估计 natural continuation distribution，不进入 reward/actor。

引擎配置与旧报告一致，包括原先已启用的 prefix caching；本轮未新增缓存优化。每个实例全新服务及独立 runtime/cache，相同逐服务 8-token warmup，计时不含启动和 warmup。两臂 request_batch_size 均为 256：旧值 8 会硬性限制 fixed_wave 的实际并发至 8，因此不能用于 realistic-concurrency 调度对照。这是 benchmark 配置变化，生产 runtime/source 未改。相同逻辑 prefill IDs 不保证内部 cache hit 工作量相同，缓存及负载均衡的调度相关影响包含在此系统结果内。

## Saturation Sweep

| C | Requests/service nominal | Wall s | Decode tok/s | Request/s | GPU util % | Idle slot-s | Peak sampled MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1.0 | 533.157 | 528.137 | 0.240 | 37.448 | 134.346 | 14773 |
| 16 | 2.0 | 300.805 | 936.090 | 0.426 | 37.817 | 465.206 | 14773 |
| 32 | 4.0 | 191.209 | 1472.628 | 0.669 | 37.129 | 1728.483 | 14773 |
| 64 | 8.0 | 122.199 | 2304.275 | 1.047 | 39.707 | 3396.177 | 14773 |
| 128 | 16.0 | 97.631 | 2884.133 | 1.311 | 46.161 | 8049.790 | 14773 |

C_sat=128；high-load C=128。C_sat 按最大 observed decode throughput 的 ≥95% 选择最低稳定 C；sweep 每点一次，用于寻找当前128-request工作集的knee；本轮尚未观察到明显平台，不声称已找到backend极限。

## Fixed-Workload Scheduler Benchmark

C_sat 与 high-load 都是 C128，共用同四组配对；额外选择 sweep 相邻的 C64 作第二个 realistic-concurrency 对照点。C64 不冒充 C_sat。

### C64

| Pair | Order | Fixed s | WC s | Fixed/WC |
| --- | --- | --- | --- | --- |
| 1 | AB | 386.727 | 200.478 | 1.929 |
| 2 | BA | 385.927 | 203.050 | 1.901 |
| 3 | AB | 386.017 | 200.717 | 1.923 |
| 4 | BA | 382.849 | 201.331 | 1.902 |

| Scheduler | Mean wall s | Request/s | Decode tok/s | Total tok/s | s/request | s/1k decode | s/1k total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_wave | 385.380 | 0.664 | 1484.344 | 2950.611 | 1.505 | 0.674 | 0.339 |
| work_conserving | 201.394 | 1.271 | 2840.414 | 5646.238 | 0.787 | 0.352 | 0.177 |

Eligibility PASS；arithmetic mean ratio 1.914；median 1.912；paired bootstrap 95% CI [1.901, 1.926]；性能 **PASS**。


### C128

| Pair | Order | Fixed s | WC s | Fixed/WC |
| --- | --- | --- | --- | --- |
| 1 | AB | 195.383 | 127.848 | 1.528 |
| 2 | BA | 194.595 | 129.272 | 1.505 |
| 3 | AB | 196.194 | 128.596 | 1.526 |
| 4 | BA | 195.951 | 129.441 | 1.514 |

| Scheduler | Mean wall s | Request/s | Decode tok/s | Total tok/s | s/request | s/1k decode | s/1k total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_wave | 195.531 | 1.309 | 2925.543 | 5815.459 | 0.764 | 0.342 | 0.172 |
| work_conserving | 128.789 | 1.988 | 4441.689 | 8829.288 | 0.503 | 0.225 | 0.113 |

Eligibility PASS；arithmetic mean ratio 1.518；median 1.520；paired bootstrap 95% CI [1.510, 1.527]；性能 **PASS**。

PASS门槛：workload eligibility全部通过、mean speedup≥1.05、paired 95% CI下界>1、无OOM/deadlock/worker loss，且release/cleanup审计PASS。
Bootstrap: 10000 samples, seed 42，配对 ratio 的均值；4 pairs 的区间不代表跨模型/硬件泛化。资格逐实例核对 request order/set、input IDs、per-request decode lengths、total prefill/decode、concurrency、batch size、sampling、service config、GPU UUID topology。所有纳入统计实例退出成功、无 timeout、release/drain 与 owned-process cleanup 审计 PASS。
正式实例 service startup 79.43–83.19s；warmup 2.26–2.33s，均不计入continuation。逐实例值见result.json/analysis.json。

## Structural Scheduler Analysis

| C | Scheduler | Active mean | P50 | P90 | Slot util | Idle slot-s | Pending idle wall-s | HOL request-s | Release request-s | Dispatch request-s | Admission queue request-s | GPU util % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | fixed_wave | 23.238 | 19.000 | 46.000 | 0.999 | 23.218 | 0.434 | 15685.343 | 0.211 | 8.329 | 37102.838 | 38.459 |
| 64 | work_conserving | 44.516 | 64.000 | 64.000 | 0.696 | 3923.514 | 0.220 | 0.183 | 0.364 | 2.759 | 10257.684 | 44.964 |
| 128 | fixed_wave | 46.207 | 37.000 | 93.000 | 0.998 | 45.312 | 0.370 | 15947.534 | 0.205 | 16.503 | 12554.468 | 46.617 |
| 128 | work_conserving | 70.019 | 65.250 | 128.000 | 0.547 | 7466.934 | 0.238 | 0.191 | 0.328 | 8.245 | 2074.016 | 51.587 |

| C | Scheduler | Generation slot util | Generation idle slot-s | Pending generation idle wall-s | Dispatch gap P50/P90/max s |
| --- | --- | --- | --- | --- | --- |
| 64 | fixed_wave | 0.363 | 15708.772 | 287.614 | 0.000/0.001/96.930 |
| 64 | work_conserving | 0.696 | 3924.062 | 0.607 | 0.232/1.047/5.056 |
| 128 | fixed_wave | 0.361 | 15993.050 | 97.127 | 0.001/0.001/97.706 |
| 128 | work_conserving | 0.547 | 7467.453 | 0.464 | 0.002/0.405/1.564 |

Generation idle 还包含完成但尚未 release 的槽位；owned/admission idle 不包含它们。二者分别报告，避免 slot ownership 接近满载掩盖 HOL 导致的 backend 供给不足。
Active=已派发且尚未观察到生成完成；slot=派发至 release ACK（含已完成等待 barrier）。HOL 为 generation_complete 到 release_start 的跨请求总和，包含极小本地控制开销。Idle slot-s 为空槽数量×时间；pending idle wall-s 才是存在待派发请求且有空槽的墙钟并集。Admission queue 是从整批 ready 到 dispatch 的请求秒，不能冒充 vLLM queue。逐请求 dispatch/release、engine mapping、时序及加权 occupancy 见每实例 analysis.json/events.json。

## GPU and engine profiling

C64 fixed_wave per-GPU mean util: [38.43, 34.81, 41.68, 39.51, 36.67, 41.75, 41.31, 33.5]；peak sampled MiB: [14773, 14773, 14773, 14773, 14773, 14773, 14773, 14773]。
C64 work_conserving per-GPU mean util: [43.75, 46.77, 47.25, 43.26, 43.36, 43.03, 46.3, 46.0]；peak sampled MiB: [14773, 14773, 14773, 14773, 14773, 14773, 14773, 14773]。
C128 fixed_wave per-GPU mean util: [47.21, 45.18, 49.71, 47.5, 43.87, 50.0, 48.82, 40.64]；peak sampled MiB: [14773, 14773, 14773, 14773, 14773, 14773, 14773, 14773]。
C128 work_conserving per-GPU mean util: [51.61, 53.92, 53.01, 47.86, 47.39, 49.99, 53.72, 55.18]；peak sampled MiB: [14773, 14773, 14773, 14773, 14773, 14773, 14773, 14773]。
启动日志记录每服务 GPU KV cache 静态容量为 72,752 tokens；它不是实时 KV 使用率。
每约 2s GPU 时间序列保留在 *_process.json；正式计时窗口的逐卡 P50/P90 在 analysis.json。HBM 为采样峰值，固定预分配会限制其解释。
Backend queue/prefill/decode request metrics availability: {"continuation_queue": {"available": false, "count": 0, "request_seconds": null}, "continuation_prefill_engine": {"available": false, "count": 0, "request_seconds": null}, "continuation_decode_engine": {"available": false, "count": 0, "request_seconds": null}, "continuation_cleanup_release": {"available": true, "count": 256, "request_seconds": 0.22825506346998736}}。
vLLM running/waiting/KV-cache/scheduled-token counters 未通过稳定现有接口获取，标记 UNKNOWN；未读取私有 API 或修改 runtime。没有这些细分指标时不能把端到端生成耗时拆成可信的 prefill 与 decode capacity。

## Natural Generation Validation — external validity only

| C | Scheduler | Requests | Actual decode | Wall s | Decode tok/s | s/1k decode | Length min/P25/P50/P75/P90/P95/max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | fixed_wave | 128 | 282503 | 191.884 | 1472.256 | 0.679 | [18.0, 527.0, 1367.0, 3398.5, 6144.0, 6144.0, 6144.0] |
| 64 | work_conserving | 128 | 294022 | 128.750 | 2283.673 | 0.438 | [18.0, 584.75, 1660.0, 3632.5, 5938.499999999999, 6144.0, 6144.0] |
| 128 | fixed_wave | 128 | 295420 | 97.397 | 3033.139 | 0.330 | [18.0, 591.25, 1652.0, 3557.0, 6027.599999999999, 6144.0, 6144.0] |
| 128 | work_conserving | 128 | 295420 | 97.034 | 3044.504 | 0.328 | [18.0, 591.25, 1652.0, 3557.0, 6027.599999999999, 6144.0, 6144.0] |
| 128 | fixed_wave | 256 | 617611 | 194.345 | 3177.910 | 0.315 | [15.0, 636.5, 1691.0, 4039.25, 6144.0, 6144.0, 6144.0] |
| 128 | work_conserving | 256 | 583990 | 130.861 | 4462.662 | 0.224 | [15.0, 565.25, 1534.5, 3689.5, 6144.0, 6144.0, 6144.0] |

C64: input requests exact；output token content differs in 92/128 requests。
C128: input requests exact；output token content differs in 0/128 requests。
C128 N256 supplemental input requests exact；output content differing requests: 201/256。
N128=C128 的自然生成对照无持续补位机会，只作单波控制。因此补充 N256>C128 的一组 natural pair，用于实际存在 pending work 的高负载外部有效性；各实例仍 fresh restart，未添加统计重复。
原小样本使用相同前128 frozen prefixes；补充高负载对照使用全部256。恢复 original max_tokens=6144 与 EOS。输出内容/长度不确定性预期存在；此处只作 normalized external-validity evidence，墙钟比不作为正式 scheduler speedup，也不影响 fixed-workload 资格。每臂一次，无 natural-generation 显著性主张。

## Bottleneck After Work-Conserving

WC C8 到 sweep 最优 C128 的吞吐比为 5.461。它量化该 frozen workload 的低并发供给不足，不能把旧 C4/C8 natural-generation 耗时比按比例分解。旧 C4 名义每服务仅 0.5 请求，C8 仅 1；两类实验工作量不同，因此历史收益中 under-feeding 的精确百分比不可辨识。
WC C64: pending 期间生成空闲墙钟均值 0.607 s；release 累计均值 0.364 request-s；全部请求派发后到最后完成的 tail-drain 阶段均值 94.002 s。Tail-drain 含必要的有效解码，不能整体称为调度浪费。
WC C128: pending 期间生成空闲墙钟均值 0.464 s；release 累计均值 0.328 request-s；全部请求派发后到最后完成的 tail-drain 阶段均值 94.695 s。Tail-drain 含必要的有效解码，不能整体称为调度浪费。
同为 C128，256-request WC 平均吞吐 4441.689 tok/s，而128-request sweep为 2884.133 tok/s；有限批次填充/尾部效应显著。因此 C_sat=128 是指定95%规则下的操作性选择，真正稳态 saturation point 尚未定位，不能报告为已证明的128并发容量上限。
主要可观测剩余限制是 backend 完成长请求与有限批次尾部：WC 在仍有 pending work 时几乎没有空闲补位间隙，release 累计不足一请求秒，而 tail-drain 约94秒。不能据此进一步区分 prefill、decode算力/带宽、eager引擎控制面；这些需要独立 backend profiling，本轮保持 UNKNOWN。GPU未满载本身不证明 admission 需要动态调节。
HOL/release/occupancy 与上述饱和曲线用于区分调度屏障、供给不足及容量上限。GPU utilization 是忙碌采样比例，不是算力效率。没有 engine 内部排队/KV counters，不能严谨地区分 decode compute、带宽与控制面瓶颈；保留 UNKNOWN，不按猜测归因。

## Capacity-Aware Decision

**NO-GO**

WC 有符合门槛的收益，但 C128 仍是最低 95% 吞吐点且稳定，没有 B1/B2/B3 的充分证据。可保留显式固定 C128 的实验配置，capacity-aware 优先级下降。

## Provenance and limits

Host `p-kt-dgx-a100-05`；checkout `/tmp/ncbr_scheduler_20260911`；branch `perf/continuation-scheduler`；HEAD `d7a0f810c678f2b20c55085dca182a109fdd500a`；model `/workspace/models/Qwen3-1.7B-Base`。源码/模型/dependency/GPU/CUDA 检查见 preflight.json 与 evidence；历史 manifest 的测试文件哈希过时已记录，scheduler/runtime 匹配 validated hash。
固定引擎：vLLM0.18.0，TP1×8，bfloat16，gpu_memory_utilization=0.35，enforce_eager=true，max_model_len=9216，max_num_batched_tokens=8192，max_num_seqs=1024；原enable_prefix_caching=true保留。完整配置见逐实例service_config.yaml。
新增harness targeted tests：13 PASS，见 [测试日志](evidence/targeted_tests_release.log)。协议见 [PROTOCOL.md](PROTOCOL.md)，输入绑定见 [workload_manifest.json](workload_manifest.json)，最终证据绑定见 [final_manifest.json](final_manifest.json)。
逐实例 startup/warmup、源码哈希、service config、Ray GPU 映射、Python paths、cleanup 和完整生成 tokens 保留；未改共享 source、全局 Python、scheduler、算法、verifier、actor 或默认配置。没有运行全量 CPU/actor gate，因为 validated runtime 未改变；新增 harness targeted tests 另附。单工作集、单 checkpoint、四配对及有限 GPU 采样是本轮边界。
结束复核：model/workload/protected-source哈希PASS；八卡独立CUDA计算检查PASS，最终0 MiB/0%且无计算进程。最终日志关闭后由seal.py封存并逐文件复验。默认仍为fixed_wave，未启动S300或下一阶段。
