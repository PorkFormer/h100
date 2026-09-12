"""Render measured evidence only; unavailable backend internals remain UNKNOWN."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
from analyze import analyze,paired
R=Path(__file__).resolve().parent;E=R/'evidence'
sel=json.loads((R/'selection.json').read_text());cs=sel['comparison_concurrencies']
rows=json.loads((R/'workload.json').read_text());manifest=json.loads((R/'workload_manifest.json').read_text());lines=[]
def out(s=''):lines.append(s)
def table(header,rows):
    out('| '+' | '.join(header)+' |');out('| '+' | '.join(['---']*len(header))+' |')
    for row in rows:out('| '+' | '.join(map(str,row))+' |')
    out()
def f(x):return f'{x:.3f}'
out('# NCBR scheduler realistic-concurrency performance')
out();out('本轮为 frozen Qwen3-1.7B-Base / 单节点八 A100 / 八 TP1 service 的 systems benchmark。默认保持 fixed_wave；不启动训练，不实现 capacity-aware admission。历史 natural-generation 结果仍为 DIAGNOSTIC_ONLY。')
out()
headstats={c:json.loads((R/f'paired_c{c}.json').read_text()) for c in cs}
out('正式 fixed-workload 结果：'+'；'.join(f"C{c}: {v['mean']:.4f}x, 95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}], {v['status']}" for c,v in headstats.items())+'。')
headgo=(headstats[sel['C_sat']]['status']=='PASS' or headstats[sel['high_load']]['status']=='PASS') and sel['C_sat']<sel['high_load']
out('Capacity-aware decision: **'+('GO' if headgo else 'NO-GO')+'**。本轮 C128 仍最佳且稳定，未定位明显 saturation knee；真实调度收益成立，但尚无足够证据优先引入动态 admission。')
out();out('## Frozen Workload');out()
lens=[r['replay_decode_length'] for r in rows]
out(f"256 requests / {len({r['request_id'] for r in rows})} unique IDs；prompt tokens {sum(len(r['prompt_token_ids']) for r in rows)}；prefix tokens {sum(len(r['prefix_token_ids']) for r in rows)}；total prefill tokens {sum(len(r['input_token_ids']) for r in rows)}；decode tokens {sum(lens)}。")
out(f"Decode length min/P25/P50/P75/P90/P95/max = {np.percentile(lens,[0,25,50,75,90,95,100]).tolist()}。")
out(f"冻结 natural tails 中 finish_reason=length 的请求 {sum(r['finish_reason']=='length' for r in rows)} 条；这些仍为原预算下右删失观测，replay 只强制已观察到的长度，不把 cap-hit 解释为自然终止。")
out(f"Manifest workload SHA256: `{manifest['sha256']}`。完整批次、真实 cap-hit 原始 IDs/natural tails/source mapping 保留在 `evidence/{manifest['source_case']}`；确定性顺序前 256 条入选，完整收集条数 {manifest['all_collected']}。")
out();out('Replay 使用原输入 IDs、seed、temperature=1/top_p=1/top_k=-1；每条 max_tokens=冻结 natural decoded length 且 ignore_eos=true。生成内容无需逐 token 一致。本实验不估计 natural continuation distribution，不进入 reward/actor。')
out();out('引擎配置与旧报告一致，包括原先已启用的 prefix caching；本轮未新增缓存优化。每个实例全新服务及独立 runtime/cache，相同逐服务 8-token warmup，计时不含启动和 warmup。两臂 request_batch_size 均为 256：旧值 8 会硬性限制 fixed_wave 的实际并发至 8，因此不能用于 realistic-concurrency 调度对照。这是 benchmark 配置变化，生产 runtime/source 未改。相同逻辑 prefill IDs 不保证内部 cache hit 工作量相同，缓存及负载均衡的调度相关影响包含在此系统结果内。')
out();out('## Saturation Sweep');out()
sweep=[analyze(f'sweep_c{c}') for c in [8,16,32,64,128]]
table(['C','Requests/service nominal','Wall s','Decode tok/s','Request/s','GPU util %','Idle slot-s','Peak sampled MiB'],[[x['concurrency'],x['concurrency']/8,f(x['continuation_seconds']),f(x['decoded_tokens_per_second']),f(x['request_per_second']),f(x['gpu_util_mean']),f(x['idle_slot_seconds']),max(x['peak_sampled_mib'])] for x in sweep])
out(f"C_sat={sel['C_sat']}；high-load C={sel['high_load']}。C_sat 按最大 observed decode throughput 的 ≥95% 选择最低稳定 C；sweep 每点一次，用于寻找当前128-request工作集的knee；本轮尚未观察到明显平台，不声称已找到backend极限。")
out();out('## Fixed-Workload Scheduler Benchmark');out()
if sel.get('additional_bracket'):out('C_sat 与 high-load 都是 C128，共用同四组配对；额外选择 sweep 相邻的 C64 作第二个 realistic-concurrency 对照点。C64 不冒充 C_sat。')
allpairs={};stats={}
for c in cs:
    pairs=[(analyze(f'pair_c{c}_{i}_fixed_wave'),analyze(f'pair_c{c}_{i}_work_conserving')) for i in range(4)];allpairs[c]=pairs;st=paired(pairs);stats[c]=st
    out();out(f'### C{c}');out()
    table(['Pair','Order','Fixed s','WC s','Fixed/WC'],[[i+1,'AB' if i%2==0 else 'BA',f(a['continuation_seconds']),f(b['continuation_seconds']),f(a['continuation_seconds']/b['continuation_seconds'])] for i,(a,b) in enumerate(pairs)])
    table(['Scheduler','Mean wall s','Request/s','Decode tok/s','Total tok/s','s/request','s/1k decode','s/1k total'],[[s]+[f(float(np.mean([pair[j][k] for pair in pairs]))) for k in ['continuation_seconds','request_per_second','decoded_tokens_per_second','total_tokens_per_second','seconds_per_request','seconds_per_1k_decode','seconds_per_1k_total']] for j,s in enumerate(['fixed_wave','work_conserving'])])
    out(f"Eligibility {st['eligibility']}；arithmetic mean ratio {f(st['mean'])}；median {f(st['median'])}；paired bootstrap 95% CI [{f(st['ci95'][0])}, {f(st['ci95'][1])}]；性能 **{st['status']}**。")
    out()
out('PASS门槛：workload eligibility全部通过、mean speedup≥1.05、paired 95% CI下界>1、无OOM/deadlock/worker loss，且release/cleanup审计PASS。')
out('Bootstrap: 10000 samples, seed 42，配对 ratio 的均值；4 pairs 的区间不代表跨模型/硬件泛化。资格逐实例核对 request order/set、input IDs、per-request decode lengths、total prefill/decode、concurrency、batch size、sampling、service config、GPU UUID topology。所有纳入统计实例退出成功、无 timeout、release/drain 与 owned-process cleanup 审计 PASS。')
formal_instances=[x for pairs in allpairs.values() for pair in pairs for x in pair]
out(f"正式实例 service startup {min(x['startup_seconds'] for x in formal_instances):.2f}–{max(x['startup_seconds'] for x in formal_instances):.2f}s；warmup {min(x['warmup_seconds'] for x in formal_instances):.2f}–{max(x['warmup_seconds'] for x in formal_instances):.2f}s，均不计入continuation。逐实例值见result.json/analysis.json。")
out();out('## Structural Scheduler Analysis');out()
keys=['active_mean','active_p50','active_p90','slot_utilization','idle_slot_seconds','pending_but_idle_wall_seconds','HOL_request_seconds','release_request_seconds','dispatch_request_seconds','admission_queue_request_seconds','gpu_util_mean']
table(['C','Scheduler','Active mean','P50','P90','Slot util','Idle slot-s','Pending idle wall-s','HOL request-s','Release request-s','Dispatch request-s','Admission queue request-s','GPU util %'],[[c,s]+[f(float(np.mean([p[j][k] for p in pairs]))) for k in keys] for c,pairs in allpairs.items() for j,s in enumerate(['fixed_wave','work_conserving'])])
table(['C','Scheduler','Generation slot util','Generation idle slot-s','Pending generation idle wall-s','Dispatch gap P50/P90/max s'],[[c,s,f(float(np.mean([p[j]['generation_slot_utilization'] for p in pairs]))),f(float(np.mean([p[j]['generation_idle_slot_seconds'] for p in pairs]))),f(float(np.mean([p[j]['pending_generation_idle_wall_seconds'] for p in pairs]))),'/'.join(f(float(np.mean([p[j][k] for p in pairs]))) for k in ['dispatch_gap_p50','dispatch_gap_p90','dispatch_gap_max'])] for c,pairs in allpairs.items() for j,s in enumerate(['fixed_wave','work_conserving'])])
out('Generation idle 还包含完成但尚未 release 的槽位；owned/admission idle 不包含它们。二者分别报告，避免 slot ownership 接近满载掩盖 HOL 导致的 backend 供给不足。')
out('Active=已派发且尚未观察到生成完成；slot=派发至 release ACK（含已完成等待 barrier）。HOL 为 generation_complete 到 release_start 的跨请求总和，包含极小本地控制开销。Idle slot-s 为空槽数量×时间；pending idle wall-s 才是存在待派发请求且有空槽的墙钟并集。Admission queue 是从整批 ready 到 dispatch 的请求秒，不能冒充 vLLM queue。逐请求 dispatch/release、engine mapping、时序及加权 occupancy 见每实例 analysis.json/events.json。')
out();out('## GPU and engine profiling');out()
for c,pairs in allpairs.items():
    for j,s in enumerate(['fixed_wave','work_conserving']):
        rs=[p[j] for p in pairs];out(f"C{c} {s} per-GPU mean util: {np.mean([r['gpu_util_per_device_mean'] for r in rs],axis=0).round(2).tolist()}；peak sampled MiB: {np.max([r['peak_sampled_mib'] for r in rs],axis=0).tolist()}。")
out('启动日志记录每服务 GPU KV cache 静态容量为 72,752 tokens；它不是实时 KV 使用率。')
out('每约 2s GPU 时间序列保留在 *_process.json；正式计时窗口的逐卡 P50/P90 在 analysis.json。HBM 为采样峰值，固定预分配会限制其解释。')
example=next(iter(allpairs.values()))[0][0]
out('Backend queue/prefill/decode request metrics availability: '+json.dumps(example['engine_timing'],ensure_ascii=False)+'。')
out('vLLM running/waiting/KV-cache/scheduled-token counters 未通过稳定现有接口获取，标记 UNKNOWN；未读取私有 API 或修改 runtime。没有这些细分指标时不能把端到端生成耗时拆成可信的 prefill 与 decode capacity。')
out();out('## Natural Generation Validation — external validity only');out()
nats=[]
for c in cs:
    for s in ['fixed_wave','work_conserving']:nats.append(analyze(f'natural_c{c}_{s}'))
for sc in ['fixed_wave','work_conserving']:nats.append(analyze(f'natural256_c128_{sc}'))
table(['C','Scheduler','Requests','Actual decode','Wall s','Decode tok/s','s/1k decode','Length min/P25/P50/P75/P90/P95/max'],[[x['concurrency'],x['scheduler'],x['request_count'],x['tokens'],f(x['continuation_seconds']),f(x['decoded_tokens_per_second']),f(x['seconds_per_1k_decode']),x['length_quantiles']] for x in nats])
for c in cs:
    ac=json.loads((E/f'natural_c{c}_fixed_wave/capture.json').read_text());bc=json.loads((E/f'natural_c{c}_work_conserving/capture.json').read_text())
    assert ac['requests']==bc['requests']
    different=sum(a['tail_token_ids']!=b['tail_token_ids'] for a,b in zip(ac['generations'],bc['generations'],strict=True))
    out(f'C{c}: input requests exact；output token content differs in {different}/128 requests。')
ac=json.loads((E/'natural256_c128_fixed_wave/capture.json').read_text());bc=json.loads((E/'natural256_c128_work_conserving/capture.json').read_text())
assert ac['requests']==bc['requests']
out('C128 N256 supplemental input requests exact；output content differing requests: '+str(sum(a['tail_token_ids']!=b['tail_token_ids'] for a,b in zip(ac['generations'],bc['generations'],strict=True)))+'/256。')
out('N128=C128 的自然生成对照无持续补位机会，只作单波控制。因此补充 N256>C128 的一组 natural pair，用于实际存在 pending work 的高负载外部有效性；各实例仍 fresh restart，未添加统计重复。')
out('原小样本使用相同前128 frozen prefixes；补充高负载对照使用全部256。恢复 original max_tokens=6144 与 EOS。输出内容/长度不确定性预期存在；此处只作 normalized external-validity evidence，墙钟比不作为正式 scheduler speedup，也不影响 fixed-workload 资格。每臂一次，无 natural-generation 显著性主张。')
out();out('## Bottleneck After Work-Conserving');out()
best=max(sweep,key=lambda x:x['decoded_tokens_per_second']);gain=best['decoded_tokens_per_second']/sweep[0]['decoded_tokens_per_second']
out(f"WC C8 到 sweep 最优 C{best['concurrency']} 的吞吐比为 {f(gain)}。它量化该 frozen workload 的低并发供给不足，不能把旧 C4/C8 natural-generation 耗时比按比例分解。旧 C4 名义每服务仅 0.5 请求，C8 仅 1；两类实验工作量不同，因此历史收益中 under-feeding 的精确百分比不可辨识。")
for c,pairs in allpairs.items():
    ws=[p[1] for p in pairs]
    out(f"WC C{c}: pending 期间生成空闲墙钟均值 {f(float(np.mean([w['pending_generation_idle_wall_seconds'] for w in ws])))} s；release 累计均值 {f(float(np.mean([w['release_request_seconds'] for w in ws])))} request-s；全部请求派发后到最后完成的 tail-drain 阶段均值 {f(float(np.mean([w['tail_drain_wall_seconds'] for w in ws])))} s。Tail-drain 含必要的有效解码，不能整体称为调度浪费。")
high_wc=float(np.mean([p[1]['decoded_tokens_per_second'] for p in allpairs[sel['high_load']]]))
high_sweep=next(x['decoded_tokens_per_second'] for x in sweep if x['concurrency']==sel['high_load'])
out(f'同为 C{sel["high_load"]}，256-request WC 平均吞吐 {f(high_wc)} tok/s，而128-request sweep为 {f(high_sweep)} tok/s；有限批次填充/尾部效应显著。因此 C_sat=128 是指定95%规则下的操作性选择，真正稳态 saturation point 尚未定位，不能报告为已证明的128并发容量上限。')
out('主要可观测剩余限制是 backend 完成长请求与有限批次尾部：WC 在仍有 pending work 时几乎没有空闲补位间隙，release 累计不足一请求秒，而 tail-drain 约94秒。不能据此进一步区分 prefill、decode算力/带宽、eager引擎控制面；这些需要独立 backend profiling，本轮保持 UNKNOWN。GPU未满载本身不证明 admission 需要动态调节。')
out('HOL/release/occupancy 与上述饱和曲线用于区分调度屏障、供给不足及容量上限。GPU utilization 是忙碌采样比例，不是算力效率。没有 engine 内部排队/KV counters，不能严谨地区分 decode compute、带宽与控制面瓶颈；保留 UNKNOWN，不按猜测归因。')
out();out('## Capacity-Aware Decision');out()
a=stats[sel['C_sat']]['status']=='PASS' or stats[sel['high_load']]['status']=='PASS';knee=sel['C_sat']<sel['high_load'];go=a and knee
out('**'+('GO' if go else 'NO-GO')+'**');out()
if go:out('至少一个 saturated/high-load 对照达到 ≥5% 且 CI 下界>1 的 fixed-workload 收益，同时 sweep 在最高 C 前达到 95% 吞吐 knee，满足 A+B1。GO 仅表示可进入下一阶段设计/验证，本轮未实现 admission。')
elif not a:out('C_sat/high-load 未达到规定的 statistically supported ≥5% speedup，不进入 capacity-aware 实现。当前证据应按已测 low/high-concurrency 范围解释。')
else:out('WC 有符合门槛的收益，但 C128 仍是最低 95% 吞吐点且稳定，没有 B1/B2/B3 的充分证据。可保留显式固定 C128 的实验配置，capacity-aware 优先级下降。')
out();out('## Provenance and limits');out()
pre=json.loads((R/'preflight.json').read_text());out(f"Host `{pre['hostname']}`；checkout `{R.parent}`；branch `{pre['branch']}`；HEAD `{pre['sha']}`；model `{pre['model']}`。源码/模型/dependency/GPU/CUDA 检查见 preflight.json 与 evidence；历史 manifest 的测试文件哈希过时已记录，scheduler/runtime 匹配 validated hash。")
out('固定引擎：vLLM0.18.0，TP1×8，bfloat16，gpu_memory_utilization=0.35，enforce_eager=true，max_model_len=9216，max_num_batched_tokens=8192，max_num_seqs=1024；原enable_prefix_caching=true保留。完整配置见逐实例service_config.yaml。')
out('新增harness targeted tests：13 PASS，见 [测试日志](evidence/targeted_tests_release.log)。协议见 [PROTOCOL.md](PROTOCOL.md)，输入绑定见 [workload_manifest.json](workload_manifest.json)，最终证据绑定见 [final_manifest.json](final_manifest.json)。')
out('逐实例 startup/warmup、源码哈希、service config、Ray GPU 映射、Python paths、cleanup 和完整生成 tokens 保留；未改共享 source、全局 Python、scheduler、算法、verifier、actor 或默认配置。没有运行全量 CPU/actor gate，因为 validated runtime 未改变；新增 harness targeted tests 另附。单工作集、单 checkpoint、四配对及有限 GPU 采样是本轮边界。')
if (E/'final_checks.log').exists():
    assert 'FINAL_BINDING_PASS' in (E/'final_checks.log').read_text()
    out('结束复核：model/workload/protected-source哈希PASS；八卡独立CUDA计算检查PASS，最终0 MiB/0%且无计算进程。最终日志关闭后由seal.py封存并逐文件复验。默认仍为fixed_wave，未启动S300或下一阶段。')
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('REPORT_WRITTEN')
