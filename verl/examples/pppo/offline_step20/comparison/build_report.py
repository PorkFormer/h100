"""Rebuild the PPPO20 versus NCBR H2048/L8192 S300 evidence report."""
import csv, json, hashlib, shutil, datetime, html, zipfile
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[5]
ROOT=Path('/workspace/rl/h100')
RUN=ROOT/'outputs/pppo_ncbr_matched_16xa10040g_s300_ckpt1_v2'
EVAL=RUN/'offline_eval_step20_h8192_math500_v1'
COST=ROOT/'analysis/qwen3_4b_ncbr_baseline_s300_cost_20260920'
HIST=ROOT/'analysis/ncbr_boundary_return_s300_seed1_20260828'
OUT=REPO/'reports/pppo20_vs_ncbr_h2048_l8192_s300_20261114'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def csvwrite(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def at(c,path):
 for k in path.split('.'): c=c[k]
 return c
records=[json.loads(l) for l in (RUN/'training_metrics.jsonl').open()]
assert [r['step'] for r in records]==list(range(1,21))
pm=[r['metrics'] for r in records]
assert all(m['actor/pppo_optimizer_updates_this_outer']==16 for m in pm)
assert pm[-1]['actor/pppo_optimizer_updates_total']==320
schedule=[json.loads(l) for l in (RUN/'schedule_events.jsonl').open()]
nrows=[r for r in csv.DictReader((COST/'steps.csv').open()) if r['run_id']=='dapo_ncbr_l8192']
assert [int(r['step']) for r in nrows]==list(range(1,301))
ns=next(r for r in csv.DictReader((COST/'summary.csv').open()) if r['id']=='dapo_ncbr_l8192')
mechanism=list(csv.DictReader((HIST/'boundary_return_mechanism_cost_steps_1_300.csv').open()))
def nsum(k):return sum(float(r[k] or 0) for r in nrows)
def psum(k):return sum(float(m.get(k,0)) for m in pm)
pause=read(RUN/'pause_step20_receipt.json')
ppath=Path(pause['training_status_before_stop']['attempt'])/'resolved_config.yaml'
npath=COST/'references/dapo_ncbr_l8192/provenance/resolved_config.yaml'
parentpath=COST/'references/dapo_ncbr_l8192_s20/provenance/resolved_config.yaml'
pc=yaml.safe_load(ppath.read_text());nc=yaml.safe_load(npath.read_text())
assert nc['data']['max_response_length']==2048 and nc['actor_rollout_ref']['rollout']['boundary_return']['long_response_length']==8192
for source,name in [(ppath,'pppo_actual_resolved.yaml'),(npath,'ncbr_s21_300_resolved.yaml'),(parentpath,'ncbr_s1_20_resolved.yaml'),(RUN/'training_metrics.jsonl','pppo_training_metrics.jsonl'),(RUN/'schedule_events.jsonl','pppo_schedule_events.jsonl')]:shutil.copyfile(source,OUT/name)
# Full field-level diff preserves inactive fields without pretending they are operative.
def flatten(v,p=''):
 if isinstance(v,dict):
  z={}
  for k,x in v.items():z.update(flatten(x,p+'.'+k if p else k))
  return z
 return {p:v}
pf,nf=flatten(pc),flatten(nc)
write(OUT/'full_config_diff.json',[{'field':k,'pppo':pf.get(k,'<missing>'),'ncbr':nf.get(k,'<missing>')} for k in sorted(pf.keys()|nf.keys()) if pf.get(k)!=nf.get(k)])
fields=['data.train_batch_size','data.gen_batch_size','data.max_prompt_length','data.max_response_length','data.shuffle','data.seed','data.dataloader_num_workers','trainer.nnodes','trainer.n_gpus_per_node','trainer.total_epochs','trainer.total_training_steps','trainer.save_freq','trainer.test_freq','trainer.resume_mode','trainer.val_before_train','actor_rollout_ref.model.path','actor_rollout_ref.model.enable_gradient_checkpointing','actor_rollout_ref.actor.ppo_mini_batch_size','actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu','actor_rollout_ref.actor.ppo_epochs','actor_rollout_ref.actor.shuffle','actor_rollout_ref.actor.data_loader_seed','actor_rollout_ref.actor.optim.lr','actor_rollout_ref.actor.optim.weight_decay','actor_rollout_ref.actor.optim.betas','actor_rollout_ref.actor.optim.lr_scheduler_type','actor_rollout_ref.actor.optim.lr_warmup_steps','actor_rollout_ref.actor.grad_clip','actor_rollout_ref.actor.fsdp_config.param_offload','actor_rollout_ref.actor.fsdp_config.optimizer_offload','actor_rollout_ref.actor.fsdp_config.model_dtype','actor_rollout_ref.actor.fsdp_config.dtype','actor_rollout_ref.actor.fsdp_config.seed','actor_rollout_ref.actor.ulysses_sequence_parallel_size','actor_rollout_ref.actor.clip_ratio_low','actor_rollout_ref.actor.clip_ratio_high','actor_rollout_ref.actor.clip_ratio_c','actor_rollout_ref.actor.entropy_coeff','actor_rollout_ref.actor.use_kl_loss','actor_rollout_ref.rollout.n','actor_rollout_ref.rollout.tensor_model_parallel_size','actor_rollout_ref.rollout.nnodes','actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu','actor_rollout_ref.rollout.temperature','actor_rollout_ref.rollout.top_p','actor_rollout_ref.rollout.top_k','actor_rollout_ref.rollout.ignore_eos','actor_rollout_ref.rollout.max_model_len','actor_rollout_ref.rollout.max_num_seqs','actor_rollout_ref.rollout.max_num_batched_tokens','actor_rollout_ref.rollout.gpu_memory_utilization','algorithm.filter_groups.metric','algorithm.filter_groups.max_num_gen_batches','reward.reward_kwargs.overlong_buffer_cfg.enable']
configrows=[{'field':k,'PPPO':json.dumps(at(pc,k),ensure_ascii=False),'NCBR':json.dumps(at(nc,k),ensure_ascii=False)} for k in fields]
csvwrite('training_configuration.csv',configrows)
# Timers: testing is outside step; internal continuation timers overlap and are not added twice.
pt=psum('timing_s/step');pv=psum('timing_s/testing');nt=nsum('step_s');nv=nsum('validation_s')
pcost={'Original generation':psum('timing_s/gen'),'Continuation incl reward':psum('timing_s/pppo_continuation_and_score'),'Actor update':psum('timing_s/update_actor'),'Old logprob':psum('timing_s/old_log_prob'),'Checkpoint':psum('timing_s/save_checkpoint'),'Weight sync':psum('timing_s/update_weights')}
ncost={'Original generation':nsum('gen_s'),'Continuation incl reward':nsum('continuation_s')+nsum('long_reward_s'),'Actor update':nsum('actor_update_s'),'Old logprob':nsum('logprob_s'),'Checkpoint':nsum('save_s'),'Weight sync':nsum('timing_s/update_weights')}
for d,total,val in [(pcost,pt,pv),(ncost,nt,nv)]:d['Other inside step']=total-sum(d.values());d['Online validation']=val
assert min(pcost.values())>=0 and min(ncost.values())>=0
components=[{'phase':k,'PPPO_GPU_hours':v*16/3600,'NCBR_GPU_hours':ncost[k]*16/3600} for k,v in pcost.items()];csvwrite('cost_components.csv',components)
process_n=read(HIST/'gpu_cost_summary.json')['training']['full_lineage_s1_300_gpu_hours_16x']
proc_p=pause['step20_process_gpu_hours']
summary={'PPPO':{'steps':20,'updates_measured':320,'step_gpu_hours':pt*16/3600,'online_validation_gpu_hours':pv*16/3600,'step_plus_testing_gpu_hours':(pt+pv)*16/3600,'process_gpu_hours':proc_p,'trailing_process_gpu_hours':pause['trailing_process_gpu_hours'],'original_rows':psum('pppo/candidate_original_rows'),'continuation_rows':psum('pppo/continuation_rows'),'original_tokens':psum('pppo/original_tokens'),'continuation_decoded_tokens':psum('pppo/tail_tokens'),'continuation_prefill_tokens':psum('pppo/continuation_prefill_tokens'),'actor_optimized_tokens':psum('pppo/optimized_tokens'),'actor_actual_input_tokens':psum('pppo/actor_input_tokens'),'actor_rows':psum('pppo/actor_rows'),'candidate_groups':psum('train/generated_prompt_groups'),'filtered_groups':psum('train/filtered_prompt_groups'),'valid_excess_groups':psum('pppo/valid_excess_groups')},'NCBR':{'steps':300,'updates_theoretical':4800,'updates_counter_available':False,'step_gpu_hours':nt*16/3600,'online_validation_gpu_hours':nv*16/3600,'step_plus_testing_gpu_hours':(nt+nv)*16/3600,'process_gpu_hours':process_n,'original_rows':nsum('candidate_trajectories'),'continuation_rows':sum(float(r['boundary_return/continuation_request_count'] or 0) for r in mechanism),'original_tokens':float(ns['candidate_response_tokens']),'continuation_decoded_tokens':float(ns['continuation_tokens']),'continuation_prefill_tokens':sum(float(r['boundary_return/continuation_input_tokens'] or 0) for r in mechanism),'actor_optimized_tokens':float(ns['update_response_tokens']),'actor_rows':300*2048}}
write(OUT/'cost_and_token_summary.json',summary)
cum=0;dynamic=[]
for r,m in zip(records,pm):
 cum+=(m['timing_s/step']+m.get('timing_s/testing',0))*16/3600
 dynamic.append({'step':r['step'],'loss':m['actor/loss'],'grad_norm':m['actor/grad_norm'],'original_candidate_accuracy':m['pppo/candidate_original_accuracy'],'cumulative_reward_mean':m['pppo/candidate_reward_mean'],'eta_percent':m['pppo/eta_percent'],'holdout_accuracy':m.get('pppo/holdout_accuracy',''),'step_minutes':m['timing_s/step']/60,'cumulative_step_plus_testing_gpu_hours':cum,'updates_total':m['actor/pppo_optimizer_updates_total'],'original_tokens':m['pppo/original_tokens'],'continuation_tokens':m['pppo/tail_tokens'],'optimized_tokens':m['pppo/optimized_tokens'],'actor_input_tokens':m['pppo/actor_input_tokens'],'filtered_groups':m['train/filtered_prompt_groups'],'valid_excess_groups':m['pppo/valid_excess_groups']})
csvwrite('pppo_training_dynamics.csv',dynamic);csvwrite('ncbr_training_steps.csv',nrows)
# Only these two model arms enter the report.
assert read(EVAL/'pppo_s20/status.json')['status']==read(EVAL/'ncbr_s300/status.json')['status']=='PASS'
pres=read(EVAL/'pppo_s20/scores.json');nmath=read(EVAL/'ncbr_s300/scores.json');nhist=read(REPO/'experiments/pppo_ncbr_matched/provenance/ncbr_step300_h8192.json')
preflight=read(EVAL/'checkpoint_preflight.json')
assert nhist['checkpoint_sha256']==[preflight['ncbr_s300']['checkpoint']['files']['model.safetensors']]
for f,h in nhist['source_sha256'].items():assert sha(f)==h
nm={m['benchmark']:m for m in nhist['metrics']+nmath['metrics']}
bench=[];paired=[]
pq={r['prompt_id']:r for r in pres['per_question']};nq={r['prompt_id']:r for r in nhist['per_question']+nmath['per_question']}
for m in pres['metrics']:
 n=nm[m['benchmark']];bench.append({'benchmark':m['benchmark'],'questions':m['questions'],'PPPO_avg32_pct':100*m['avg32'],'PPPO_avg32_se_pp':100*m['avg32_se'],'NCBR_avg32_pct':100*n['avg32'],'NCBR_avg32_se_pp':100*n['avg32_se'],'avg32_delta_pp':100*(m['avg32']-n['avg32']),'PPPO_pass32_pct':100*m['pass32'],'PPPO_pass32_se_pp':100*m['pass32_se'],'NCBR_pass32_pct':100*n['pass32'],'NCBR_pass32_se_pp':100*n['pass32_se'],'pass32_delta_pp':100*(m['pass32']-n['pass32']),'PPPO_mean_tokens':m['mean_tokens'],'NCBR_mean_tokens':n['mean_tokens']})
 ids=[r['prompt_id'] for r in pres['per_question'] if r['benchmark']==m['benchmark']]
 assert len(ids)==m['questions'] and all(i in nq for i in ids)
 for metric in ['avg32','pass32']:
  ds=np.array([pq[i][metric]-nq[i][metric] for i in ids])*100;rng=np.random.default_rng(20260812);boot=ds[rng.integers(0,len(ds),size=(10000,len(ds)))].mean(axis=1)
  paired.append({'benchmark':m['benchmark'],'metric':metric,'delta_pp':float(ds.mean()),'ci95_pp':np.quantile(boot,[.025,.975]).tolist()})
csvwrite('offline_results.csv',bench);write(OUT/'paired_bootstrap.json',paired)
# Exportable figures; no GPU inference involved in report generation.
fig,ax=plt.subplots(2,2,figsize=(12,8));x=[d['step'] for d in dynamic]
ax[0,0].plot(x,[d['original_candidate_accuracy']*100 for d in dynamic],label='Original candidate accuracy');ax[0,0].plot([s['step'] for s in schedule],[s['accuracy']*100 for s in schedule],'o-',label='Independent holdout');ax[0,0].set_ylabel('Accuracy (%)');ax[0,0].legend()
ax[0,1].plot(x,[d['loss'] for d in dynamic],label='Prefix loss');ax2=ax[0,1].twinx();ax2.plot(x,[d['grad_norm'] for d in dynamic],color='orange',label='Grad norm');ax[0,1].set_ylabel('PPPO loss');ax2.set_ylabel('Grad norm')
ax[1,0].plot(x,[d['step_minutes'] for d in dynamic]);ax[1,0].set_ylabel('Step wall minutes')
ax[1,1].plot(x,[d['cumulative_step_plus_testing_gpu_hours'] for d in dynamic]);ax[1,1].set_ylabel('Cumulative step + testing GPU-hours')
for a in ax.flat:a.set_xlabel('Outer step');a.grid(alpha=.2)
fig.suptitle('PPPO formal run: steps 1-20 (eta remained 15%)');fig.tight_layout();fig.savefig(OUT/'pppo_dynamics.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4));left=np.zeros(2)
for k in pcost:
 vals=np.array([pcost[k],ncost[k]])*16/3600;ax.barh(['PPPO S20','NCBR S300'],vals,left=left,label=k);left+=vals
ax.set_xlabel('Allocated GPU-hours: step + online validation (not process lifetime)');ax.legend(bbox_to_anchor=(1.02,1),loc='upper left');fig.tight_layout();fig.savefig(OUT/'cost_breakdown.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(12,4));xx=np.arange(4)
for a,metric in zip(ax,['avg32','pass32']):
 for off,key,color in [(-.18,'PPPO','tab:blue'),(.18,'NCBR','tab:orange')]:a.bar(xx+off,[r[f'{key}_{metric}_pct'] for r in bench],.36,yerr=[r[f'{key}_{metric}_se_pp'] for r in bench],label=key,color=color,capsize=3)
 a.set_xticks(xx,[r['benchmark'] for r in bench]);a.set_ylabel(metric+' (%) +/- question SE');a.legend()
fig.suptitle('H_eval=8192, 32 samples/question; PPPO S20 vs NCBR H2048/L8192 S300');fig.tight_layout();fig.savefig(OUT/'offline_comparison.png',dpi=170);plt.close(fig)
P,N=summary['PPPO'],summary['NCBR'];delta=(proc_p/process_n-1)*100
lines=['# PPPO Step20 与 NCBR H=2048/L=8192 Step300 对比报告','',f'生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。仅比较这两组模型，不纳入 vanilla DAPO 或其他 NCBR 长度配置。','',
'## 1. 结论与比较边界','',f'PPPO 正式训练已安全暂停在 Step20（实际 320 次 optimizer update），NCBR 历史实验为 Step300。训练进程口径分别为 {proc_p:.3f} 与 {process_n:.3f} GPU-hours，PPPO 高 {delta:.3f}%；这是近似预算对照，不是严格相同算力或相同训练步数对照。',
'在相同 H_eval=8192、每题32次采样的离线协议下，PPPO 的四个题库 Pass@1/Avg@32 均低于 NCBR；MATH-500 Pass@32 更高。单次正确率与32次尝试覆盖率是不同指标，不能据后者宣称整体优于 NCBR。仅一个训练种子，不能把题目 bootstrap 当成跨训练种子显著性。',
'PPPO 的 holdout 改善表明独立开发集上的训练进展，但不能替代 AIME/AMC/MATH500 测试结果，也不能与 NCBR 的在线 AIME 分数直接比较。','',
'## 2. PPPO 训练动态','',table(['验证 step','独立 holdout 准确率','eta'],[(s['step'],f"{s['accuracy']*100:.4f}%",s['eta']) for s in schedule]),'',
'Holdout 为256个唯一题目；eta 全程15%，尚未发生 plateau 上调。规则 patience=3、min_delta=0、cooldown=1、每10步验证，为复现补全，并非确认来自原作者代码。',
 table(['step','原始候选正确率','累计奖励均值','loss','grad norm','分钟/step','累计更新'],[(r['step'],f"{r['original_candidate_accuracy']*100:.2f}%",f"{r['cumulative_reward_mean']:.4f}",f"{r['loss']:.6f}",f"{r['grad_norm']:.5f}",f"{r['step_minutes']:.2f}",int(r['updates_total'])) for r in dynamic]),'',
'原始候选正确率的分母是当步生成的候选回答；累计奖励范围0–9；均不同于筛选后的训练 reward，也不同于固定测试集成绩。PPPO loss 使用原始总长度归一化，不能直接拿其数值或 grad norm 大小与 NCBR 判断优化优劣。','',
'![训练动态](pppo_dynamics.png)','',
'## 3. 真实 GPU-hours 与生成/更新开销','',
 table(['口径','PPPO S20','NCBR S300'],[['进程链 GPU-hours',f'{proc_p:.3f}',f'{process_n:.3f}'],['step timer GPU-hours',f'{P["step_gpu_hours"]:.3f}',f'{N["step_gpu_hours"]:.3f}'],['在线验证 timer GPU-hours（step外）',f'{P["online_validation_gpu_hours"]:.3f}',f'{N["online_validation_gpu_hours"]:.3f}'],['step + testing GPU-hours',f'{P["step_plus_testing_gpu_hours"]:.3f}',f'{N["step_plus_testing_gpu_hours"]:.3f}'],['平均 step timer 分钟',f'{pt/20/60:.2f}',f'{nt/300/60:.2f}']]),'',
'GPU-hours=资源保留期间墙钟时间×16/3600，不是按GPU利用率或功率积分。NCBR进程链包含原始1–20步与恢复后21–300步；PPPO进程终点取Step20 committed sidecar时间。启动/Step0验证等边界开销落在进程口径中，step计时不能覆盖全部。两者周期在线验证位于step计时之外；checkpoint计时已在step内，不得重复相加。',
 f'PPPO committed保存到driver退出另有 {pause["trailing_process_gpu_hours"]:.5f} GPU-hours，单列，不算作Step20模型训练成本。正式训练首次从Base初始化，未从smoke/profiling checkpoint继续。',
 table(['阶段（互斥归类）','PPPO GPU-h','NCBR GPU-h'],[(r['phase'],f"{r['PPPO_GPU_hours']:.3f}",f"{r['NCBR_GPU_hours']:.3f}") for r in components]),'',
'PPPO continuation总阶段包含其生成、评分和调度，内部嵌套timer未重复相加；NCBR使用continuation与long reward计时。各阶段乘16表示此阶段占用整个训练资源池的机会成本，不表示所有16卡都执行该阶段内核。','',
'![成本分解](cost_breakdown.png)','',
 table(['累计统计','PPPO S20','NCBR S300'],[(k,f'{P[k]:,.0f}',f'{N[k]:,.0f}') for k in ['original_rows','continuation_rows','original_tokens','continuation_decoded_tokens','continuation_prefill_tokens','actor_optimized_tokens','actor_rows']]+[['actor实际输入tokens（含prompt）',f'{P["actor_actual_input_tokens"]:,.0f}','本报告未重建，不与response tokens混用']]),'',
 f'PPPO 候选组共 {P["candidate_groups"]:,.0f}，零方差筛掉 {P["filtered_groups"]:,.0f}，另有有效但超出目标批大小的组 {P["valid_excess_groups"]:,.0f}；全部候选原始生成和续写成本均已计入，不能只计算保留的256组。每候选批为6144条原始回答+49152条额外续写。',
 f'PPPO 的 continuation 阶段占 step timer {pcost["Continuation incl reward"]/pt*100:.2f}%，actor update 占 {pcost["Actor update"]/pt*100:.2f}%。这解释了为什么前缀actor计算较少，整体仍可比300步NCBR昂贵：G=8续写覆盖所有原始分支，而NCBR只对命中短边界的轨迹续写一次。不得将15% prefix比例表述成85% GPU-hours节省。',
'NCBR candidate response token总量来自历史审计的 EXACT_OR_INTEGER_RECOVERED 汇总，并非本报告重新跑推理计数。NCBR 4800次更新为300×16理论值，历史表中未提供逐步optimizer计数；本报告不把理论值冒充实测。','',
'## 4. 算法流程对照','',
'**PPPO + DAPO Dynamic Sampling（NCBR-Matched）**：每题先生成N=8条完整原始回答，cap=8192；每条按floor(eta×实际长度)保留prefix，再从原prompt IDs+prefix IDs独立生成G=8条continuation，总回答长度仍≤8192。对原始回答和8条续写做binary correctness评分并求和，在同题8个原始分支内标准化advantage；全分支评分后用累计奖励进行动态筛选。只对原始分支prefix执行clipped PPO，续写不进入actor样本。每题分母为该题8条原始完整回答长度之和，不能改为prefix长度和。可写作：L_q = -[Σ_i Σ_{t≤floor(eta L_i)} min(r_it A_i, clip(r_it,0.8,1.28) A_i)] / [Σ_i L_i]。',
'**NCBR H2048/L8192**：先生成N=8条短回答，H=2048。只对命中短response cap且未正常结束的轨迹，从原prompt+短prefix自然续写一次至总L=8192（最多额外6144 tokens）；不是仅对全错组或仅对错答续写。长回答只用于reward。短轨迹末端reward加上 task_score_long − task_score_short，保留原短回答overlong shaping残差。最终仍对原短回答（≤2048）的全部有效response tokens做DAPO/GRPO更新，续写tail不进入actor loss。replace模式动态筛选实际用boundary_acc；虽然配置字段写acc，代码会覆盖，不能误称先按原短acc过滤。',
'两者均保持同版本on-policy生成、完整组筛选及每步256×8条actor轨迹；共同使用非对称clip 0.20/0.28。PPPO关闭dual-clip、overlong shaping；NCBR dual-clip c=3并保留长度410、系数1的overlong buffer。两者均关闭critic、KL reward/loss和entropy bonus。本对比中的PPPO不是未经修改的原论文采样协议。','',
'## 5. 实际训练配置','',table(['字段','PPPO','NCBR H2048/L8192'],[(r['field'],r['PPPO'],r['NCBR']) for r in configrows]),'',
'硬件：同一2节点，每节点8×A100-SXM4-40GB；FSDP world/actor DP=16、SP=1；rollout TP=2，为8个TP组。rollout.nnodes=0表示共置资源语义，不是零节点运行。两者BF16计算、FP32主模型/优化器状态，optimizer offload=true、parameter offload=false；AdamW 1e-6、betas=(0.9,0.999)、weight_decay=.01、constant、无warmup、grad clip=1。mini-batch16问题=128轨迹，每步16次更新，micro/GPU=1，old-logprob micro/GPU=2。',
'PPPO额外：cropped actor、固定线性tile=128、BF16 full reduction、continuation seed=42、并发128、请求chunk512、reward chunk256、timeout600s，optimizer audit关闭。BF16验收使用此前获用户授权的log-prob门槛0.06，其余门槛未放宽；这属于实现/数值路径披露，不等于数学精确相等。NCBR续写并发128、请求chunk512、reward chunk256、timeout600s、seed42。data.seed两者均为null，actor dataloader/FSDP seed为42，不将null宣称为显式统一全局seed。',
'当前PPPO实际save_freq=1（后续用户要求改变保存方式后的正式run），不是最初计划中的50。NCBR save_freq=50；PPPO更多checkpoint I/O是协议开销差异。PPPO总日程仍300，实际仅完成20后暂停，不能线性推算或冒充300步完成结果。','',
'## 6. 数据与在线验证差异','',
'二者源数据为相同历史 DAPO-Math Parquet（SHA256见sources.json）。历史源共有1,791,700行，含重复副本；PPPO按唯一题目身份抽取256题holdout并排除全部副本，训练集1,764,500行，prompt过滤后1,763,700行。NCBR没有排除该holdout，因此不是严格相同训练样本。划分seed42、holdout SHA a0214577119338638f63adb4638130502d5d638928c7159c6b8837c6f9755469。',
'PPPO在线每10步评估独立holdout（Step0 baseline）；NCBR历史在线每10步使用AIME24/25，不能将这两条在线曲线直接比较。PPPO未把最终AIME、AMC、MATH500测试成绩用于eta调度。MATH500精确规范化文本与训练/holdout匹配为0，仅证明该匹配口径，不能证明不存在语义或预训练重叠。',
'NCBR原始1–20步来自Base，后从其Step20完整状态恢复继续到300；PPPO本次从同一Base全新初始化。完整配置副本包括NCBR父段与恢复段，保留路径、resume和验证差异。','',
'## 7. 统一8K离线结果','',
'评测：H_eval=8192，每题32样本，temperature=1、top_p=1、top_k=-1、正常EOS；同Base tokenizer、Answer:模板、master seed42派生请求seed、历史math_dapo verifier。Pass@1估计=每题c/32再平均，即Avg@32；Pass@32=至少一次正确的题目比例。SE按题目计算，单位为百分点。历史verifier为规范化字符串匹配，不是通用符号等价判断。',
 table(['题库','PPPO Avg@32±SE','NCBR Avg@32±SE','差值pp','PPPO Pass@32±SE','NCBR Pass@32±SE','差值pp','平均长度 P/N'],[(r['benchmark'],f"{r['PPPO_avg32_pct']:.2f}±{r['PPPO_avg32_se_pp']:.2f}",f"{r['NCBR_avg32_pct']:.2f}±{r['NCBR_avg32_se_pp']:.2f}",f"{r['avg32_delta_pp']:+.2f}",f"{r['PPPO_pass32_pct']:.2f}±{r['PPPO_pass32_se_pp']:.2f}",f"{r['NCBR_pass32_pct']:.2f}±{r['NCBR_pass32_se_pp']:.2f}",f"{r['pass32_delta_pp']:+.2f}",f"{r['PPPO_mean_tokens']:.1f}/{r['NCBR_mean_tokens']:.1f}") for r in bench]),'',
 table(['题库','指标','PPPO−NCBR(pp)','配对bootstrap 95%区间'],[(r['benchmark'],r['metric'],f"{r['delta_pp']:+.3f}",str([round(x,3) for x in r['ci95_pp']])) for r in paired]),'',
'![离线结果](offline_comparison.png)','',
'NCBR AIME24/25/AMC23使用已冻结历史Step300成绩，已核验权重SHA和原始评分Parquet SHA；MATH500为本轮新生成。PPPO全部四题库为本轮Step20新生成。二者报告所需结果均已PASS，无须等待不在本报告范围内的其他评测组。',
'PPPO在MATH500的Avg@32降低但Pass@32提高，说明这批32次样本的解题覆盖更广、平均成功次数更少；这不能单凭汇总指标归因于某个算法机制。AIME样本量仅30题，配对区间与训练种子不确定性应一并考虑。','',
'## 8. 离线评测开销与交付','']
pe=read(EVAL/'pppo_s20/status.json');ne=read(EVAL/'ncbr_s300/status.json');diag=read(EVAL/'diagnostic/status.json');fail=read(EVAL/'failed_attempt_costs.json')
lines += [table(['本轮评测范围','GPU-hours（head 8卡保留口径）'],[['PPPO四题库19200回答',f'{pe["allocated_gpu_hours"]:.6f}'],['NCBR MATH500 16000回答',f'{ne["allocated_gpu_hours"]:.6f}'],['独立诊断256回答',f'{diag["allocated_gpu_hours"]:.6f}'],['失败fork启动（0回答）',f'{fail["failed_attempt_allocated_gpu_hours"]:.6f}']]),
'该评测timer包括加载/生成/评分，worker节点闲置不计作活跃评测GPU。NCBR历史AIME/AMC成本未并入本轮；四题库与单MATH500开销不是同范围比较，不能直接计算效率倍数。失败初始化已保留现场，spawn修复只改变执行方式，未改采样协议。',
'附件包含完整配置、字段差异、逐步CSV/JSONL、图表、离线结果、配对bootstrap和SHA256来源清单。训练保持暂停，不自动恢复。']
text='\n\n'.join(lines)+'\n';(OUT/'REPORT_ZH.md').write_text(text)
try:
 import markdown
 body=markdown.markdown(text,extensions=['tables','fenced_code'])
except ImportError:body='<pre>'+html.escape(text)+'</pre>'
(OUT/'REPORT_ZH.html').write_text('<!doctype html><meta charset="utf-8"><title>PPPO vs NCBR</title><style>body{max-width:1200px;margin:auto;font:16px/1.6 sans-serif}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #bbb;padding:5px}img{max-width:100%}pre{white-space:pre-wrap}</style>'+body)
sources=[ppath,npath,parentpath,RUN/'training_metrics.jsonl',RUN/'schedule_events.jsonl',RUN/'pause_step20_receipt.json',COST/'steps.csv',COST/'summary.csv',HIST/'gpu_cost_summary.json',HIST/'boundary_return_mechanism_cost_steps_1_300.csv',EVAL/'pppo_s20/scores.json',EVAL/'ncbr_s300/scores.json',EVAL/'checkpoint_preflight.json',REPO/'experiments/pppo_ncbr_matched/provenance/ncbr_step300_h8192.json',ROOT/'outputs/pppo_ncbr_matched_16xa10040g_s300/data/split_manifest.json']
write(OUT/'sources.json',[{'path':str(p),'sha256':sha(p)} for p in sources]);write(OUT/'verification.json',{'PPPO_steps':20,'PPPO_updates_measured':320,'NCBR_steps':300,'NCBR_updates':'4800 theoretical; historical counter unavailable','evaluations':'both requested arms PASS','source_weight_and_historical_scores':'SHA256 verified','only_NCBR_H2048_L8192':True,'training_state':read(RUN/'training_status.json')['state']})
shutil.copyfile(Path(__file__), OUT/'build_report.py')
with zipfile.ZipFile(OUT/'report_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():
  if p.is_file() and p.suffix!='.zip' and p.name!='delivery_receipt.json':z.write(p,p.name)
print(json.dumps({'output':str(OUT),'costs':summary,'results':bench},ensure_ascii=False,indent=2))
