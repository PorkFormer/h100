"""Independent full-response scoring and six-cell prompt bootstrap report."""
import csv,json,sys,time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from common import ROOT,VERL,TOKENIZER,read_rows,atomic_json,sha,models

def main():
    sys.path.insert(0,str(VERL))
    from verl.utils.reward_score.math_dapo import compute_score
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,trust_remote_code=False)
    assert len(models())==1
    scores=[];errors=[];repairs=[];raw_hash={}
    for h in (2048,8192):
        path=ROOT/f'formal/generation/h{h}/raw_generations.jsonl';raw_hash[str(h)]=sha(path)
        rows=read_rows(h,complete=True);assert len(rows)==3200
        for r in rows:
            try:
                text=tokenizer.decode(r['response_token_ids'],skip_special_tokens=True)
                assert text==r['response_text'],'text/token mismatch'
                v=compute_score(solution_str=text,ground_truth=r['ground_truth'])
                assert isinstance(v,dict) and 'acc' in v and 'score' in v
                acc=bool(v['acc']);score=float(v['score']);pred=None if v.get('pred') is None else str(v['pred'])
                assert score==(1.0 if acc else -1.0),'invalid verifier score'
                if r.get('scoring_error'):
                    # Two fresh successful calls are required when initial scoring raised.
                    again=compute_score(solution_str=text,ground_truth=r['ground_truth'])
                    assert bool(again['acc'])==acc and float(again['score'])==score
                    repairs.append(dict(trajectory_id=r['trajectory_id'],original_error=r['scoring_error'],correct=acc,score=score))
                else:assert acc==r['reward_acc'] and score==r['reward_score'] and pred==r['reward_pred'],'stored score mismatch'
                scores.append({**{k:r[k] for k in ['trajectory_id','model_id','checkpoint_sha256','horizon','benchmark','prompt_id','rollout_index','stable_seed','response_token_count','hit_cap']},'correct':acc,'score':score,'pred':pred})
            except Exception as exc:errors.append(dict(trajectory_id=r['trajectory_id'],error=repr(exc)))
        assert sha(path)==raw_hash[str(h)],'generation changed during scoring'
    atomic_json(ROOT/f'formal/scoring_audit_{time.time_ns()}.json',dict(errors=errors,repaired_initial_errors=repairs))
    if errors:raise RuntimeError(f'{len(errors)} scoring errors (not counted as wrong answers)')
    assert len(scores)==6400
    pq.write_table(pa.Table.from_pylist(scores),ROOT/'formal/trajectory_scores.parquet',compression='zstd')
    metrics=[]
    for h in (2048,8192):
        for b,n in [('AIME2024',30),('AIME2025',30),('AMC23',40)]:
            rs=[r for r in scores if r['horizon']==h and r['benchmark']==b]
            ids=sorted({r['prompt_id'] for r in rs});assert len(ids)==n and len(rs)==n*32
            per=np.array([[r['correct'] for r in rs if r['prompt_id']==pid] for pid in ids],dtype=float);assert per.shape==(n,32)
            avg=per.mean(axis=1);any_correct=per.max(axis=1)
            ix=np.random.default_rng(20260812).integers(0,n,size=(10000,n))
            lo,hi=np.quantile(avg[ix].mean(axis=1),[.025,.975]);l32,u32=np.quantile(any_correct[ix].mean(axis=1),[.025,.975])
            metrics.append(dict(model_id=models()[0]['model_id'],step=models()[0]['step'],benchmark=b,horizon=h,prompts=n,trajectories=len(rs),pass_at_1=float(avg.mean()),pass_at_1_ci_low=float(lo),pass_at_1_ci_high=float(hi),pass_at_32=float(any_correct.mean()),pass_at_32_ci_low=float(l32),pass_at_32_ci_high=float(u32),mean_response_length=float(np.mean([r['response_token_count'] for r in rs])),truncation_rate=float(np.mean([r['hit_cap'] for r in rs]))))
    with (ROOT/'formal/metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    pq.write_table(pa.Table.from_pylist(metrics),ROOT/'formal/metrics.parquet')
    atomic_json(ROOT/'formal/scoring_receipt.json',dict(passed=True,rows=6400,units=6,verifier_errors=0,stored_reward_mismatches=0,repaired_initial_errors=len(repairs),raw_sha256=raw_hash))
    lines=[f"# {models()[0]['arm']}：S{models()[0]['step']} 离线评测",'',f"权重：`{models()[0]['checkpoint_path']}`。仅报告本次指定 checkpoint。",'', 'H2048、H8192 分别独立生成；每题 32 次，共 6400 条。pass@1 为 Avg@32，pass@32 为至少一次正确的题目比例。置信区间按题目 bootstrap 10000 次，seed=20260812。','', '| 数据集 | 回答预算 | pass@1（95% CI） | pass@32（95% CI） | 平均回答长度 | 截断率 |','|---|---:|---:|---:|---:|---:|']
    for m in metrics:lines.append(f"| {m['benchmark']} | {m['horizon']} | {m['pass_at_1']:.2%} ({m['pass_at_1_ci_low']:.2%}, {m['pass_at_1_ci_high']:.2%}) | {m['pass_at_32']:.2%} ({m['pass_at_32_ci_low']:.2%}, {m['pass_at_32_ci_high']:.2%}) | {m['mean_response_length']:.1f} | {m['truncation_rate']:.2%} |")
    lines+=['','生成与独立复算评分已完成；所选分片 worker 均正常退出，GPU 进程已释放，Ray 预留已移除，共享 Ray 保留。控制器最终退出状态见 `status.json` 和 `driver_exit.json`。',f'生成时评分异常 {len(repairs)} 条，已保留异常并以两次成功评分修复；未将错误填为答错。','', '离线回答预算独立于训练预算。未运行其他 checkpoint 或历史对照，未选择最佳 checkpoint，未发送邮件。']
    (ROOT/'formal/REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
