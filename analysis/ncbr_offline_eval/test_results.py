import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'templates'))
import common as cm
import results
class ResultsTest(unittest.TestCase):
    def test_complete_synthetic_pipeline_and_identity_rejection(self):
        from transformers import AutoTokenizer
        import sys
        sys.path.insert(0,str(cm.VERL))
        from verl.utils.reward_score.math_dapo import compute_score
        import pyarrow.parquet as pq
        tokenizer=AutoTokenizer.from_pretrained(cm.TOKENIZER)
        source=next(x['source'] for x in json.loads((Path(__file__).resolve().parent/'input_provenance.json').read_text())['files'] if x['name']=='prompts.parquet')
        ps=pq.read_table(source).to_pylist();model=dict(model_id='test_s300',arm='test',step=300,checkpoint_path='/synthetic',sha256='synthetic')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'prompts.parquet').write_bytes(Path(source).read_bytes());cm.atomic_json(root/'checkpoint_manifest.json',{'checkpoints':[model]})
            for h in (2048,8192):
                rows=[]
                for p in ps:
                    for i in range(32):
                        text='Answer: '+p['ground_truth'] if i==0 else 'No answer.'
                        tokens=tokenizer.encode(text,add_special_tokens=False);text=tokenizer.decode(tokens,skip_special_tokens=True)
                        v=compute_score(solution_str=text,ground_truth=p['ground_truth']);key=f"test_s300|{p['prompt_id']}|{i}"
                        rows.append(dict(model_id=model['model_id'],arm=model['arm'],step=300,checkpoint_path=model['checkpoint_path'],checkpoint_sha256=model['sha256'],horizon=h,prompt_id=p['prompt_id'],rollout_index=i,stable_seed=cm.stable_sampling_seed(42,p['prompt_id'],i),pair_key=key,trajectory_id=f'{key}|h{h}',request_id='offline-'+hashlib.sha256((key+f'|h{h}').encode()).hexdigest()[:24],**{k:p[k] for k in ['benchmark','data_source','prompt_hash','prompt_token_ids','ground_truth','original_dataset_index','extra_info_json']},response_token_ids=tokens,response_token_count=len(tokens),response_text=text,finish_reason='stop',stop_reason=None,hit_cap=False,reward_acc=bool(v['acc']),reward_score=float(v['score']),reward_pred=None if v.get('pred') is None else str(v['pred'])))
                cm.check_rows(rows,h,[model],ps,True)
                broken=[dict(rows[0],stable_seed=-1)]
                with self.assertRaises(ValueError):cm.check_rows(broken,h,[model],ps)
                with self.assertRaises(ValueError):cm.check_rows(rows+[rows[0]],h,[model],ps,True)
                path=root/f'formal/generation/h{h}/raw_generations.jsonl';path.parent.mkdir(parents=True)
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with patch.object(cm,'ROOT',root),patch.object(results,'ROOT',root):results.main()
            ms=pq.read_table(root/'formal/metrics.parquet').to_pylist()
            self.assertEqual(len(ms),6)
            for m in ms:self.assertEqual(m['pass_at_1'],1/32);self.assertEqual(m['pass_at_32'],1.0)
            self.assertTrue((root/'formal/REPORT_ZH.md').exists())
if __name__=='__main__':unittest.main()
