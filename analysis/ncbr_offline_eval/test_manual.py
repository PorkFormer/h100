import argparse,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import evaluate as ev
class ManualTests(unittest.TestCase):
    def args(self,root):
        from safetensors.numpy import save_file
        import numpy as np
        hf=root/'hf';hf.mkdir();(hf/'config.json').write_bytes((ev.TOKENIZER/'config.json').read_bytes())
        save_file({'synthetic':np.ones((2,2),dtype=np.float32)},str(hf/'model.safetensors'))
        return argparse.Namespace(checkpoint=hf,output=root/'eval',model_id='synthetic_s300',step=300,label='synthetic',node='10.8.191.127',ray_address='10.8.191.127:6397')
    def test_prepare_freeze_and_no_gpu_launch(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.args(Path(d))
            with patch.object(ev.subprocess,'Popen') as popen:out=ev.prepare(args);popen.assert_not_called()
            ev.verify_prepared(out)
            self.assertEqual(ev.read(out/'generation_config.json')['expected_per_horizon'],3200)
            self.assertEqual(len(ev.read(out/'checkpoint_manifest.json')['checkpoints']),1)
            self.assertEqual(ev.read(out/'status.json')['stage'],'PREPARED')
            with self.assertRaises(ValueError):ev.prepare(args)
            (out/'worker.py').write_text('changed')
            with self.assertRaises(ValueError):ev.verify_prepared(out)
    def test_start_duplicate_rejected_and_unique_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.args(Path(d));out=ev.prepare(args);a=ev.read(out/'run_config.json')
            args.output=Path(d)/'eval2';other=ev.prepare(args);b=ev.read(other/'run_config.json')
            self.assertNotEqual(a['cache_root'],b['cache_root']);self.assertNotEqual(a['namespace'],b['namespace'])
            with patch.object(ev.subprocess,'Popen',return_value=Mock(pid=12345)) as popen:
                self.assertEqual(ev.start(out),0);self.assertEqual(popen.call_count,1)
                with self.assertRaises(ValueError):ev.start(out)
                self.assertEqual(popen.call_count,1)
    def test_bad_architecture_missing_shard_and_changed_weight(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.args(Path(d));cfg=ev.read(args.checkpoint/'config.json');cfg['hidden_size']=1;ev.write(args.checkpoint/'config.json',cfg)
            with self.assertRaises(ValueError):ev.prepare(args)
            (args.checkpoint/'config.json').write_bytes((ev.TOKENIZER/'config.json').read_bytes())
            ev.write(args.checkpoint/'model.safetensors.index.json',{'weight_map':{'synthetic':'missing.safetensors'}})
            with self.assertRaises(ValueError):ev.prepare(args)
            (args.checkpoint/'model.safetensors.index.json').unlink();out=ev.prepare(args)
            with (args.checkpoint/'model.safetensors').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):ev.verify_prepared(out)
if __name__=='__main__':unittest.main()
