import unittest
from workload import replay_params
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.r=dict(max_tokens=6144,replay_decode_length=123,seed=42)
        self.p=dict(max_tokens=6144,seed=42,ignore_eos=False,temperature=1.,top_p=1.,top_k=-1,n=1)
    def test_fixed_decode_only_override(self):
        x=replay_params(self.r,self.p);self.assertEqual(x,dict(self.p,max_tokens=123,ignore_eos=True));self.assertFalse(self.p['ignore_eos']);self.assertEqual(self.p['max_tokens'],6144)
    def test_natural_restoration(self):self.assertEqual(replay_params(self.r,self.p,True),self.p)
    def test_seed_mismatch(self):
        with self.assertRaises(AssertionError):replay_params(self.r,dict(self.p,seed=0))
if __name__=='__main__':unittest.main()
