import copy,unittest
from workload import validate
class EligibilityTests(unittest.TestCase):
    def row(self):return dict(request_id='a',input_token_ids=[2]+[3]*2048,prompt_token_ids=[2],prefix_token_ids=[3]*2048,policy_version=0,replay_decode_length=2,max_tokens=6144,natural_tail_token_ids=[4,5])
    def test_exact(self):self.assertTrue(validate([self.row()]))
    def test_duplicate(self):
        with self.assertRaises(AssertionError):validate([self.row(),self.row()])
    def test_bad_input(self):
        r=self.row();r['input_token_ids'][0]=9
        with self.assertRaises(AssertionError):validate([r])
    def test_bad_length(self):
        for n in [0,1,6145]:
            r=self.row();r['replay_decode_length']=n
            with self.assertRaises(AssertionError):validate([r])
    def test_wrong_version(self):
        r=self.row();r['policy_version']=1
        with self.assertRaises(AssertionError):validate([r])
if __name__=='__main__':unittest.main()
