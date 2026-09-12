import unittest
from analyze import paired,weighted_percentile
class StatisticsTests(unittest.TestCase):
    def row(self,t,signature='same'):return dict(eligibility_signature=signature,audit='PASS',continuation_seconds=t)
    def test_known_speedup(self):
        r=paired([(self.row(12),self.row(10))]*4);self.assertEqual(r['status'],'PASS');self.assertEqual(r['ci95'],[1.2,1.2])
    def test_mismatch_blocks(self):
        with self.assertRaises(AssertionError):paired([(self.row(12),self.row(10,'different'))]*4)
    def test_no_gain(self):self.assertEqual(paired([(self.row(10),self.row(10))]*4)['status'],'FAIL')
    def test_weighted_occupancy(self):self.assertEqual(weighted_percentile([0,8],[99,1],.5),0)
    def test_insufficient_pairs(self):self.assertEqual(paired([(self.row(12),self.row(10))])['status'],'FAIL')
if __name__=='__main__':unittest.main()
