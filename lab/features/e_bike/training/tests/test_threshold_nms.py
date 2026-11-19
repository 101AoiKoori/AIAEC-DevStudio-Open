import unittest
import numpy as np
from lab.features.e_bike.training.scripts.threshold_nms_experiment import gaussian_soft_nms,compare,specifications


class ThresholdNmsTest(unittest.TestCase):
    def test_gaussian_decay_removes_duplicate_without_changing_inputs(self):
        scores=np.array([.9,.8,.7])
        kept=gaussian_soft_nms([[0,0,10,10],[0,0,10,10],[20,0,30,10]],scores,.2)
        self.assertEqual([i for i,_ in kept],[0,2])
        np.testing.assert_array_equal(scores,[.9,.8,.7])

    def test_soft_nms_reorders_by_decayed_scores(self):
        kept=gaussian_soft_nms([[0,0,10,10],[0,0,10,10],[20,0,30,10]],[.9,.8,.7],.1)
        self.assertEqual([i for i,_ in kept],[0,2,1])
        self.assertLess(kept[-1][1],.2)

    def test_false_prediction_swap_is_not_hidden_by_constant_fp_total(self):
        truth=[{'class_id':0,'xyxy':[0,0,10,10]}]
        base={'ground_truth':truth,'predictions':[{'class_id':0,'raw_id':1,'xyxy':[20,0,30,10]}]}
        row={'ground_truth':truth,'predictions':[{'class_id':0,'raw_id':2,'xyxy':[0,0,10,10]},
                                                   {'class_id':0,'raw_id':3,'xyxy':[40,0,50,10]}]}
        audit=compare(base,row)
        self.assertEqual(audit['recovered_gt_indices'],[0])
        self.assertEqual(audit['new_false_raw_ids'],[3])
        self.assertEqual(audit['removed_false_raw_ids'],[1])

    def test_parameter_grid_contains_original_reference(self):
        specs=specifications()
        self.assertEqual(len(specs),28)
        self.assertEqual(specs['hard_c025_i030'],{'method':'hard','confidence':.25,'iou':.3})


if __name__=='__main__':unittest.main()
