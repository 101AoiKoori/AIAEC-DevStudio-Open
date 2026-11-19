import unittest
import numpy as np
from lab.features.e_bike.training.scripts.input_scale_experiment import diagnose_gt,translate_boxes,crop_rectangle


class CandidateDiagnosticTest(unittest.TestCase):
    def pass_data(self,score=.8,local=False,border=False):
        p={'xyxy':[0,0,20,20],'class_id':0,'confidence':.9}
        return {'boxes':np.array([[0,0,20,20]]),'scores':np.array([score]),
                'after_local_nms':[p] if local else [],'after_border_filter':[p] if border else [],'postprocess_count':1}

    def reason(self,p,final=None,matches=None):
        return diagnose_gt([{'xyxy':[0,0,20,20],'class_id':0}],[p],final or [],matches or [])[0]

    def test_high_score_missing_after_local_nms(self):
        self.assertEqual(self.reason(self.pass_data())['reason'],'local_nms')

    def test_low_score_is_not_blame_for_nms(self):
        self.assertEqual(self.reason(self.pass_data(.1))['reason'],'below_confidence_025')

    def test_nms_trace_identifies_retained_overlapping_box(self):
        p=self.pass_data()
        p['after_local_nms']=[{'xyxy':[8,0,28,20],'class_id':0,'confidence':.9}]
        p['after_border_filter']=p['after_local_nms']
        event=self.reason(p)
        self.assertEqual(event['reason'],'local_nms')
        self.assertEqual(event['evidence_by_pass'][0]['high_score_overlap_evidence'][0]['raw_id'],0)

    def test_candidate_floor_is_explicit(self):
        self.assertEqual(self.reason(self.pass_data(.005))['reason'],'no_candidate_above_001')

    def test_crop_filter_and_cross_view_nms_are_separate(self):
        self.assertEqual(self.reason(self.pass_data(local=True))['reason'],'crop_border_filter')
        self.assertEqual(self.reason(self.pass_data(local=True,border=True))['reason'],'fusion_nms')

    def test_pose_error_does_not_become_localization_miss(self):
        event=self.reason(self.pass_data(local=True,border=True),[{'xyxy':[0,0,20,20],'class_id':1}],[(0,0,1.)])
        self.assertEqual(event['reason'],'detected');self.assertTrue(event['pose_wrong'])

    def test_competing_gt_assignment_is_explicit(self):
        self.assertEqual(self.reason(self.pass_data(local=True,border=True),[{'xyxy':[0,0,20,20],'class_id':0}])['reason'],'one_to_one_competition')

    def test_crop_coordinates_restore_to_original(self):
        np.testing.assert_array_equal(translate_boxes([[1,2,30,40]],100,200),[[101,202,130,240]])
        self.assertEqual(crop_rectangle((480,640,3),(0,.5,.65,.88)),[0,240,416,422])


if __name__=='__main__':unittest.main()
