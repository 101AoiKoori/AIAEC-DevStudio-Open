import unittest
from lab.features.e_bike.training.scripts.candidate_verifier_experiment import candidate_label,foreground,family_key,family_split


class CandidateVerifierTest(unittest.TestCase):
    def test_opt_in_completeness_can_reject_vehicle_part(self):
        objects=[{'class_id':0,'xyxy':[0,0,100,100]}]
        self.assertEqual(candidate_label([10,10,30,30],[8,8,32,32],objects), (None,None))
        self.assertEqual(candidate_label([10,10,30,30],[8,8,32,32],objects,reject_partial_below_iou=.3), ('background',None))

    def test_completeness_keeps_borderline_candidates_unknown(self):
        objects=[{'class_id':0,'xyxy':[0,0,100,100]}]
        self.assertEqual(candidate_label([0,0,60,60],[0,0,65,65],objects,reject_partial_below_iou=.3), (None,None))

    def test_partial_vehicle_is_ignored_not_background(self):
        objects=[{'class_id':0,'xyxy':[10,10,30,30]}]
        self.assertEqual(candidate_label([0,0,15,15],[0,0,17,17],objects),(None,None))

    def test_background_checks_actual_padded_crop(self):
        objects=[{'class_id':0,'xyxy':[20,20,40,40]}]
        self.assertEqual(candidate_label([0,0,18,18],[0,0,22,22],objects),(None,None))
        self.assertEqual(candidate_label([0,0,10,10],[0,0,12,12],objects),('background',None))

    def test_overlapping_opposite_postures_are_not_forced_into_one_label(self):
        objects=[{'class_id':0,'xyxy':[0,0,20,20]},{'class_id':1,'xyxy':[4,0,24,20]}]
        self.assertEqual(candidate_label([0,0,20,20],[0,0,22,22],objects),(None,None))

    def test_high_iou_single_vehicle_gets_posture(self):
        self.assertEqual(candidate_label([0,0,20,20],[0,0,22,22],[{'class_id':1,'xyxy':[0,0,20,20]}]),('fallen',0))

    def test_same_mirror_pair_gets_same_internal_split(self):
        a={'source_record':{'state_id':'low_000073','background':{'id':'bg'}}}
        b={'source_record':{'state_id':'low_000074','background':{'id':'bg'}}}
        self.assertEqual(family_key(a),family_key(b))
        self.assertEqual(family_split(family_key(a)),family_split(family_key(b)))

    def test_background_argmax_is_rejected_even_when_foreground_sum_exceeds_half(self):
        self.assertFalse(foreground({'background':.45,'fallen':.3,'upright':.25}))
        self.assertTrue(foreground({'background':.1,'fallen':.2,'upright':.7}))


if __name__=='__main__':unittest.main()
