import unittest
from lab.features.e_bike.training.scripts.evaluation_audit import optimal_pairs, image_metrics


def obj(box, cls=0):
    return {'xyxy':box, 'class_id':cls}


class EvaluationAuditTest(unittest.TestCase):
    def test_overlapping_gt_can_accept_duplicate_boxes(self):
        # Box matching is a geometry criterion, not proof of visual identities.
        m=image_metrics({'ground_truth':[obj([0,0,10,10]),obj([2,0,12,10])],
                         'predictions':[obj([1,0,11,10]),obj([1,0,11,10])]})
        self.assertTrue(m['strict_instance_exact'])
        self.assertTrue(m['strict_exact_with_assignment_ambiguity'])
        self.assertEqual(m['predictions_compatible_with_multiple_GT'],2)

    def test_greedy_can_lose_valid_pair_in_crowd(self):
        gt = [obj([0,0,10,10]), obj([4,0,14,10])]
        pred = [obj([1,0,11,10]), obj([-2,0,8,10])]
        m = image_metrics({'ground_truth':gt, 'predictions':pred})
        self.assertEqual((m['greedy_pairs'],m['maximum_pairs']), (1,2))

    def test_equal_counts_do_not_prove_correct_instances(self):
        m = image_metrics({'ground_truth':[obj([0,0,10,10])],
                           'predictions':[obj([20,20,30,30])]})
        self.assertTrue(m['count_exact'])
        self.assertFalse(m['strict_instance_exact'])

    def test_posture_is_separate_from_geometry(self):
        m = image_metrics({'ground_truth':[obj([0,0,10,10])],
                           'predictions':[obj([0,0,10,10],1)]})
        self.assertTrue(m['geometry_exact'])
        self.assertEqual(m['matched_pose_errors'],1)
        self.assertFalse(m['strict_instance_exact'])

    def test_empty_images_and_missing_predictions(self):
        self.assertEqual(optimal_pairs([],[],.5), [])
        self.assertTrue(image_metrics({'ground_truth':[],'predictions':[]})['strict_instance_exact'])
        self.assertEqual(image_metrics({'ground_truth':[obj([0,0,10,10])],'predictions':[]})['entity_fn'],1)


if __name__ == '__main__':
    unittest.main()
