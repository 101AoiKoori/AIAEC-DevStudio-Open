import unittest
from lab.features.e_bike.training.scripts.report_pose_counts import pose_counts


class PoseCountReportTest(unittest.TestCase):
    def row(self,truth,prediction,missed=(),false=()):
        return {'source_image':'sample.png','ground_truth':[{'class_id':c} for c in truth],
                'predictions':[{'class_id':c} for c in prediction],
                'missed_gt_indices':list(missed),'false_prediction_indices':list(false)}

    def test_same_total_with_wrong_postures_does_not_pass(self):
        report=pose_counts([self.row([0,1],[0,0],missed=[1],false=[1])])
        self.assertEqual(report['exact_images'],0)
        self.assertEqual(report['mean_absolute_pose_count_error'],2)

    def test_compensating_miss_and_false_positive_remains_visible(self):
        report=pose_counts([self.row([0,1],[0,1],missed=[1],false=[1])])
        self.assertEqual(report['exact_images'],1)
        self.assertEqual(report['all_instances_matched_images'],0)

    def test_duplicate_evaluation_images_are_rejected(self):
        row=self.row([0],[0])
        with self.assertRaisesRegex(ValueError,'Repeated'):
            pose_counts([row,row])


if __name__=='__main__':unittest.main()
