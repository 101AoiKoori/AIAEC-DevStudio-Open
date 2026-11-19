import unittest
from lab.features.e_bike.training.scripts.data_ablation_experiment import choose_additions,error_summary,paired_errors,parent_frames


class DataAblationTest(unittest.TestCase):
    def fixture(self):
        old={'records':[{'image':f'old{i}','split':'train'} for i in range(32)]}
        focused={'records':[]}
        for kind in ('mined_vehicle_free_background','complete_real_crop'):
            focused['records'] += [{'image':f'{kind}{i}','split':'train','condition':kind} for i in range(20)]
        for bg in range(4):
            for family in range(2):
                focused['records'] += [{'image':f'{bg}_{family}_{i}','split':'train','condition':'focused_composite',
                    'source_record':{'research_details':{'family_id':f'bg{bg}_{family:03d}'}}} for i in range(4)]
        focused['records'].append({'image':'heldout','split':'holdout','condition':'complete_real_crop'})
        return old,focused

    def test_equal_quotas_keep_synthetic_family_quartets(self):
        old,focused=self.fixture();result=choose_additions(old,focused,16,42)
        self.assertEqual({len(v) for v in result.values()},{16})
        self.assertNotIn('heldout',[r['image'] for rows in result.values() for r in rows])
        groups={r['source_record']['research_details']['family_id'] for r in result['synthetic']}
        self.assertEqual(len(groups),4)
        self.assertEqual(result,choose_additions(old,focused,16,42))

    def test_pose_confusion_is_separated_from_localization(self):
        row={'source_image':'example','group':'original_real',
             'ground_truth':[{'class_id':0,'xyxy':[0,0,10,10]},{'class_id':1,'xyxy':[20,0,30,10]}],
             'predictions':[{'class_id':1,'xyxy':[0,0,10,10]},{'class_id':1,'xyxy':[20,0,30,10]}]}
        summary=error_summary([row])
        self.assertEqual(summary['matched_box_posture_errors_by_group'],{'original_real':1})
        self.assertEqual(summary['matched_posture_confusion_gt_rows_pred_columns'],[[0,1],[0,1]])
        self.assertEqual(summary['fully_localized_images_with_posture_errors'],['example'])

    def test_incomplete_synthetic_family_is_rejected(self):
        old,focused=self.fixture()
        focused['records']=[r for r in focused['records'] if not (r['image'].startswith('0_') and r['image'].endswith('_3'))]
        with self.assertRaisesRegex(ValueError,'quartet'):
            choose_additions(old,focused,16,42)

    def test_pair_audit_separates_recovery_from_pose_fix(self):
        gt=[{'class_id':0,'xyxy':[0,0,10,10]},{'class_id':1,'xyxy':[20,0,30,10]}]
        before={'source_image':'same','group':'original_real','ground_truth':gt,'predictions':[{'class_id':1,'xyxy':[0,0,10,10]}]}
        after={**before,'predictions':[{'class_id':0,'xyxy':[0,0,10,10]},{'class_id':1,'xyxy':[20,0,30,10]}]}
        audit=paired_errors(before,after)
        self.assertEqual(audit['recovered_gt_indices'],[1])
        self.assertEqual(audit['fixed_posture_gt_indices'],[0])

    def test_parent_resampling_preserves_crop_order_and_repetition(self):
        old={'records':[{'image':'trainA','split':'train'},{'image':'trainB','split':'train'},{'image':'val','split':'val'}]}
        crops=[{'source_record':{'research_details':{'parent_image':p}}} for p in ('trainB','trainA','trainB')]
        self.assertEqual([r['image'] for r in parent_frames(old,crops)],['trainB','trainA','trainB'])


if __name__=='__main__':unittest.main()
