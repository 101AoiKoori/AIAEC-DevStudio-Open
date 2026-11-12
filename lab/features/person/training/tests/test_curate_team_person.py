import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from curate_team_person import bbox_from_mask, preview_exclusion, teammate_reasons, yolo_row


class CurateTeamPersonTest(unittest.TestCase):
    def test_unknown_negative_occlusion_and_crop_are_excluded(self):
        self.assertEqual([], teammate_reasons({'label': 0, 'aug': {'flip': False}}, 5))
        reasons = teammate_reasons({'label': 2, 'aug': {'occlude': True, 'crop': True, 'flip': False}}, 30)
        self.assertEqual(4, len(reasons))
        self.assertIn('background_may_contain_unlabelled_standees', reasons)
        self.assertIn('occlusion_not_reflected_in_bbox', reasons)
        self.assertIn('crop_visibility_unverified', reasons)
        self.assertIn('negative_requires_full_resolution_review', reasons)

    def test_mirrored_or_unknown_person_illustration_is_excluded(self):
        self.assertIn('mirrored_person_illustration', teammate_reasons({'label': 0, 'aug': {'flip': True}}, 5))
        for aug in ({}, {'flip': None}, {'flip': 0}, {'flip': 'false'}, None):
            with self.subTest(aug=aug):
                self.assertIn('mirror_status_unknown', teammate_reasons({'label': 0, 'aug': aug}, 5))
        reasons = teammate_reasons({'label': 2, 'aug': {'flip': True, 'crop': True}}, 30)
        self.assertIn('mirrored_person_illustration', reasons)
        self.assertIn('background_may_contain_unlabelled_standees', reasons)
        self.assertIn('crop_visibility_unverified', reasons)
        self.assertIn('negative_requires_full_resolution_review', reasons)

    def test_gray_background_preview_is_an_exclusion_not_a_training_pair(self):
        row = {'person_class': 'community', 'source_group': 'shared',
               'vehicle_video': 'video.mp4', 'person_source': 'person.png'}
        result = preview_exclusion(row, 'datasets/exports/e_bike/preview.png')
        self.assertEqual(0, result['class_id'])
        self.assertEqual(['gray_background_preview_excluded_from_real_scene_candidates'], result['reasons'])
        self.assertNotIn('image', result)
        self.assertNotIn('label', result)
        with self.assertRaises(ValueError):
            preview_exclusion({**row, 'person_class': 'unknown'}, 'preview.png')

    def test_mask_bbox_and_yolo_coordinate_contract(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:6, 4:10] = True
        self.assertEqual([4, 2, 10, 6], bbox_from_mask(mask))
        self.assertEqual('1 0.350000 0.400000 0.300000 0.400000\n', yolo_row(1, [4, 2, 10, 6], 20, 10))
        with self.assertRaises(ValueError):
            yolo_row(0, [0, 0, 21, 1], 20, 10)


if __name__ == '__main__':
    unittest.main()
