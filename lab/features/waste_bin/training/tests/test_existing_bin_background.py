import unittest
import random

import numpy as np

from datasets.scripts.common.review.placement import GroundPlacement
from datasets.scripts.waste_bin_state import generate_waste_v4_grouped as grouped


class ExistingBinBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.review = {'image_size': [640, 360], 'label': {'presence': 'present',
                       'completeness': 'complete', 'scope': {'kind': 'frame'},
                       'instances': [{'class_id': 1, 'lid_state': 'closed', 'color': 'gray',
                                      'object_id': 'bin_gray', 'xyxy': [100, 40, 200, 180]},
                                     {'class_id': 0, 'lid_state': 'open', 'color': 'blue',
                                      'object_id': 'bin_blue', 'xyxy': [300, 40, 400, 190]}]}}

    def test_existing_states_are_mapped_and_protected_in_letterbox(self):
        existing = grouped.existing_bin_labels(self.review)
        self.assertEqual(existing[0]['bbox_xyxy'], [100, 100, 200, 240])
        self.assertEqual(existing[0]['state'], 'closed')
        self.assertEqual(existing[1]['state'], 'open')
        grouped.reject_existing_bin_overlap([[10, 300, 50, 380]], existing)
        with self.assertRaisesRegex(ValueError, 'obscures a labelled background bin'):
            grouped.reject_existing_bin_overlap([[150, 200, 250, 300]], existing)

    def test_existing_black_bin_uses_state_alias_without_becoming_a_render_color(self):
        self.review['label']['instances'] = [
            {'class_id': 0, 'class_name': 'open', 'state': 'open', 'bin_color': 'black',
             'kind': 'waste_bin', 'object_id': 'original_black_bin', 'xyxy': [100, 40, 200, 180]},
        ]
        retained = grouped.existing_bin_labels(self.review)
        self.assertEqual(retained[0]['bin'], 'black')
        self.assertEqual(retained[0]['state'], 'open')
        self.assertEqual(retained[0]['bbox_xyxy'], [100, 100, 200, 240])
        self.review['label']['instances'][0]['lid_state'] = 'closed'
        with self.assertRaisesRegex(ValueError, 'source-pixel box'):
            grouped.existing_bin_labels(self.review)

    def test_saved_ground_and_pixels_receive_the_same_padding(self):
        candidate = np.ones((360, 640), dtype=bool)
        assessment = GroundPlacement('candidate', 'fixture', candidate, np.zeros_like(candidate))
        mapped = grouped.output_ground(assessment, [640, 360])
        self.assertEqual(mapped.candidate_mask.shape, (480, 640))
        self.assertFalse(mapped.candidate_mask[0].any())
        self.assertTrue(mapped.candidate_mask[60:420].all())
        frame = grouped.output_frame(np.ones((360, 640, 3), np.uint8))
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(int(frame[60, 0, 0]), 1)
        self.assertEqual(int(frame[59, 0, 0]), 0)

    def test_higher_resolution_source_scales_labels_and_checks_source_contact(self):
        self.review['image_size'] = [1280, 720]
        self.review['label']['instances'][0]['xyxy'] = [200, 80, 400, 360]
        self.review['label']['instances'][1]['xyxy'] = [600, 80, 800, 380]
        self.assertEqual(grouped.existing_bin_labels(self.review)[0]['bbox_xyxy'],
                         [100, 100, 200, 240])
        candidate = np.ones((720, 1280), dtype=bool)
        assessment = GroundPlacement('candidate', 'fixture', candidate, np.zeros_like(candidate))
        mapped = grouped.output_ground(assessment, [1280, 720])
        self.assertEqual(mapped.contact_status((320, 300), contact_width=20), 'candidate')
        self.assertEqual(mapped.contact_status((320, 20), contact_width=20), 'unknown')
        self.assertEqual(grouped.output_frame(np.ones((720, 1280, 3), np.uint8)).shape,
                         (480, 640, 3))
        self.assertEqual(grouped.output_polygon(
            [[0, 400], [1279, 400], [1279, 719], [0, 719]], [1280, 720]),
            [[0, 260], [639, 260], [639, 419], [0, 419]])

    def test_incomplete_or_conflicting_old_state_is_rejected(self):
        self.review['label']['instances'][0]['class_id'] = 0
        with self.assertRaisesRegex(ValueError, 'source-pixel box'):
            grouped.existing_bin_labels(self.review)
        self.review['label']['instances'] = []
        with self.assertRaisesRegex(ValueError, 'needs complete visible instances'):
            grouped.existing_bin_labels(self.review)
        self.review['label']['presence'] = 'absent'
        self.assertEqual(grouped.existing_bin_labels(self.review), [])

    def test_contact_can_use_reviewed_ground_above_legacy_row_410(self):
        floor = np.zeros((480, 640), np.uint8)
        floor[280:405] = 1
        background = {'ground_polygon_xy': [[0, 280], [639, 280], [639, 404], [0, 404]],
                      'existing_bins': [{'bbox_xyxy': [300, 80, 500, 270]}]}
        crop = np.zeros((140, 65, 3), np.uint8)
        _, _, _, left, top, width, height, _, _ = grouped.ground_placement(
            crop, {'blue': np.ones((140, 65), bool)},
            [(background, np.zeros((480, 640, 3), np.uint8), floor)], random.Random(7))
        self.assertLess(top + height, 410)
        grouped.reject_existing_bin_overlap([[left, top, left+width, top+height]],
                                            background['existing_bins'])


if __name__ == '__main__':
    unittest.main()
