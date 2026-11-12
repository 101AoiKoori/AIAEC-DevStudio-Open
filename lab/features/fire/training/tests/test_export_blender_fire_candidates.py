"""Tests for rejecting unsafe synthetic fire exports."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[5] / 'lab/features/fire/training/scripts'
sys.path.insert(0, str(SCRIPTS))
from export_blender_fire_candidates import decide


def candidate():
    return {'split': 'train', 'status': 'candidate_requires_visual_review',
            'active_buildings': ['A'], 'requested_fire_counts': {'A': 1},
            'instances': [{'kind': 'fire_sticker', 'building': 'A', 'state': 'visible',
                           'pixel_bbox_xyxy': [10, 20, 30, 40]}],
            'occluder': False, 'fire_label': 'labels/fire/train/fire_000001.txt',
            'fire_label_suppression_reason': None}


class FireCandidateExportTests(unittest.TestCase):
    def test_positive_and_empty_negative(self):
        item = candidate()
        self.assertEqual(decide(item, 100), (None, ['0 0.200000 0.300000 0.200000 0.200000']))
        item['requested_fire_counts']['A'] = 0
        item['instances'] = []
        self.assertEqual(decide(item, 100), (None, []))

    def test_occlusion_and_truncation_are_not_negative(self):
        item = candidate()
        item.update(occluder=True, fire_label=None,
                    fire_label_suppression_reason='occlusion_graphic_visibility_unverified')
        self.assertEqual(decide(item, 100)[0], 'occlusion_graphic_visibility_unverified')
        item = candidate()
        item['instances'][0]['state'] = 'visible_truncated'
        item.update(fire_label=None, fire_label_suppression_reason='truncated_graphic_visibility_unverified')
        self.assertEqual(decide(item, 100)[0], 'truncated_graphic_visibility_unverified')
        item['fire_label'] = 'labels/fire/train/x.txt'
        with self.assertRaises(ValueError):
            decide(item, 100)

    def test_inconsistent_or_unknown_records_are_rejected(self):
        item = candidate()
        item['requested_fire_counts']['A'] = 2
        with self.assertRaises(ValueError):
            decide(item, 100)
        item = candidate()
        item['instances'][0]['state'] = 'unknown_projection'
        with self.assertRaises(ValueError):
            decide(item, 100)
        item = candidate()
        item['split'] = 'test'
        with self.assertRaises(ValueError):
            decide(item, 100)
        item = candidate()
        item['active_buildings'] = ['B']
        with self.assertRaises(ValueError):
            decide(item, 100)


if __name__ == '__main__':
    unittest.main()
