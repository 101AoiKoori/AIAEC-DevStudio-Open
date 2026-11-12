"""Tests for label geometry, mask ID decoding and coverage scheduling."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/fire/training/scripts/fire_blender_labels.py'
spec = importlib.util.spec_from_file_location('fire_blender_labels', SCRIPT)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


class FireBlenderLabelTests(unittest.TestCase):
    def test_decodes_actual_8bit_srgb_instance_pass_color(self):
        # Observed PNG output of a linear emission (1, 1/32, 0.25) in
        # Blender 5.2 Standard view; PNG is sRGB, not linear light.
        stored_rgb = np.array([[[255, 49, 137], [0, 0, 0]]], dtype=np.float32) / 255
        linear = labels.srgb_to_linear(stored_rgb)
        self.assertEqual(labels.decode_ids(linear, {1}).tolist(), [[1, 0]])
        self.assertAlmostEqual(float(linear[0, 0, 1]), 1 / 32, delta=.001)
        self.assertAlmostEqual(float(linear[0, 0, 2]), .25, delta=.003)

    def test_id_decoding_uses_relative_color_to_avoid_antialias_false_ids(self):
        rgb = np.zeros((2, 3, 3), dtype=np.float32)
        rgb[0, 0] = (1, 5 / 32, .25)
        rgb[0, 1] = (.50, .50 * 5 / 32, .125)
        rgb[0, 2] = (1, 9 / 32, .25)
        rgb[1, 0] = (.20, .20 * 5 / 32, .05)
        rgb[1, 1] = (1, 5 / 32, .70)
        decoded = labels.decode_ids(rgb, {5, 9})
        self.assertEqual(decoded.tolist(), [[5, 5, 9], [0, 0, 0]])

    def test_visible_bbox_and_yolo_use_exclusive_pixel_bounds(self):
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[2:7, 4:10] = 1
        bbox, pixels = labels.visible_bbox(mask)
        self.assertEqual(bbox, [4, 2, 10, 7])
        self.assertEqual(pixels, 30)
        self.assertEqual(labels.yolo_row(bbox, 20, 10), '0 0.350000 0.450000 0.300000 0.500000')
        self.assertEqual(labels.visible_bbox(mask * 0), (None, 0))

    def test_unseen_or_subpixel_is_not_treated_as_empty(self):
        inside = {'in_front': True, 'intersects': True, 'truncated': False}
        self.assertEqual(labels.label_state(inside, None, 0), 'unknown_too_small_or_occluded')
        self.assertEqual(labels.label_state(inside, [1, 1, 5, 5], 16), 'visible')
        self.assertEqual(labels.label_state(dict(inside, truncated=True), [1, 1, 5, 5], 16),
                         'visible_truncated')
        self.assertEqual(labels.label_state(dict(inside, intersects=False), None, 0), 'outside_frame')
        self.assertEqual(labels.label_state(dict(inside, in_front=False), None, 0), 'unknown_projection')

    def test_ten_sample_plan_covers_declared_scenarios(self):
        plan = [labels.scenario(i) for i in range(10)]
        self.assertTrue(any(sum(p['counts']) == 0 for p in plan))
        self.assertTrue(any(sum(p['counts']) >= 4 for p in plan))
        self.assertTrue(any(len(p['active']) > 1 for p in plan))
        self.assertTrue(any(p['kind'] == 'occluder' for p in plan))
        self.assertTrue(any(p['kind'] == 'truncated' for p in plan))
        self.assertEqual(labels.scenario(10), plan[0])

    def test_fire_export_rejects_unverified_occlusion_and_truncation(self):
        self.assertIsNone(labels.fire_label_suppression([], False))
        self.assertIsNone(labels.fire_label_suppression(['visible', 'outside_frame'], False))
        self.assertEqual(labels.fire_label_suppression(['visible'], True),
                         'occlusion_graphic_visibility_unverified')
        self.assertEqual(labels.fire_label_suppression(['visible_truncated'], False),
                         'truncated_graphic_visibility_unverified')
        self.assertEqual(labels.fire_label_suppression(['unknown_too_small_or_occluded'], False),
                         'unknown_target_visibility')
        self.assertEqual(labels.fire_label_suppression(['unknown_projection'], False),
                         'unknown_target_visibility')

    def test_ocr_status_does_not_promote_truncated_or_unseen_truth(self):
        self.assertEqual(labels.ocr_status('visible'), 'candidate_requires_readability_review')
        self.assertEqual(labels.ocr_status('visible_truncated'), 'ineligible_truncated')
        self.assertEqual(labels.ocr_status('visible', True), 'ineligible_occlusion_unverified')
        self.assertEqual(labels.ocr_status('unknown_too_small_or_occluded'),
                         'ineligible_not_fully_visible')


if __name__ == '__main__':
    unittest.main()
