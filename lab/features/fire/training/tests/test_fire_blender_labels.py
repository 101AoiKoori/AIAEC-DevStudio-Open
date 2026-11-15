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
    def test_ground_anchor_scale_and_foreground_exclusion(self):
        placement = {'ground_anchor_xyxy': [420, 340, 620, 460], 'building_height_px': [120, 200],
                     'forbidden_rects_xyxy': [[100, 50, 390, 445]], 'max_overlap_fraction': .01}
        self.assertEqual(labels.placement_violations([[455, 220, 550, 400]], placement), [])
        self.assertIn('foreground_overlap_exceeds_limit',
                      labels.placement_violations([[350, 220, 500, 400]], placement))
        self.assertIn('building_height_outside_reviewed_range',
                      labels.placement_violations([[455, 100, 550, 400]], placement))
        self.assertIn('ground_anchor_outside_reviewed_region',
                      labels.placement_violations([[455, 110, 550, 290]], placement))
        self.assertIn('building_clipped_without_approval',
                      labels.placement_violations([[590, 220, 680, 400]], placement))

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

    def test_explicit_scenarios_cover_each_building_and_validate_boundaries(self):
        target = {'scenario': 'explicit', 'active_buildings': ['A'], 'fire_counts': {'A': 0},
                  'view': 'near_frontal', 'placement': 'all', 'visibility': 'no_added_occluder'}
        for building in ('A', 'B', 'C'):
            for count in (0, 1, 5):
                target.update(active_buildings=[building], fire_counts={building: count})
                plan = labels.explicit_scenario(target)
                self.assertEqual((plan['active'], plan['counts']), ((building,), (count,)))
        target.update(active_buildings=['A', 'C'], fire_counts={'A': 1, 'C': 5},
                      view='oblique_left', placement='upper', visibility='occluder')
        self.assertEqual(labels.explicit_scenario(target)['kind'], 'occluder')
        for bad in (
            dict(target, active_buildings=['C', 'A']),
            dict(target, active_buildings=['A', 'A']),
            dict(target, fire_counts={'A': 1, 'C': 6}),
            dict(target, fire_counts={'A': True, 'C': 1}),
            dict(target, fire_counts={'A': 0, 'C': 1}),
            dict(target, view='side'), dict(target, placement='side_wall'),
            dict(target, visibility='verified_occlusion'),
        ):
            with self.assertRaises(ValueError):
                labels.explicit_scenario(bad)

    def test_front_facade_placement_zones_are_bounded_grid_subsets(self):
        all_slots = set(labels.front_facade_slots('all'))
        self.assertEqual(len(all_slots), 20)
        for zone in ('left', 'right', 'upper', 'lower'):
            slots = labels.front_facade_slots(zone)
            self.assertGreaterEqual(len(slots), 5)
            self.assertTrue(set(slots) < all_slots)
        self.assertTrue(all(x < 0 for x, _ in labels.front_facade_slots('left')))
        self.assertTrue(all(x > 0 for x, _ in labels.front_facade_slots('right')))
        self.assertTrue(all(z >= .265 for _, z in labels.front_facade_slots('upper')))
        self.assertTrue(all(z <= .165 for _, z in labels.front_facade_slots('lower')))
        with self.assertRaises(ValueError):
            labels.front_facade_slots('side_wall')

    def test_supported_c_wall_states_reject_other_buildings_counts_and_occlusion(self):
        target = {'scenario': 'explicit', 'active_buildings': ['C'], 'fire_counts': {'C': 1},
                  'view': 'near_frontal', 'placement': 'c_supported_left', 'visibility': 'no_added_occluder'}
        for zone in ('c_supported_left', 'c_supported_right'):
            target['placement'] = zone
            self.assertEqual(len(labels.front_facade_slots(zone)), 5)
            self.assertEqual(labels.explicit_scenario(target)['placement'], zone)
        for bad in (dict(target, active_buildings=['A'], fire_counts={'A': 1}),
                    dict(target, fire_counts={'C': 0}), dict(target, fire_counts={'C': 2}),
                    dict(target, visibility='occluder'), dict(target, visibility='truncated')):
            with self.assertRaises(ValueError):
                labels.explicit_scenario(bad)

    def test_full_support_detects_holes_and_overlapping_faces(self):
        box = (0., 0., 2., 1.)
        faces = [(0., 0., 1., 1.), (1., 0., 2., 1.)]
        self.assertAlmostEqual(labels.rectangle_uncovered_area(faces, box), 0.)
        self.assertAlmostEqual(labels.rectangle_uncovered_area(faces + [faces[0]], box), 0.)
        self.assertAlmostEqual(labels.rectangle_uncovered_area([(0., 0., .9, 1.), (1.1, 0., 2., 1.)], box), .2)
        with self.assertRaises(ValueError):
            labels.rectangle_uncovered_area([(1., 0., 0., 1.)], box)

    def test_rotated_envelope_bounds_all_intermediate_angles(self):
        import math
        ex, ez = labels.rotated_paper_envelope(.0127, .0177, .002, .09)
        for angle in np.linspace(-.09, .09, 101):
            for x in (-.0127, .0127):
                for z in (-.0177, .0177):
                    self.assertLessEqual(abs(x * math.cos(angle) + z * math.sin(angle)) + .002, ex + 1e-12)
                    self.assertLessEqual(abs(z * math.cos(angle) - x * math.sin(angle)) + .002, ez + 1e-12)

    def test_paper_depth_rejects_floating_or_embedded_templates(self):
        self.assertTrue(labels.paper_depth_supported(-.1008, -.10162, -.101019, .0002, .00003))
        self.assertFalse(labels.paper_depth_supported(-.1008, -.11, -.109, .0002, .00003))
        self.assertFalse(labels.paper_depth_supported(-.1008, -.101, -.1005, .0002, .00003))
        with self.assertRaises(ValueError):
            labels.paper_depth_supported(-.1008, -.101, -.102, .0002, .00003)

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

    def test_tiny_flame_graphics_do_not_create_complete_labels(self):
        self.assertEqual(labels.fire_scale_suppression([{'kind': 'fire_sticker',
            'pixel_bbox_xyxy': [20, 20, 30, 34]}]), 'fire_graphic_below_reviewable_pixel_scale')
        self.assertIsNone(labels.fire_scale_suppression([{'kind': 'fire_sticker',
            'pixel_bbox_xyxy': [20, 20, 34, 40]}]))
        self.assertIsNone(labels.fire_scale_suppression([]))

    def test_ocr_status_does_not_promote_truncated_or_unseen_truth(self):
        self.assertEqual(labels.ocr_status('visible'), 'candidate_requires_readability_review')
        self.assertEqual(labels.ocr_status('visible_truncated'), 'ineligible_truncated')
        self.assertEqual(labels.ocr_status('visible', True), 'ineligible_occlusion_unverified')
        self.assertEqual(labels.ocr_status('unknown_too_small_or_occluded'),
                         'ineligible_not_fully_visible')


if __name__ == '__main__':
    unittest.main()
