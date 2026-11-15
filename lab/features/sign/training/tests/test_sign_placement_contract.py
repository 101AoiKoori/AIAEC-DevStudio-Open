import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compose_sign_v4_measured as measured
import compose_sign_arena_v4 as arena


class SignPlacementContractTests(unittest.TestCase):
    def setUp(self):
        self.background = np.zeros((480, 640, 3), dtype=np.uint8)
        self.sprite = np.full((100, 20, 4), 255, dtype=np.uint8)
        self.placement = {
            'height': 100, 'left': 100, 'top': 300,
            'ground_polygon': [[0, 350], [640, 350], [640, 550], [0, 550]],
            'occlusion_polygons': [], 'visible_kind': 'fixture', 'visibility_evidence': 'fixture details visible',
        }

    def compose(self, placement):
        return measured.compose(self.background, self.sprite, 'placed', 'normal', None, None, placement)

    def test_ground_placement_and_occlusion_update_pixels_and_visible_box(self):
        placement = {**self.placement, 'occlusion_polygons': [[[90, 290], [130, 290], [130, 349], [90, 349]]]}
        image, box, info = self.compose(placement)
        self.assertEqual(box, [100, 350, 120, 400])
        self.assertTrue(np.all(image[300:350, 100:120] == 0))
        self.assertTrue(np.all(image[350:400, 100:120] == 255))
        self.assertEqual(info['contact_point_xy'], [110.0, 400.0])
        self.assertEqual(info['visible_alpha_pixels'], 1000)

    def test_four_edges_clip_to_visible_pixels_and_allow_identifiable_pole(self):
        placements = [
            ({'left': -10, 'top': 300}, [0, 300, 10, 400]),
            ({'left': 630, 'top': 300}, [630, 300, 640, 400]),
            ({'left': 100, 'top': -50, 'ground_polygon': [[0, 0], [640, 0], [640, 100], [0, 100]]}, [100, 0, 120, 50]),
            ({'left': 100, 'top': 430}, [100, 430, 120, 480]),
        ]
        for changes, expected in placements:
            with self.subTest(changes=changes):
                _, box, info = self.compose({**self.placement, **changes, 'visible_kind': 'identifiable_pole',
                    'ground_polygon': changes.get('ground_polygon', [[-100, 0], [740, 0], [740, 600], [-100, 600]])})
                self.assertEqual(box, expected)
                self.assertEqual(info['visible_kind'], 'identifiable_pole')

    def test_unknown_and_invalid_placement_never_become_negative(self):
        cases = [
            ({**self.placement, 'visible_kind': 'unknown'}, 'Unknown sign visibility'),
            ({**self.placement, 'ground_polygon': [[0, 0], [10, 0], [10, 10]]}, 'outside the declared ground'),
            ({**self.placement, 'occlusion_polygons': [[[90, 290], [130, 290], [130, 410], [90, 410]]]}, 'No visible sign pixels'),
        ]
        for placement, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.compose(placement)

    def test_plan_contract_accepts_explicit_region_and_rejects_unknown(self):
        row = {'image': 'bg.png', 'sign_absence': 'confirmed', 'sign_absence_evidence': 'background reviewed'}
        state = {'background': {'image': 'bg.png'}, 'target': {'state': 'placed', 'mask': 'train_20.png'},
                 'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                 'augmentation': {'tone': 'normal'}, 'parameters': self.placement}
        self.assertEqual(measured.validate_planned_stratum(state, [row]), row)
        with self.assertRaisesRegex(ValueError, 'background sign absence'):
            measured.validate_planned_stratum(state, [{'image': 'bg.png'}])
        unknown = {**state, 'parameters': {**self.placement, 'visible_kind': 'unknown'}}
        with self.assertRaisesRegex(ValueError, 'Unknown sign visibility'):
            measured.validate_planned_stratum(unknown, [row])
        # Arena uses the same validator before its own source bookkeeping.
        with self.assertRaisesRegex(ValueError, 'Unknown sign visibility'):
            arena._prepare({'model_key': 'sign', 'constraints': {'image_size': [640, 480],
                'label_kind': 'visible_fixture_box'}, 'strata': [unknown]}, [row], {'records': []})

    def test_ordinary_pole_can_only_be_empty_when_background_has_absence_observation(self):
        state = {'background': {'image': 'pole.png'}, 'target': {'state': 'negative', 'mask': 'none'},
                 'labels': {'class_id': 0, 'kind': 'empty_candidate'}, 'augmentation': {'tone': 'normal'},
                 'parameters': {'kind': 'empty_background'}}
        with self.assertRaisesRegex(ValueError, 'absence candidate'):
            measured.validate_planned_stratum(state, [{'image': 'pole.png', 'review_status': 'unknown'}])
        row = {'image': 'pole.png', 'review_status': 'candidate: no identifiable sign'}
        self.assertEqual(measured.validate_planned_stratum(state, [row]), row)

    def test_both_entry_adapters_record_visible_box_and_background_evidence(self):
        state = {'id': 'placed', 'planned_count': 1, 'background': {'image': 'bg.png'},
                 'target': {'state': 'placed', 'mask': 'train_20.png'},
                 'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                 'augmentation': {'tone': 'normal'}, 'parameters': self.placement, 'sources': ['bg.png']}
        plan = {'target_split': 'train', 'strata': [state]}
        row = {'image': 'bg.png', 'parent_image': 'parent.png', 'pts_seconds': 1.0,
               'parent': 'parent.png', 'time': 1.0, 'kind': 'blender_render', 'absence': 'unknown',
               'sign_absence': 'confirmed', 'sign_absence_evidence': 'review record'}
        origin = {'source': 'sign.png'}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(measured, 'OUTPUT_ROOT', Path(tmp) / measured.__name__), \
                mock.patch.object(arena, 'OUTPUT_ROOT', Path(tmp)), \
                mock.patch.object(measured.base, 'read_image', return_value=self.background), \
                mock.patch.object(measured.base, 'cutout', return_value=self.sprite), \
                mock.patch.object(arena, 'crop_alpha', return_value=self.sprite):
            for entry, prepared, label in (
                (measured, {'placed': (row, origin)}, 'labels_candidate/sign_00000.txt'),
                (arena, {'placed': (row, origin, arena.MASKS / 'train_20.png')},
                 'labels_candidate/sign_arena_0000.txt'),
            ):
                run = mock.Mock()
                output = Path(tmp) / entry.__name__
                result = entry._generate(output, plan, run, prepared)
                actual = result['samples'][0]['actual']
                self.assertEqual(actual['bbox_xyxy'], [100, 300, 120, 400])
                self.assertEqual(actual['background_sign_absence_evidence'], 'review record')
                self.assertEqual(len((output / label).read_text().split()), 5)
                self.assertEqual(run.record_sample.call_count, 1)


if __name__ == '__main__':
    unittest.main()
