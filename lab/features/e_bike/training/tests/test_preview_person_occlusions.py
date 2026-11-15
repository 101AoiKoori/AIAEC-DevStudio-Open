import importlib.util
import copy
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/e_bike/training/scripts/preview_person_occlusions.py'
spec = importlib.util.spec_from_file_location('preview_person_occlusions', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def plan_fixture():
    available = {'community': 'datasets/person.png', 'noncommunity': 'datasets/other.png',
                 'upright': 'datasets/upright.png', 'fallen': 'datasets/fallen.png'}
    state = {'id': 'one', 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
             'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1, 'conditions': [],
             'background': {'kind': 'synthetic_flat'},
             'target': {'person_class': 'community', 'person_source': available['community'],
                        'vehicle_class': 'upright', 'vehicle_source': available['upright']},
             'labels': copy.deepcopy(module.PREVIEW_LABELS), 'augmentation': {'style': 'none'},
             'parameters': {'person_height': 100, 'bike_width': 30, 'floor': 500,
                            'person_x': 400, 'bike_x': 410},
             'sources': sorted([available['community'], available['upright']])}
    plan = {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train',
            'purpose': 'contract fixture', 'budget': 1, 'uncertainty_weight': 0,
            'independence_basis': 'fixture', 'constraints': dict(module.PREVIEW_CONSTRAINTS),
            'strata': [state]}
    return plan, state, available


class OcclusionTest(unittest.TestCase):
    def test_holes_leave_person_visible_and_box_uses_visible_pixels(self):
        canvas = np.zeros((module.HEIGHT, module.WIDTH, 3), np.uint8)
        person = np.full((4, 4, 4), 255, np.uint8)
        person_mask = module.mask_and_place(canvas, person, 10, 10)
        bike = np.full((4, 4, 4), 255, np.uint8)
        bike[:, :, :3] = 0
        bike[:, :, 3] = 0
        bike[:, :2, 3] = 255
        bike[1, 0, 3] = 0
        bike_mask = module.mask_and_place(canvas, bike, 10, 10)
        visible = np.logical_and(person_mask, np.logical_not(bike_mask))
        self.assertEqual(int(visible.sum()), 9)
        self.assertTrue(visible[11, 10])
        self.assertFalse(visible[10, 10])
        self.assertEqual(module.box(visible), [10, 10, 14, 14])

    def test_rejects_out_of_bounds_placement(self):
        canvas = np.zeros((module.HEIGHT, module.WIDTH, 3), np.uint8)
        with self.assertRaises(ValueError):
            module.mask_and_place(canvas, np.ones((4, 4, 4), np.uint8), 1023, 767)

    def test_explicit_state_generates_occlusion_from_requested_sources(self):
        plan, state, available = plan_fixture()
        module.validate_preview_plan(plan, available)
        sprite = np.full((60, 30, 4), 255, np.uint8)
        with mock.patch.object(module, 'load_rgba', return_value=sprite) as load:
            _, full, visible, vehicle, actual = module.render_planned_preview(state)
        self.assertEqual(load.call_count, 2)
        self.assertGreater(int(visible.sum()), 0)
        self.assertLess(int(visible.sum()), int(full.sum()))
        self.assertGreater(int(vehicle.sum()), 0)
        self.assertEqual(actual['person_source'], available['community'])
        self.assertEqual(actual['vehicle_source'], available['upright'])

    def test_rejects_unsupported_or_duplicate_states(self):
        plan, state, available = plan_fixture()
        state['max_count'] = 2
        with self.assertRaisesRegex(ValueError, 'at most one'):
            module.validate_preview_plan(plan, available)
        state['max_count'] = 1
        duplicate = copy.deepcopy(state)
        duplicate['id'] = 'two'
        plan['strata'].append(duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.validate_preview_plan(plan, available)
        plan['strata'].pop()
        state['augmentation']['style'] = 'unimplemented'
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            module.validate_preview_plan(plan, available)

    def test_legacy_unplanned_entry_points_fail_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'explicit production plan'):
            module.preview('a', 'b', 'c', 'd', 'output')
        with self.assertRaisesRegex(ValueError, 'unplanned generation'):
            module._preview('a', 'b', 'c', 'd', 'output')


if __name__ == '__main__':
    unittest.main()
