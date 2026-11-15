import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import compose_v4_grounded_train as entry


def fixture():
    bg = {'id': 'frame_007000', 'image': 'datasets/bg.png', 'parent_image': 'datasets/parent.png',
          'source_video': 'datasets/scene.mp4', 'task_state': 'candidate_complete_negative'}
    source = {'path': 'datasets/bike.png', 'source_frame': 'datasets/frame.png',
              'source_video': 'datasets/bike.mp4', 'class': 'upright'}
    sprite = np.full((60, 30, 4), 255, dtype=np.uint8)
    vehicles = {'upright': [(source, sprite)], 'fallen': []}
    state = {'id': 'one', 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
             'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
             'conditions': [], 'background': {'id': bg['id'], 'image': bg['image']},
             'target': {'vehicles': [{'class': 'upright', 'source': source['path']}]},
             'labels': copy.deepcopy(entry.GROUND_LABELS), 'augmentation': {'tone': 'original'},
             'parameters': {'actors': [{'foot_x': 320, 'foot_y': 400, 'size_px': 80,
                                        'yaw': 0, 'pitch': 0, 'roll': 0, 'relative_exposure': 1.0}],
                            'value_gain': 1.0, 'saturation_gain': 1.0},
             'sources': sorted([bg['image'], bg['parent_image'], bg['source_video'],
                                source['path'], source['source_frame'], source['source_video']])}
    plan = {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train',
            'purpose': 'contract fixture', 'budget': 1, 'uncertainty_weight': 0,
            'independence_basis': 'fixture', 'constraints': dict(entry.GROUND_CONSTRAINTS), 'strata': [state]}
    return plan, state, bg, vehicles


class GroundPlanTest(unittest.TestCase):
    def test_declared_source_and_layout_drive_image_and_label(self):
        plan, state, bg, vehicles = fixture()
        with mock.patch.object(entry, 'source_role', return_value=('train', None)), \
             mock.patch.object(entry, 'read_image', return_value=np.full((480, 640, 3), 120, np.uint8)):
            entry.validate_ground_plan(plan, [bg], vehicles, {})
            image, objects, masks, actual_bg = entry.render_ground_state(state, [bg], vehicles)
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(objects[0]['class'], 'upright')
        self.assertEqual(objects[0]['foreground_source']['path'], state['target']['vehicles'][0]['source'])
        self.assertEqual(actual_bg['image'], state['background']['image'])
        self.assertEqual(len(masks), 1)
        self.assertEqual(objects[0]['foot_xy'], [320, 400])

    def test_unsupported_targets_and_duplicate_states_fail(self):
        plan, state, bg, vehicles = fixture()
        with mock.patch.object(entry, 'source_role', return_value=('train', None)):
            state['target']['vehicles'] *= 8
            with self.assertRaisesRegex(ValueError, 'zero to seven'):
                entry.validate_ground_plan(plan, [bg], vehicles, {})
            state['target']['vehicles'] = state['target']['vehicles'][:1]
            state['max_count'] = 2
            with self.assertRaisesRegex(ValueError, 'at most one'):
                entry.validate_ground_plan(plan, [bg], vehicles, {})
            state['max_count'] = 1
            duplicate = copy.deepcopy(state)
            duplicate['id'] = 'two'
            state['parameters']['sample_seed'] = 1
            duplicate['parameters']['sample_seed'] = 2
            plan['strata'].append(duplicate)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                entry.validate_ground_plan(plan, [bg], vehicles, {})


if __name__ == '__main__':
    unittest.main()
