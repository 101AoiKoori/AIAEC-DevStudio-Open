import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import synthesize_vehicle_scene as bike
from synthesize_vehicle_scene import scenario


class VehicleScenarioTest(unittest.TestCase):
    def test_twenty_sample_cycle_has_requested_scenes(self):
        plans = [scenario(i) for i in range(20)]
        self.assertEqual(Counter(kind for kind, _ in plans),
                         {'upright_only': 8, 'fallen_present': 8, 'empty': 4})
        for kind, classes in plans:
            if kind == 'upright_only':
                self.assertTrue(classes and set(classes) == {'upright'})
            elif kind == 'fallen_present':
                self.assertIn('fallen', classes)
            else:
                self.assertEqual(classes, [])
        self.assertIn(7, [len(classes) for _, classes in plans])
        self.assertTrue(any(set(classes) == {'upright', 'fallen'} for _, classes in plans))

    def test_planned_scene_uses_declared_vehicle_and_rejects_unknown_fields(self):
        image_path = bike.ROOT / 'datasets/test_background.jpg'
        background = {'path': image_path, 'frame': 1, 'seconds': 0.1}
        foreground = {'path': 'datasets/test_bike.png', 'class': 'upright',
                      'source_video': 'datasets/test_bike.mp4'}
        sprite = np.zeros((60, 30, 4), dtype=np.uint8)
        sprite[:, :, :3] = 180
        sprite[:, :, 3] = 255
        state = {'background': {'image': image_path.relative_to(bike.ROOT).as_posix(),
                                'crop_xyxy': [0, 0, 960, 480]},
                 'target': {'vehicles': [{'class': 'upright', 'source': foreground['path']}]},
                 'labels': {'format': 'yolo_visible_box', 'classes': {'upright': 0, 'fallen': 1}},
                 'augmentation': {'gain': 1.0, 'gamma': 1.0},
                 'parameters': {'actors': [{'foot_x': 400, 'foot_y': 400, 'size_px': 80,
                                            'yaw': 0, 'pitch': 0, 'roll': 0,
                                            'relative_exposure': 1.0}]},
                 'sources': sorted([image_path.relative_to(bike.ROOT).as_posix(),
                                    foreground['path'], foreground['source_video'], 'datasets/test_scene.mp4'])}
        vehicles = {'upright': [(foreground, sprite)], 'fallen': []}
        with mock.patch.object(bike.scene, 'read_color', return_value=np.full((480, 960, 3), 120, dtype=np.uint8)):
            image, objects, masks, info = bike.planned_scene(state, [background], vehicles,
                                                              'datasets/test_scene.mp4', 'test')
            self.assertEqual(image.shape[:2], (480, 960))
            self.assertEqual([obj['class'] for obj in objects], ['upright'])
            self.assertEqual(objects[0]['mask_source'], foreground['path'])
            self.assertEqual(len(masks), 1)
            self.assertEqual(info['image'], state['background']['image'])
            state['augmentation']['saturation'] = 0.5
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                bike.planned_scene(state, [background], vehicles, 'datasets/test_scene.mp4', 'test')

    def test_undeclared_conditions_rejected(self):
        fields = {'id': 's', 'failure_event': 'miss', 'min_count': 0, 'max_count': 1,
                  'prior': {}, 'exposure_weight': 1, 'failure_cost': 1,
                  'conditions': ['unimplemented'], 'background': {}, 'target': {},
                  'labels': {}, 'augmentation': {}, 'parameters': {}, 'sources': []}
        declaration = {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train', 'budget': 1,
                       'purpose': 'test', 'uncertainty_weight': 0, 'independence_basis': 'test',
                       'constraints': {'minimum_visible_pixels': 120, 'minimum_visible_fraction': .55},
                       'strata': [fields]}
        with self.assertRaisesRegex(ValueError, 'conditions'):
            bike.validate_declaration(declaration)
        fields['conditions'] = []
        fields['max_count'] = 2
        with self.assertRaisesRegex(ValueError, 'at most one'):
            bike.validate_declaration(declaration)

    def test_main_rejects_missing_plan_before_source_loading(self):
        with mock.patch.object(sys, 'argv', ['synthesize_vehicle_scene.py', '--output', 'unused']), \
             mock.patch.object(bike.scene, 'eligible_backgrounds') as backgrounds, \
             mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                bike.main()
            backgrounds.assert_not_called()


if __name__ == '__main__':
    unittest.main()
