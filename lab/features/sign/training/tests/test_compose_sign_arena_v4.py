import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import compose_sign_arena_v4 as arena


class SignArenaPlanTests(unittest.TestCase):
    def test_background_batch_path_is_not_hardcoded(self):
        parent = arena.OUTPUT_ROOT / 'arena_backgrounds_20261114'
        rows = arena._backgrounds({'source_scene': 'scene.blend', 'samples': [{'image': 'images/a.png'}]}, None, parent)
        self.assertEqual(rows[0]['image'], (parent / 'images/a.png').relative_to(arena.ROOT).as_posix())
        self.assertEqual(rows[0]['absence'], 'unknown')

    def setUp(self):
        self.row = {'image': 'datasets/example/background.png', 'road_anchor_center_x': [140, 500],
                    'parent': 'datasets/example/scene.blend', 'time': None,
                    'kind': 'blender_render', 'absence': 'unknown'}
        self.origin = {'index': 20, 'source': 'datasets/example/frame.png'}
        self.state = {'id': 'front', 'max_count': 1, 'conditions': ['mid_full', 'front', 'normal'],
                      'background': {'image': self.row['image']},
                      'target': {'state': 'mid_full', 'mask': 'train_20.png'},
                      'labels': {'class_id': 0, 'kind': 'visible_fixture_box'},
                      'augmentation': {'tone': 'normal'},
                      'parameters': {'height': 300, 'left': 200, 'top': 120},
                      'sources': [self.row['image'], self.row['parent'],
                          (arena.MASKS / 'train_20.png').relative_to(arena.ROOT).as_posix(),
                          self.origin['source'], arena.base.SIGN_VIDEO]}
        self.declaration = {'model_key': 'sign', 'constraints': {'image_size': [640, 480],
                            'label_kind': 'visible_fixture_box'}, 'strata': [self.state]}

    def test_prepare_no_pixels_and_rejects_repeated_or_unsupported_states(self):
        with mock.patch.object(arena.measured, 'compose', side_effect=AssertionError('premature synthesis')), \
                mock.patch.object(arena.base, 'read_image', side_effect=AssertionError('premature image read')):
            arena._prepare(self.declaration, [self.row], {'records': [self.origin]})
            for states in ([self.state, {**self.state, 'id': 'another'}],
                           [{**self.state, 'max_count': 2}],
                           [{**self.state, 'augmentation': {'tone': 'unknown'}}],
                           [{**self.state, 'sources': ['missing']} ]):
                with self.subTest(states=states), self.assertRaises(ValueError):
                    arena._prepare({**self.declaration, 'strata': states}, [self.row], {'records': [self.origin]})

    def test_planned_image_and_label_keep_unknown_background_status(self):
        state = {**self.state, 'planned_count': 1}
        plan = {'target_split': 'train', 'strata': [state]}
        prepared = {'front': (self.row, self.origin, arena.MASKS / 'train_20.png')}
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        run = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(arena, 'OUTPUT_ROOT', Path(tmp)), \
                mock.patch.object(arena.base, 'read_image', return_value=image), \
                mock.patch.object(arena, 'crop_alpha', return_value=image), \
                mock.patch.object(arena.measured, 'compose', return_value=(image, [200, 120, 250, 420], {'left': 200})):
            output = Path(tmp) / 'batch'
            result = arena._generate(output, plan, run, prepared)
            self.assertEqual(result['status'], 'candidates_pending_review')
            self.assertEqual(run.record_sample.call_count, 1)
            self.assertEqual(run.record_sample.call_args.kwargs['actual']['background_sign_absence'], 'unknown')
            self.assertEqual(len((output / 'labels_candidate/sign_arena_0000.txt').read_text().split()), 5)
            self.assertEqual(json.loads((output / 'manifest.json').read_text(encoding='utf8'))['samples'][0]['state_id'], 'front')


if __name__ == '__main__':
    unittest.main()
