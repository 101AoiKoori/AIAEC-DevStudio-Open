"""Regression checks for parent declarations and actual cross-split frames."""
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from datasets.scripts.waste_bin_state import build_full_dataset as dataset
from datasets.scripts.waste_bin_state import plan_full_evaluation as evaluation


PARENT = 'Archive/Assets/captures/first/full_scene.mp4'
SAME_NAME = 'Archive/Assets/captures/second/full_scene.mp4'
MOVED_PARENT = 'Archive/Assets/captures/relocated/full_scene.mp4'
HISTORY = ((MOVED_PARENT, PARENT),)
PAIRING = {'backgrounds': [{
    'id': 'camera_parent',
    'paths': [PARENT],
    'split': {'primary': 'train', 'status': 'reserved', 'same_source_allowed_in': ['val']},
}]}


def capture(parent, identity, *, kind='real_background_composite'):
    record = {
        'source_kind': kind, 'source_record_id': identity, 'parent_group': parent,
        'physical_parent_group': dataset.physical_group({'parent_group': parent}, HISTORY),
        'source_image': f'datasets/raw/{identity}.png',
        'background_image': f'datasets/raw/{identity}.png',
        'frame_environment': {'content_sha256': identity},
    }
    return None, '', record


class FullSourceScopeTests(unittest.TestCase):
    def test_declared_parent_reuse_does_not_allow_identical_frames(self):
        composites = {'train': [capture(PARENT, 'first')],
                      'val': [capture(MOVED_PARENT, 'second')], 'test': []}
        distinct_parent = capture(SAME_NAME, 'third', kind='direct_real')
        repeated_frame = capture(MOVED_PARENT, 'first', kind='direct_real')
        reserved_parent = capture(PARENT, 'fourth', kind='direct_real')
        direct = {'train': [], 'val': [distinct_parent, repeated_frame], 'test': [reserved_parent]}
        selected, exclusions = dataset.filter_direct(direct, composites, PAIRING, HISTORY)
        self.assertEqual(selected['val'], [distinct_parent])
        self.assertEqual(selected['test'], [])
        self.assertEqual([row['reason'] for row in exclusions],
                         ['camera_frame_reserved_in_other_split', 'camera_parent_reserved_in_other_split'])
        self.assertEqual(dataset.camera_parent_reuse(composites), [{
            'relation': 'source_group', 'source': PARENT, 'splits': ['train', 'val'],
            'counts': {'train': 1, 'val': 1},
        }])
        composites['val'] = [capture(MOVED_PARENT, 'first')]
        with self.assertRaisesRegex(ValueError, 'Camera frame shared across production splits'):
            dataset.filter_direct(direct, composites, PAIRING, HISTORY)

    def test_planning_accepts_distinct_parent_with_same_basename(self):
        source = {'databases': {'frame_environment': 'datasets/test.sqlite3'},
                  'current_selection': {'qualified_pool': {'database': 'frame_environment'}},
                  'pairing': PAIRING}
        earlier = {'frame_id': 'earlier', 'parent_group': PARENT, 'content_sha256': 'first'}
        candidates = [
            {'id': 'same_filename', 'frame_id': 'different', 'parent_group': SAME_NAME,
             'parent_image': 'datasets/raw/distinct.png'},
            {'id': 'duplicated', 'frame_id': 'other_alias', 'parent_group': MOVED_PARENT,
             'parent_image': 'datasets/raw/alias.png'},
            {'id': 'declared', 'frame_id': 'another', 'parent_group': MOVED_PARENT,
             'parent_image': 'datasets/raw/allowed.png'},
        ]
        qualified = {
            candidate['id']: ({'parent_group': candidate['parent_group'], 'geometry': {},
                               'content_sha256': digest},
                              SimpleNamespace(candidate_mask=np.ones((10, 10), dtype=np.uint8)))
            for candidate, digest in zip(candidates, ('second', 'first', 'third'))
        }
        with (
            patch.object(evaluation, 'read_source_history', return_value=HISTORY),
            patch.object(evaluation, 'query_frames', return_value=[earlier]),
            patch.object(evaluation, 'qualified_backgrounds', return_value=candidates),
            patch.object(evaluation.compositor, 'eligible_ground_backgrounds',
                         return_value=(None, qualified)),
            patch.object(evaluation.compositor, 'build_queue'),
            patch.object(evaluation.compositor, 'source_role', return_value=('train',)),
            patch.object(evaluation.compositor, 'current_scale_observation'),
            patch.object(evaluation, 'prepared_background', side_effect=lambda row, *_: {
                'id': row['id'], 'ground_polygon_xy': [[1, 1], [8, 1], [8, 8], [1, 8]],
                'source_image_size': [10, 10],
            }),
        ):
            selected, _, rejected, contacts = evaluation.evaluation_backgrounds(source, 'val', random.Random(5))
        self.assertEqual([row['id'] for row in selected], ['same_filename', 'declared'])
        self.assertEqual(dict(rejected), {'camera_frame_assigned_to_earlier_split': 1})
        self.assertEqual(contacts, {MOVED_PARENT: ['train']})


if __name__ == '__main__':
    unittest.main()
