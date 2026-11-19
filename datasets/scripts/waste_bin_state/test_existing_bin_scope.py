"""A background may contain out-of-scope distant bins without teaching their state."""

import unittest
from pathlib import Path
from unittest.mock import patch

from datasets.scripts.waste_bin_state.generate_waste_v4_grouped import existing_bin_labels
from datasets.scripts.waste_bin_state.qualified_pool import qualified_backgrounds


class ExistingBinScope(unittest.TestCase):
    def test_complete_absent_allows_unlabelled_distant_bin(self):
        review = {'image_size': [640, 480], 'label': {
            'scope': {'kind': 'frame'}, 'completeness': 'complete',
            'presence': 'absent', 'instances': [],
            'excluded_instances': [{'object_id': 'distant-1', 'disposition': 'ignore',
                                    'reason': 'outside_task_recognition_range'}],
        }}
        self.assertEqual(existing_bin_labels(review), [])

    def test_complete_positive_keeps_only_task_targets(self):
        review = {'image_size': [640, 480], 'label': {
            'scope': {'kind': 'frame'}, 'completeness': 'complete',
            'presence': 'present',
            'instances': [{'object_id': 'main-1', 'kind': 'waste_bin',
                           'color': 'blue', 'lid_state': 'closed', 'class_id': 1,
                           'xyxy': [80, 180, 170, 380]}],
            'excluded_instances': [{'object_id': 'distant-1', 'disposition': 'ignore'}],
        }}
        self.assertEqual([row['bin_id'] for row in existing_bin_labels(review)], ['main-1'])

    def test_eligible_cross_task_raw_frame_enters_bin_pool(self):
        root = Path('D:/fixture')
        path = 'datasets/raw/fire/camera/frame.jpg'
        source = {'current_selection': {'qualified_pool': {
            'database': 'frame_environment', 'role': 'composite_background'}},
            'databases': {'frame_environment': 'unused.sqlite3'}}
        row = {'frame_id': 'frame-1', 'record_id': 'decision-1',
               'paths': [path], 'parent_group': 'capture-1', 'split': 'train'}
        with patch('datasets.scripts.waste_bin_state.qualified_pool.WORKSPACE', root), \
                patch('datasets.scripts.waste_bin_state.qualified_pool.query_frames', return_value=[row]), \
                patch('pathlib.Path.is_file', return_value=True):
            frames = qualified_backgrounds(source, 'train', database=root / 'unused.sqlite3')
        self.assertEqual([frame['image'] for frame in frames], [path])
        self.assertEqual(frames[0]['frame_id'], 'frame-1')


if __name__ == '__main__':
    unittest.main()
