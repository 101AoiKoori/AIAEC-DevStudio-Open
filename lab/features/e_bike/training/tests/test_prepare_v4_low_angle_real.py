import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import prepare_v4_low_angle_real as entry


def fixture():
    row = {'id': 'frame1', 'image': 'datasets/fake_crop.png', 'crop_xyxy': [10, 0, 110, 100],
           'source_frame': 'datasets/fake_frame.png', 'source_video': 'datasets/fake_video.mp4',
           'split': 'train_candidate', 'label_status': 'complete_candidate',
           'output_image_size': [100, 100], 'label': 'datasets/fake_label.txt',
           'boxes': [{'class': 'upright', 'xyxy': [10, 20, 30, 60]}]}
    state = {'id': 's1', 'failure_event': 'miss', 'min_count': 1, 'max_count': 1,
             'prior': {'alpha': 1, 'beta': 1}, 'exposure_weight': 1, 'failure_cost': 1,
             'conditions': [], 'background': {'image': row['image'], 'crop_xyxy': row['crop_xyxy']},
             'target': {'classes': ['upright']}, 'labels': copy.deepcopy(entry.LABELS),
             'augmentation': {'tone': 'mild'}, 'parameters': {'value_gain': .9, 'saturation_gain': .85},
             'sources': sorted([row['image'], row['source_frame'], row['source_video']])}
    declaration = {'schema_version': 1, 'model_key': 'e_bike', 'target_split': 'train',
                   'purpose': 'contract test', 'budget': 1, 'uncertainty_weight': 0,
                   'independence_basis': 'fixture', 'constraints': dict(entry.CONSTRAINTS), 'strata': [state]}
    return declaration, row, state


class PlannedLowAngleTest(unittest.TestCase):
    def test_exact_reviewed_state_and_sources(self):
        declaration, row, state = fixture()
        entry.validate_declaration(declaration)
        expected = '0 0.20000000 0.40000000 0.20000000 0.40000000'
        with mock.patch.object(entry, 'source_role', return_value=('train', None)), \
             mock.patch.object(Path, 'read_text', return_value=expected):
            checked, labels = entry.check_state(state, [row], {})
            self.assertIs(checked, row)
            self.assertEqual(labels, [expected])
            state['sources'].pop()
            with self.assertRaisesRegex(ValueError, 'sources'):
                entry.check_state(state, [row], {})

    def test_unsupported_tone_and_mismatched_target_fail(self):
        _, row, state = fixture()
        with mock.patch.object(entry, 'source_role', return_value=('train', None)):
            state['augmentation']['tone'] = 'unknown'
            with self.assertRaisesRegex(ValueError, 'tone'):
                entry.check_state(state, [row], {})
            state['augmentation']['tone'] = 'mild'
            state['target']['classes'] = ['fallen']
            with self.assertRaisesRegex(ValueError, 'targets'):
                entry.check_state(state, [row], {})

    def test_duplicates_overallocation_and_unimplemented_conditions_fail(self):
        declaration, _, state = fixture()
        state['max_count'] = 2
        with self.assertRaisesRegex(ValueError, 'at most one'):
            entry.validate_declaration(declaration)
        state['max_count'] = 1
        duplicate = copy.deepcopy(state)
        duplicate['id'] = 's2'
        declaration['strata'].append(duplicate)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            entry.validate_declaration(declaration)
        declaration['strata'].pop()
        state['conditions'] = ['not implemented']
        with self.assertRaisesRegex(ValueError, 'conditions'):
            entry.validate_declaration(declaration)

    def test_missing_plan_stops_before_queue_and_pixels(self):
        with mock.patch.object(sys, 'argv', ['prepare_v4_low_angle_real.py', '--output', 'unused']), \
             mock.patch.object(entry, 'build_queue') as queue, \
             mock.patch.object(entry, 'apply_tone') as tone, mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                entry.main()
            queue.assert_not_called()
            tone.assert_not_called()

    def test_allocation_cannot_repeat_exact_state(self):
        self.assertEqual(entry.allocated_states({'strata': [{'planned_count': 1}]}), [{'planned_count': 1}])
        with self.assertRaisesRegex(ValueError, 'one-image'):
            entry.allocated_states({'strata': [{'planned_count': 2}]})


if __name__ == '__main__':
    unittest.main()
