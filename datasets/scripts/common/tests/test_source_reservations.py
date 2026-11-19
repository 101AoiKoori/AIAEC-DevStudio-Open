"""Parent reservation preview preserves uncertainty and explicit reuse."""
import unittest

from datasets.scripts.common.source_ledger.config import build_groups
from datasets.scripts.common.source_ledger.reservations import candidate_reservation, matching_groups


class ReservationPreview(unittest.TestCase):
    def groups(self, *items):
        return build_groups('person', {'task_real': list(items)})

    def test_same_parent_different_frames_and_historical_alias(self):
        groups = self.groups({'id': 'route', 'paths': ['Archive/original'],
                              'split': {'status': 'reserved', 'primary': 'test'}})
        history = (('datasets/current', 'Archive/original'),)
        for frame in ('frame1.png', 'frame2.png'):
            result = candidate_reservation({'paths': ['datasets/current/' + frame]}, 'train', groups, history)
            self.assertEqual(result['status'], 'conflict')
            self.assertEqual(result['conflicts'], [{'group': 'route', 'reserved_split': 'test'}])

    def test_unknown_ambiguous_and_explicit_reuse(self):
        group = {'id': 'route', 'paths': ['Archive/route'], 'split': {
            'status': 'reserved', 'primary': 'test', 'same_source_allowed_in': ['train']}}
        groups = self.groups(group)
        self.assertEqual(candidate_reservation({'parent_group': 'other'}, 'train', groups, ())['status'], 'unknown')
        row = {'parent_group': 'route'}
        self.assertEqual(candidate_reservation(row, 'train', groups, ())['status'], 'no_declared_conflict')
        duplicate = {'id': 'second', 'paths': ['Archive/route'], 'split': {'status': 'proposed', 'primary': 'val'}}
        result = candidate_reservation({'paths': ['Archive/route/a.png']}, 'train', self.groups(group, duplicate), ())
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['conflicts'], [])

    def test_proposed_is_not_a_reservation_and_person_matching_is_unchanged(self):
        groups = self.groups({'id': 'route', 'paths': ['Archive/route'],
                              'split': {'status': 'proposed', 'primary': 'test'}})
        self.assertEqual(candidate_reservation({'parent_group': 'route'}, 'train', groups, ())['status'],
                         'no_declared_conflict')
        self.assertEqual(matching_groups({'parent_group': 'missing'}, groups, ()), [])


if __name__ == '__main__':
    unittest.main()
