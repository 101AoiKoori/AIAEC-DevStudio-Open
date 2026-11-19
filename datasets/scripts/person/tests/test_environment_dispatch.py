"""Environment dispatch keeps usable split branches and reports source contacts."""
from types import SimpleNamespace
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from datasets.scripts.common.frame_environment import edit_database
from datasets.scripts.common.paths import WORKSPACE
from datasets.scripts.person.compile_background_dispatch import environment_dispatch
from datasets.scripts.person.planned_sources import _eligible_environment_background


def decision(frame, split, parent):
    return {'frame_id': frame, 'record_id': 'record_' + frame, 'image': 'datasets/raw/' + frame + '.jpg',
            'paths': ['datasets/raw/' + frame + '.jpg'], 'split': split, 'parent_group': parent,
            'scope': {'kind': 'frame'}, 'status': 'eligible', 'role': 'composite_background',
            'image_size': [640, 480], 'content_sha256': frame,
            'label': {'instances': [], 'completeness': 'complete', 'presence': 'absent'},
            'geometry': {'placeable_ground': {'reference_quad_px': [[0, 0], [639, 0], [639, 479], [0, 479]]}}}


class EnvironmentDispatch(unittest.TestCase):
    def test_shared_parent_contacts_and_reserved_branch_are_local(self):
        rows = [decision('a', 'train', 'shared'), decision('b', 'val', 'shared'),
                decision('c', 'train', 'reserved'), decision('d', 'test', 'reserved'),
                decision('e', 'train', 'unmatched')]
        groups = [SimpleNamespace(group_id='shared', paths=[], split={'primary': 'train', 'status': 'proposed'}),
                  SimpleNamespace(group_id='reserved', paths=[], split={'primary': 'test', 'status': 'reserved'})]
        with patch('datasets.scripts.common.frame_environment.query_frames', return_value=rows), \
             patch('datasets.scripts.common.frame_environment.verified_image'), \
             patch('datasets.scripts.common.source_ledger.config.load_source_config', return_value={'pairing': {}}), \
             patch('datasets.scripts.common.source_ledger.config.build_groups', return_value=groups), \
             patch('datasets.scripts.common.source_history.read_source_history', return_value={}), \
             patch('datasets.scripts.person.planned_sources.recorded_ground',
                   return_value=(None, SimpleNamespace(status='candidate'))):
            result = environment_dispatch('unused.sqlite')
        self.assertEqual([row['frame_id'] for row in result['records']], ['a', 'b', 'd', 'e'])
        self.assertTrue(all(_eligible_environment_background(row) for row in result['records']))
        self.assertEqual(result['source_contacts'], [{'parent_group': 'shared',
                                                      'splits': {'train': ['record_a'], 'val': ['record_b']}}])
        self.assertEqual(result['skipped']['source_reservation_conflict'], 1)
        self.assertEqual(result['reservation_conflicts'][0]['frame_id'], 'c')
        self.assertEqual(result['reservation_conflicts'][0]['reservation']['conflicts'][0]['reserved_split'], 'test')

    def test_explicit_reserved_reuse_keeps_both_splits(self):
        rows = [decision('f', 'train', 'shared'), decision('g', 'val', 'shared')]
        group = SimpleNamespace(group_id='shared', paths=[],
                                split={'primary': 'train', 'status': 'reserved',
                                       'same_source_allowed_in': ['val']})
        with patch('datasets.scripts.common.frame_environment.query_frames', return_value=rows), \
             patch('datasets.scripts.common.frame_environment.verified_image'), \
             patch('datasets.scripts.common.source_ledger.config.load_source_config', return_value={'pairing': {}}), \
             patch('datasets.scripts.common.source_ledger.config.build_groups', return_value=[group]), \
             patch('datasets.scripts.common.source_history.read_source_history', return_value={}), \
             patch('datasets.scripts.person.planned_sources.recorded_ground',
                   return_value=(None, SimpleNamespace(status='candidate'))):
            result = environment_dispatch('unused.sqlite')
        self.assertEqual({row['split'] for row in result['records']}, {'train', 'val'})
        self.assertEqual(len(result['source_contacts']), 1)
        self.assertFalse(result['reservation_conflicts'])

    def test_database_records_from_two_splits_are_retained(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE / 'tmp') as temporary:
            root = Path(temporary)
            database = root / 'frames.sqlite3'
            for index, split in enumerate(('train', 'val')):
                image = root / f'{split}.png'
                Image.new('RGB', (80, 60), (index * 100, 0, 0)).save(image)
                path = image.relative_to(WORKSPACE).as_posix()
                with edit_database(database) as writer:
                    frame = writer.register(path, image_size=[80, 60], content_sha256=sha256(image.read_bytes()).hexdigest(),
                                            parent_group='fixture_shared')
                    label = writer.observe(frame, task='person', kind='label', scope='frame', evidence_key='label',
                                           reference={'fixture': True}, payload={'instances': [], 'completeness': 'complete',
                                                                                 'presence': 'absent'})
                    geometry = writer.observe(frame, task='person', kind='geometry', scope='frame', evidence_key='ground',
                                              reference={'fixture': True}, payload={'placeable_ground': {
                                                  'reference_quad_px': [[0, 0], [79, 0], [79, 59], [0, 59]]}})
                    writer.decide(frame, task='person', role='composite_background', split=split, status='eligible',
                                  evidence_ids=[label, geometry], details={'image': path, 'scope': {'kind': 'frame'},
                                      'label_evidence_id': label, 'geometry_evidence_id': geometry})
            with patch('datasets.scripts.common.source_ledger.config.load_source_config', return_value={'pairing': {}}), \
                 patch('datasets.scripts.common.source_ledger.config.build_groups', return_value=[]), \
                 patch('datasets.scripts.common.source_history.read_source_history', return_value={}), \
                 patch('datasets.scripts.person.planned_sources.recorded_ground',
                       return_value=(None, SimpleNamespace(status='candidate'))):
                result = environment_dispatch(database)
            self.assertEqual({row['split'] for row in result['records']}, {'train', 'val'})
            self.assertEqual(result['source_contacts'][0]['parent_group'], 'fixture_shared')


if __name__ == '__main__':
    unittest.main()
