"""Connect an explicitly assigned frame decision to direct-capture selection."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from datasets.scripts.common import package_direct_real as direct
from datasets.scripts.common.assign_split import assign_selected
from datasets.scripts.common.frame_environment import edit_database
from datasets.scripts.common.paths import WORKSPACE


class DirectFrameSelection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='direct_handoff_', dir=WORKSPACE / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / 'frames.sqlite3'
        self.image = self.root / 'camera.png'
        Image.new('RGB', (80, 60), 'gray').save(self.image)
        self.image_path = self.image.relative_to(WORKSPACE).as_posix()
        config = self.root / 'datasets/scripts/person/config/source.yaml'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({'task': 'person', 'pairing': {}}), encoding='utf-8')
        with edit_database(self.database) as writer:
            frame = writer.register(self.image_path, image_size=[80, 60], parent_group='fixture')
            label = writer.observe(frame, task='person', kind='label', scope='frame', evidence_key='truth',
                                   reference={'fixture': True}, payload={'presence': 'absent',
                                   'completeness': 'complete', 'scope': {'kind': 'frame'}, 'instances': []})
            self.pending = writer.decide(frame, task='person', role='empty_negative', split='unassigned',
                                         status='eligible', evidence_ids=[label], details={
                                             'image': self.image_path, 'source_kind': 'camera_photo',
                                             'label_evidence_id': label})
        outcome = assign_selected('person', 'empty_negative', 'train', [self.pending],
                                  database=self.database, root=self.root)[0]
        self.assertEqual(outcome['status'], 'assigned')
        self.assigned = outcome['assigned_record_id']
        self.source = {'databases': {'frame_environment': self.database.relative_to(WORKSPACE).as_posix()}}
        self.load = patch.object(direct, 'load_task_config', return_value=({}, self.source))
        self.history = patch.object(direct, 'read_source_history', return_value={})
        self.catalog = patch.object(direct, 'read_records', return_value=iter(()))
        for p in (self.load, self.history, self.catalog):
            p.start()
            self.addCleanup(p.stop)

    def test_assigned_id_selects_real_camera_and_existing_id_still_works(self):
        rows, _ = direct.prepare('person', split='train', frame_record_ids=[self.assigned])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1]['frame_environment']['record_id'], self.assigned)
        self.assertEqual(rows[0][6], '')
        legacy, _ = direct.prepare('person', split='train', record_ids=[rows[0][1]['id']])
        self.assertEqual(legacy, rows)
        with self.assertRaisesRegex(ValueError, 'Requested frame record'):
            direct.prepare('person', split='val', frame_record_ids=[self.assigned])
        with self.assertRaisesRegex(ValueError, 'Requested frame record'):
            direct.prepare('person', split='train', frame_record_ids=[self.pending])

    def test_cli_reports_frame_id_and_checks_evidence(self):
        out = io.StringIO()
        argv = ['package_direct_real', '--task', 'person', '--split', 'train',
                '--frame-record-id', self.assigned]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(out):
            direct.main()
        self.assertEqual(json.loads(out.getvalue())['sources'][0]['frame_record_id'], self.assigned)
        # Current qualification is revoked when its label evidence changes.
        with edit_database(self.database) as writer:
            frame = writer.conn.execute('SELECT frame_id FROM decisions WHERE record_id = ?',
                                        (self.assigned,)).fetchone()['frame_id']
            writer.observe(frame, task='person', kind='label', scope='frame', evidence_key='truth',
                           reference={'fixture': True}, payload={'presence': 'unknown'})
        with self.assertRaisesRegex(ValueError, 'Requested frame record'):
            direct.prepare('person', split='train', frame_record_ids=[self.assigned])

    def test_explicit_selection_keeps_file_identity_check(self):
        Image.new('RGB', (80, 60), 'black').save(self.image)
        with self.assertRaisesRegex(ValueError, 'pixels changed'):
            direct.prepare('person', split='train', frame_record_ids=[self.assigned])

    def test_frame_selection_matches_existing_catalog_record(self):
        manifest = WORKSPACE / 'datasets/scripts/person/config/source.yaml'
        row = {'id': 'catalog-entry', 'image': self.image_path, 'image_size': [80, 60],
               'source_kind': 'camera_photo', 'capture_identity': 'fixture'}
        with patch.object(direct, 'read_records', side_effect=lambda *a, **k: iter([(manifest, row, {})])):
            selected, _ = direct.prepare('person', split='train', frame_record_ids=[self.assigned])
            self.assertEqual(selected[0][1]['id'], 'catalog-entry')
            both, _ = direct.prepare('person', split='train', record_ids=['catalog-entry'],
                                     frame_record_ids=[self.assigned])
            self.assertEqual(both, selected)


if __name__ == '__main__':
    unittest.main()
