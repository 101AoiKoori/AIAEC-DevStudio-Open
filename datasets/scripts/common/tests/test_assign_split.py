"""Explicit inventory assignment is transactional and preserves source uncertainty."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets.scripts.common.assign_split import assign_selected
from datasets.scripts.common.frame_environment import edit_database, query_frames, _connect, FrameWriter
from datasets.scripts.common.paths import WORKSPACE


class AssignSplit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=WORKSPACE / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.database = self.root / 'frames.sqlite3'
        config = self.root / 'datasets/scripts/person/config/source.yaml'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({'task': 'person', 'pairing': {'task_real': [
            {'id': 'allowed', 'paths': ['Archive/allowed'], 'split': {'status': 'proposed', 'primary': 'train'}},
            {'id': 'reserved', 'paths': ['Archive/reserved'], 'split': {'status': 'reserved', 'primary': 'test'}},
        ]}}), encoding='utf-8')

    def add(self, name, parent, split='unassigned'):
        with edit_database(self.database) as writer:
            frame = writer.register(f'tmp/{name}.png', image_size=[10, 10], parent_group=parent,
                                    content_sha256=f'{name:0<64}'[:64])
            evidence = writer.observe(frame, task='person', kind='label', scope='frame', evidence_key='k',
                                      reference={'fixture': name}, payload={'revision': 1})
            record = writer.decide(frame, task='person', role='empty_negative', split=split,
                                   status='eligible', evidence_ids=[evidence], details={})
        return frame, record

    def assign(self, *ids, split='train'):
        return assign_selected('person', 'empty_negative', split, ids, database=self.database, root=self.root)

    def test_selected_record_moves_and_retry_is_idempotent(self):
        _, record = self.add('a', 'allowed')
        first = self.assign(record)[0]
        self.assertEqual(first['status'], 'assigned')
        self.assertEqual(self.assign(record)[0]['status'], 'unchanged')
        rows = query_frames(task='person', role='empty_negative', split='train', database=self.database)
        self.assertEqual([row['record_id'] for row in rows], [first['assigned_record_id']])
        self.assertEqual(query_frames(task='person', role='empty_negative', split='unassigned',
                                      database=self.database), [])

    def test_conflicts_and_stale_do_not_block_unrelated_selection(self):
        _, reserved = self.add('b', 'reserved')
        _, unknown = self.add('c', 'unknown')
        frame, stale = self.add('d', 'allowed')
        _, eligible = self.add('e', 'allowed')
        with edit_database(self.database) as writer:
            writer.observe(frame, task='person', kind='label', scope='frame', evidence_key='k',
                           reference={'fixture': 'd'}, payload={'revision': 2})
        result = self.assign(reserved, unknown, stale, eligible, 'missing')
        self.assertEqual([row['status'] for row in result],
                         ['conflict', 'assigned', 'stale', 'assigned', 'unknown_record'])
        self.assertEqual(result[1]['reservation']['status'], 'unknown')

    def test_other_frame_same_parent_is_reported_and_not_a_gate(self):
        self.add('f', 'allowed', split='test')
        _, pending = self.add('g', 'allowed')
        result = self.assign(pending)[0]
        self.assertEqual(result['status'], 'assigned')
        self.assertEqual(len(result['related_records']), 1)
        self.assertTrue(result['related_records'][0]['same_parent'])
        self.assertFalse(result['related_records'][0]['same_frame'])

    def test_same_frame_other_split_still_needs_writer_evidence(self):
        frame, _ = self.add('z', 'allowed', split='test')
        conn = _connect(self.database, writable=True)
        try:
            with conn:
                evidence = conn.execute('SELECT evidence_id FROM observations WHERE frame_id = ?',
                                        (frame,)).fetchone()['evidence_id']
                pending = FrameWriter(conn).decide(frame, task='person', role='empty_negative',
                                                   split='unassigned', status='eligible',
                                                   evidence_ids=[evidence], details={})
        finally:
            conn.close()
        result = self.assign(pending)[0]
        self.assertEqual(result['status'], 'conflict')
        self.assertIn('split reuse', result['reason'])
        self.assertEqual(len(query_frames(task='person', role='empty_negative', split='unassigned',
                                          database=self.database)), 1)

    def test_declared_reuse_is_preserved_and_obsolete_result_is_not_success(self):
        config = self.root / 'datasets/scripts/person/config/source.yaml'
        source = json.loads(config.read_text(encoding='utf-8'))
        source['pairing']['task_real'][0]['split']['same_source_allowed_in'] = ['test']
        config.write_text(json.dumps(source), encoding='utf-8')
        self.add('h', 'allowed', split='test')
        frame, pending = self.add('i', 'allowed')
        self.assertEqual(self.assign(pending)[0]['status'], 'assigned')
        with edit_database(self.database) as writer:
            writer.observe(frame, task='person', kind='label', scope='frame', evidence_key='k',
                           reference={'fixture': 'i'}, payload={'revision': 2})
        self.assertEqual(self.assign(pending)[0]['status'], 'stale')

    def test_changed_source_config_rolls_back_all_assignments(self):
        _, pending = self.add('j', 'allowed')
        config = self.root / 'datasets/scripts/person/config/source.yaml'
        original = type(config).read_bytes
        reads = 0

        def changing_read(path):
            nonlocal reads
            if path == config:
                reads += 1
                if reads == 2:
                    config.write_text('{"task":"person","pairing":{}}', encoding='utf-8')
            return original(path)

        with patch.object(type(config), 'read_bytes', changing_read):
            with self.assertRaisesRegex(ValueError, 'changed during assignment'):
                self.assign(pending)
        self.assertEqual(len(query_frames(task='person', role='empty_negative', split='unassigned',
                                          database=self.database)), 1)
        self.assertEqual(query_frames(task='person', role='empty_negative', split='train',
                                      database=self.database), [])


if __name__ == '__main__':
    unittest.main()
