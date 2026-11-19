"""Cross-task candidate inventory preserves uncertainty and frame identity."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from datasets.scripts.common.discover_raw_background_candidates import discover, main
from datasets.scripts.common.frame_environment import edit_database
from datasets.scripts.common.paths import WORKSPACE


class DiscoverRawBackgroundCandidatesTests(unittest.TestCase):
    def setUp(self):
        (WORKSPACE / 'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='raw_candidate_', dir=WORKSPACE / 'tmp')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw = root / 'raw'
        self.person = self.raw / 'person' / 'frame.png'
        self.shared = self.raw / 'shared' / 'backgrounds' / 'frame.png'
        self.bin = self.raw / 'waste_bin_state' / 'frame.png'
        self.unregistered = self.raw / 'fire' / 'unregistered.png'
        for path, content in ((self.person, b'one'), (self.shared, b'one'),
                              (self.bin, b'two'), (self.unregistered, b'three')):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.db = root / 'frames.sqlite3'
        self.pool = root / 'pool.json'
        self.pool.write_text(json.dumps({'records': [
            {'image': self.relative(self.person)}, {'image': self.relative(self.shared)},
        ]}), encoding='utf-8')
        with edit_database(self.db) as writer:
            self.person_id = writer.register(
                self.relative(self.person), image_size=[10, 10], parent_group='capture_1',
                verified_copy_paths=[self.relative(self.shared)],
                content_sha256=sha256(b'one').hexdigest(),
            )
            self.bin_id = writer.register(
                self.relative(self.bin), image_size=[10, 10], parent_group='capture_2',
                content_sha256=sha256(b'two').hexdigest(),
            )
            writer.observe(self.person_id, task='person', kind='label', scope='frame',
                           evidence_key='labels', reference={},
                           payload={'scope': {'kind': 'frame'}, 'presence': 'absent',
                                    'completeness': 'complete'})
            writer.observe(self.person_id, kind='objects', scope='frame',
                           evidence_key='objects', reference={},
                           payload={'objects': [{'object_id': 'object_1', 'kind': 'person',
                                                 'xyxy': [1, 1, 3, 4]}]})
            first_space = writer.observe(self.person_id, kind='geometry',
                                         scope='shared_background', evidence_key='space',
                                         reference={}, payload={'status': 'observed',
                                                                'ground_polygon': [[0, 0], [1, 0], [1, 1]]})
            writer.decide(self.person_id, task='waste_bin_state', role='composite_background',
                          split='unassigned', status='eligible', evidence_ids=[first_space],
                          details={'reason_codes': ['fixture']})
            writer.observe(self.person_id, kind='geometry', scope='shared_background',
                           evidence_key='space', reference={}, payload={'status': 'revised'})
            writer.observe(self.bin_id, task='waste_bin_state', kind='label', scope='crop',
                           evidence_key='partial', reference={},
                           payload={'scope': {'kind': 'crop'}, 'presence': 'absent',
                                    'completeness': 'complete'})

    @staticmethod
    def relative(path):
        return path.relative_to(WORKSPACE).as_posix()

    def test_identity_pool_scope_and_superseded_decision(self):
        summary, frames = discover(database=self.db, raw_root=self.raw, pool_manifest=self.pool)
        self.assertEqual(summary['raw_image_paths'], 4)
        self.assertEqual(summary['registered_raw_image_paths'], 3)
        self.assertEqual(summary['raw_frame_identities'], 2)
        self.assertEqual(summary['shared_pool_records'], 2)
        self.assertEqual(summary['shared_pool_registered_frame_identities'], 1)
        self.assertEqual(summary['shared_pool_frame_identities_in_raw'], 1)
        self.assertEqual(summary['raw_frame_identities_outside_pool'], 1)
        self.assertEqual(summary['unregistered_raw_image_paths'], [self.relative(self.unregistered)])
        by_id = {frame['frame_id']: frame for frame in frames}
        person = by_id[self.person_id]
        self.assertEqual(person['raw_top_level_directories'], ['person', 'shared'])
        self.assertEqual(person['task_labels']['person']['state'], 'complete_negative')
        self.assertEqual(person['task_labels']['waste_bin_state']['state'], 'not_recorded')
        self.assertEqual(person['observed_objects'][0]['objects'][0]['kind'], 'person')
        self.assertEqual(person['composite_background']['waste_bin_state'][0]['status'], 'unknown')
        self.assertIn('evidence_superseded', person['composite_background']['waste_bin_state'][0]['reason_codes'])
        self.assertEqual(by_id[self.bin_id]['task_labels']['waste_bin_state']['state'],
                         'incomplete_or_local')

    def test_optional_json_and_jsonl_outputs(self):
        root = Path(self.temp.name)
        json_path = root / 'report.json'
        jsonl_path = root / 'frames.jsonl'
        with redirect_stdout(StringIO()):
            main(['--database', str(self.db), '--raw-root', str(self.raw),
                  '--pool-manifest', str(self.pool), '--output-json', str(json_path),
                  '--output-jsonl', str(jsonl_path)])
        self.assertEqual(json.loads(json_path.read_text(encoding='utf-8'))['summary']['raw_frame_identities'], 2)
        self.assertEqual(len(jsonl_path.read_text(encoding='utf-8').splitlines()), 2)

    def test_render_batch_extension_joins_shared_pool(self):
        extension = Path(self.temp.name) / 'render_batch.json'
        extension.write_text(json.dumps({'records': [{'image': self.relative(self.bin)}]}),
                             encoding='utf-8')
        pool = json.loads(self.pool.read_text(encoding='utf-8'))
        pool['extensions'] = [self.relative(extension)]
        self.pool.write_text(json.dumps(pool), encoding='utf-8')
        summary, frames = discover(database=self.db, raw_root=self.raw, pool_manifest=self.pool)
        self.assertEqual(summary['shared_pool_records'], 3)
        self.assertEqual(summary['shared_pool_registered_frame_identities'], 2)
        self.assertEqual(summary['raw_frame_identities_outside_pool'], 0)
        self.assertTrue(next(frame for frame in frames if frame['frame_id'] == self.bin_id)['in_shared_pool'])


if __name__ == '__main__':
    unittest.main()
