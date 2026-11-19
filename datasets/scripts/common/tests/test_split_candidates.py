"""Planning inventory keeps source decisions distinct from assignments."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from datasets.scripts.common.frame_environment import edit_database, query_frames, query_split_candidates
from datasets.scripts.common.paths import WORKSPACE


class SplitCandidates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='split_candidates_', dir=WORKSPACE / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / 'frames.sqlite3'
        with edit_database(self.database) as writer:
            for index, split in enumerate(('unassigned', 'train', 'test', 'unassigned')):
                # Distinct identities; registering a supplied digest does not read the image.
                frame = writer.register(f'tmp/candidate_{index}.png', image_size=[10, 10],
                                        parent_group=f'parent_{index}', content_sha256=f'{index:064x}')
                evidence = writer.observe(frame, task='person', kind='label', scope='frame',
                                          evidence_key='candidate', reference={'fixture': index},
                                          payload={'revision': 1}, origin='machine')
                writer.decide(frame, task='person', role='empty_negative', split=split,
                              status='eligible', evidence_ids=[evidence], details={}, origin='machine')
                if index == 3:
                    writer.observe(frame, task='person', kind='label', scope='frame',
                                   evidence_key='candidate', reference={'fixture': index},
                                   payload={'revision': 2}, origin='machine')

    def test_candidate_state_is_not_assignment_and_obsolete_evidence_is_excluded(self):
        rows = query_split_candidates(task='person', role='empty_negative', split='train', database=self.database)
        self.assertEqual({row['split'] for row in rows}, {'train', 'unassigned'})
        self.assertEqual({row['assignment_state'] for row in rows}, {'assigned', 'pending'})
        self.assertEqual({row['parent_group'] for row in rows}, {'parent_0', 'parent_1'})
        self.assertTrue(all(row['record_id'] and row['evidence'] for row in rows))
        self.assertTrue(all(row['candidate_for_split'] == 'train' for row in rows))
        self.assertEqual(len(query_frames(task='person', role='empty_negative', split='train',
                                          database=self.database)), 1)
        with self.assertRaises(ValueError):
            query_split_candidates(task='person', role='empty_negative', split='unassigned', database=self.database)

    def test_existing_cli_reports_real_split_without_writing(self):
        before = self.database.read_bytes()
        completed = subprocess.run(
            [sys.executable, '-m', 'datasets.scripts.common.backfill_frame_environment', 'candidates',
             '--task', 'person', '--role', 'empty_negative', '--split', 'val', '--database', str(self.database)],
            cwd=WORKSPACE, text=True, encoding='utf-8', capture_output=True, check=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report['requested_split'], 'val')
        self.assertEqual([(row['split'], row['assignment_state']) for row in report['records']],
                         [('unassigned', 'pending')])
        self.assertEqual(report['records'][0]['reservation']['status'], 'unknown')
        self.assertEqual(self.database.read_bytes(), before)

    def test_cli_assigns_only_explicit_record_and_preserves_unknown_parent(self):
        rows = query_split_candidates(task='person', role='empty_negative', split='val', database=self.database)
        selected = next(row for row in rows if row['split'] == 'unassigned')
        completed = subprocess.run(
            [sys.executable, '-m', 'datasets.scripts.common.backfill_frame_environment', 'assign',
             '--task', 'person', '--role', 'empty_negative', '--split', 'val',
             '--record-id', selected['record_id'], '--database', str(self.database)],
            cwd=WORKSPACE, text=True, encoding='utf-8', capture_output=True, check=True,
        )
        outcome = json.loads(completed.stdout)['records'][0]
        self.assertEqual(outcome['status'], 'assigned')
        self.assertEqual(outcome['reservation']['status'], 'unknown')
        self.assertEqual(len(query_frames(task='person', role='empty_negative', split='val',
                                          database=self.database)), 1)

    def test_cli_distinguishes_partial_and_failed_results(self):
        rows = query_split_candidates(task='person', role='empty_negative', split='val', database=self.database)
        selected = next(row for row in rows if row['split'] == 'unassigned')
        command = [sys.executable, '-m', 'datasets.scripts.common.backfill_frame_environment', 'assign',
                   '--task', 'person', '--role', 'empty_negative', '--split', 'val',
                   '--database', str(self.database)]
        partial = subprocess.run([*command, '--record-id', selected['record_id'], '--record-id', 'missing'],
                                 cwd=WORKSPACE, text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(partial.returncode, 2)
        report = json.loads(partial.stdout)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['counts'], {'assigned': 1, 'unchanged': 0, 'failed': 1})
        self.assertEqual(len(query_frames(task='person', role='empty_negative', split='val',
                                          database=self.database)), 1)
        failed = subprocess.run([*command, '--record-id', 'missing'], cwd=WORKSPACE, text=True,
                                encoding='utf-8', capture_output=True)
        self.assertEqual(failed.returncode, 1)
        self.assertEqual(json.loads(failed.stdout)['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
