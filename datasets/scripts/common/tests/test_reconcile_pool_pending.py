"""Missing pool statuses must be explicit without replacing a reviewed decision."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from datasets.scripts.common.frame_environment import edit_database, query_frames
from datasets.scripts.common.paths import TASKS, WORKSPACE
from datasets.scripts.common.reconcile_pool_pending import reconcile


class ReconcilePoolPendingTests(unittest.TestCase):
    def test_marks_missing_tasks_once_and_preserves_human_decision(self):
        (WORKSPACE / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORKSPACE / 'tmp') as directory:
            root = Path(directory)
            image = root / 'frame.png'
            Image.new('RGB', (80, 60), 'gray').save(image)
            relative = image.relative_to(WORKSPACE).as_posix()
            pool = root / 'pool.json'
            pool.write_text(json.dumps({'role': 'shared_background_pool',
                                        'records': [{'image': relative}]}), encoding='utf-8')
            database = root / 'frames.sqlite3'
            with edit_database(database) as writer:
                frame_id = writer.register(relative, image_size=[80, 60], parent_group='source_video')
                shared = writer.observe(frame_id, kind='geometry', scope='shared_background',
                                        evidence_key='shared_background_space',
                                        reference={'method': 'fixture'}, payload={'status': 'observed'})
                previous = writer.decide(frame_id, task='person', role='composite_background',
                                         split='train', status='blocked', evidence_ids=[shared],
                                         details={'reason_codes': ['human_exclusion']}, origin='human')
            self.assertEqual(reconcile(pool=pool, database=database)['added_or_planned'], len(TASKS) - 1)
            applied = reconcile(pool=pool, database=database, write=True)
            self.assertEqual(applied['added_or_planned'], len(TASKS) - 1)
            self.assertEqual(reconcile(pool=pool, database=database, write=True)['added_or_planned'], 0)
            self.assertEqual(query_frames(task='person', role='composite_background', status=None,
                                          database=database)[0]['record_id'], previous)
            for task in TASKS - {'person'}:
                row = query_frames(task=task, role='composite_background', status=None,
                                   database=database)[0]
                self.assertEqual(row['status'], 'unknown')
                self.assertIn('task_label_not_reviewed', row['reason_codes'])
                self.assertIn('task_specific_placement_not_reviewed', row['reason_codes'])


if __name__ == '__main__':
    unittest.main()
