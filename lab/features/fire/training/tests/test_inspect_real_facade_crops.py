"""Keep task-use filtering scoped to one source relation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import inspect_real_facade_crops as subject


class SelectionTests(unittest.TestCase):
    def test_requires_each_dimension_on_same_relation(self):
        records = []
        for key in subject.PATCHES:
            records.append({'path': key + '.jpg', 'task_use_status': [
                {'source_id': key, 'task': 'fire_sticker', 'split': 'train',
                 'role': 'background', 'status': 'candidate'},
                {'source_id': key, 'task': 'other', 'split': 'test',
                 'role': 'background', 'status': 'reserved'}]})
        self.assertEqual(len(subject.select_sources({'records': records})), 3)
        records[0]['task_use_status'][0]['split'] = 'test'
        records[0]['task_use_status'].append({'source_id': list(subject.PATCHES)[0],
            'task': 'other', 'split': 'train', 'role': 'background', 'status': 'candidate'})
        with self.assertRaises(ValueError):
            subject.select_sources({'records': records})

    def test_duplicate_source_path_is_rejected(self):
        records = [{'path': 'same.jpg', 'task_use_status': [
            {'source_id': key, 'task': 'fire_sticker', 'split': 'train',
             'role': 'background', 'status': 'candidate'}]} for key in subject.PATCHES]
        with self.assertRaises(ValueError):
            subject.select_sources({'records': records})


if __name__ == '__main__':
    unittest.main()
