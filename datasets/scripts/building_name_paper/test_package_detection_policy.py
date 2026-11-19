"""Source and byte-identity policy for paper detection dataset publication."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import json
import hashlib

from datasets.scripts.building_name_paper import package_detection_dataset as package


class PaperSourcePolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = 'Archive/Assets/source/route.mp4'

    def row(self, split, name, contents, *, parent=None, source_id=None):
        image = self.root / name
        image.write_bytes(contents)
        record = {'image': f'exports/{split}/{name}', 'split': split,
                  'source_id': source_id or self.parent,
                  'parent_video': parent or self.parent}
        return record, image

    def apply_policy(self, entries, config=None, history=()):
        return package.source_policy([record for record, _ in entries],
                                     {record['image']: image for record, image in entries},
                                     config or {}, history)

    def test_same_parent_different_images_report_diagnostic(self):
        train = self.row('train', 'one.png', b'one')
        val = self.row('val', 'two.png', b'two')
        reuse = self.apply_policy([train, val])
        self.assertTrue(any(item.get('parent_video') == self.parent for item in reuse))
        self.assertEqual(val[0]['evaluation_scope'], 'same_source_diagnostic')
        self.assertNotIn('evaluation_scope', train[0])

    def test_identical_bytes_rejected_across_splits(self):
        entries = [self.row('train', 'one.png', b'same'), self.row('test', 'two.png', b'same')]
        with self.assertRaisesRegex(ValueError, 'Identical paper output images cross splits'):
            self.apply_policy(entries)

    def test_explicit_reservation_checked_against_actual_source(self):
        config = {'pairing': {'task_real': [{'id': 'route', 'paths': [self.parent],
                   'split': {'primary': 'test', 'status': 'reserved'}}]}}
        with self.assertRaisesRegex(ValueError, 'reserved for test'):
            self.apply_policy([self.row('train', 'one.png', b'one')], config)
        self.apply_policy([self.row('test', 'two.png', b'two')], config)
        config['pairing']['task_real'][0]['split']['same_source_allowed_in'] = ['train']
        self.apply_policy([self.row('train', 'three.png', b'three')], config)

    def test_historical_aliases_and_shared_blender_are_reported(self):
        train = self.row('train', 'one.png', b'one', parent='working/route.mp4', source_id='first')
        val = self.row('val', 'two.png', b'two', parent=self.parent, source_id='second')
        train[0]['mother_scene'] = val[0]['mother_scene'] = 'datasets/assets/blender/aiaec_dataset.blend'
        config = {'pairing': {'blender': [{'id': 'scene', 'path': train[0]['mother_scene'],
                   'split': {'primary': 'shared', 'status': 'confirmed'}}]}}
        reuse = self.apply_policy([train, val], config, (('working/route.mp4', self.parent),))
        self.assertTrue(any(item.get('parent_video') == self.parent for item in reuse))
        self.assertTrue(any(item.get('mother_scene') == train[0]['mother_scene'] for item in reuse))

    def test_evaluation_source_bytes_not_yet_copied_are_checked(self):
        train = self.row('train', 'one.png', b'one')
        evaluation = self.row('val', 'two.png', b'one')
        self.assertFalse((self.root / evaluation[0]['image']).exists())
        with self.assertRaisesRegex(ValueError, 'cross splits'):
            self.apply_policy([train, evaluation])

    def test_entrypoint_writes_reuse_to_training_source_index(self):
        base = self.root / 'exports'
        base.mkdir()
        plan = self.root / 'config/detection_evaluation.yaml'
        plan.parent.mkdir()
        plan.write_text('{}', encoding='utf-8')
        train_image = self.root / 'train.png'
        train_image.write_bytes(b'train-bytes')
        train_label = self.root / 'train.txt'
        train_label.write_text('', encoding='utf-8')
        train = {'image': 'train.png', 'label': 'train.txt', 'split': 'train',
                 'source_id': self.parent, 'parent_video': self.parent,
                 'review_evidence': 'config/detection_evaluation.yaml', 'instances': 0}
        evaluation = []
        for split in ('val', 'test'):
            source = self.root / (split + '.png')
            source.write_bytes(split.encode())
            target = base / split / 'images' / source.name
            label = base / split / 'labels' / (split + '.txt')
            row = {'image': target.relative_to(self.root).as_posix(),
                   'label': label.relative_to(self.root).as_posix(), 'split': split,
                   'source_id': self.parent, 'parent_video': self.parent, 'instances': 0,
                   'image_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'source_review': {}}
            evaluation.append((source, target, label, [], row))

        class Guard:
            reused = False
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def set_result(self, _result):
                pass

        with patch.object(package, 'ROOT', self.root), \
             patch.object(package, 'dataset_dir', return_value=base), \
             patch.object(package, 'split_dir', side_effect=lambda _task, split: base / split), \
             patch.object(package, 'load_task_config', return_value=({}, {'databases': {'frame_environment': 'db'}})), \
             patch.object(package, 'source_path', return_value=plan), \
             patch.object(package, 'train_entries', return_value=[(train_image, train_label, train)]), \
             patch.object(package, 'evaluation_entries', return_value=evaluation), \
             patch.object(package, 'generation_guard', return_value=Guard()), \
             patch.object(package, 'image_digest', side_effect=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()):
            package.main()
        sources = json.loads((base / 'workspace_sources.json').read_text(encoding='utf-8'))
        self.assertTrue(any(row.get('parent_video') == self.parent for row in sources['source_reuse']))
        self.assertTrue(all(row['evaluation_scope'] == 'same_source_diagnostic'
                            for row in sources['records'] if row['split'] != 'train'))


if __name__ == '__main__':
    unittest.main()
