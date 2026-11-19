"""Source reservations and byte identity in the person package index."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets.scripts.common.source_ledger.config import build_groups
from datasets.scripts.person import package_person_dataset as package


class PersonPackagePolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = 'Archive/Assets/capture/route.mp4'

    def group(self, *, primary='test', status='reserved', allowed=()):
        return build_groups('person', {'task_real': [{
            'id': 'route', 'paths': [self.parent], 'split': {
                'primary': primary, 'status': status, 'same_source_allowed_in': list(allowed)}}]})

    def row(self, split, data, *, group='route_group', parent=None, identity=None):
        path = self.root / split / ('image_' + str(len(list(self.root.rglob('*.png')))) + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {'split': split, 'image': path.relative_to(self.root).as_posix(), 'source_group': group,
                'parent_video': parent or self.parent, 'capture_identity': identity}

    def run_finalize(self, rows, groups):
        output = self.root / 'output'
        output.mkdir(exist_ok=True)
        with patch.object(package, 'ROOT', self.root), patch.object(package, 'dataset_dir', return_value=output), \
             patch.object(package, '_source_policy', return_value=(groups, ())), \
             patch.object(package, 'read_source_history', return_value=()), \
             patch.object(package, '_inspect_split', side_effect=lambda split: (rows[split], {})), \
             patch.object(package.direct, 'workspace_file', side_effect=lambda path: self.root / path):
            result = package.finalize(list(rows))
        return result, json.loads((output / 'workspace_sources.json').read_text(encoding='utf-8'))

    def test_reserved_car_source_follows_config_not_name(self):
        car = {'source_group': 'car_real_capture_20261112', 'parent_video': self.parent}
        with self.assertRaisesRegex(ValueError, 'reserved for test'):
            package._check_reservations(car, 'train', self.group(), ())
        package._check_reservations(car, 'test', self.group(), ())
        package._check_reservations(car, 'train', self.group(primary='train'), ())

    def test_declared_shared_parent_is_reported(self):
        train = self.row('train', b'one', identity='capture-a')
        val = self.row('val', b'two')
        _, result = self.run_finalize({'train': [train], 'val': [val]},
                                      self.group(primary='train', allowed=('val',)))
        self.assertEqual(result['selection_mode'], 'manifest_only')
        self.assertTrue(result['source_reuse'])
        self.assertEqual(result['source_reuse'][0]['evaluation_scope'], 'same_source_diagnostic')
        self.assertEqual(next(row for row in result['records'] if row['split'] == 'val')['evaluation_scope'],
                         'same_source_diagnostic')

    def test_undeclared_cross_split_reuse_is_reported(self):
        train = self.row('train', b'one', identity='capture-a')
        val = self.row('val', b'two')
        _, result = self.run_finalize({'train': [train], 'val': [val]}, [])
        self.assertTrue(result['source_reuse'])
        self.assertTrue(all(item['evaluation_scope'] == 'same_source_diagnostic'
                            for item in result['source_reuse']))

    def test_pure_synthetic_parent_reuse_is_reported(self):
        train = self.row('train', b'first synthetic')
        val = self.row('val', b'second synthetic')
        _, result = self.run_finalize({'train': [train], 'val': [val]},
                                      self.group(primary='train', allowed=('val',)))
        self.assertTrue(any(item.get('parent_video') == self.parent
                            for item in result['source_reuse']))
        self.assertEqual(next(row for row in result['records'] if row['split'] == 'val')['evaluation_scope'],
                         'same_source_diagnostic')

    def test_shared_blender_scene_is_reported(self):
        scene = 'datasets/assets/blender/aiaec_dataset.blend'
        train = self.row('train', b'synthetic one', group='arena', parent='parent-a')
        test = self.row('test', b'synthetic two', group='arena', parent='parent-b')
        train['mother_scene'] = test['mother_scene'] = scene
        groups = build_groups('person', {'blender': [{'id': 'arena', 'path': scene,
            'split': {'primary': 'shared', 'status': 'confirmed'}}]})
        _, result = self.run_finalize({'train': [train], 'test': [test]}, groups)
        entry = next(item for item in result['source_reuse'] if item.get('mother_scene') == scene)
        self.assertEqual(entry['groups'], ['arena'])
        self.assertEqual(entry['evaluation_scope'], 'same_source_diagnostic')

    def test_reserved_pure_synthetic_parent_cannot_cross_without_permission(self):
        train = self.row('train', b'synthetic one')
        test = self.row('test', b'synthetic two')
        with self.assertRaisesRegex(ValueError, 'reserved for train'):
            self.run_finalize({'train': [train], 'test': [test]}, self.group(primary='train'))

    def test_identical_bytes_rejected_even_for_different_source(self):
        train = self.row('train', b'exact same', group='one', parent='parent-one')
        test = self.row('test', b'exact same', group='two', parent='parent-two')
        with self.assertRaisesRegex(ValueError, 'Identical person output images'):
            self.run_finalize({'train': [train], 'test': [test]}, [])

    def test_alias_maps_to_reserved_parent(self):
        row = {'source_group': 'other', 'parent_video': 'datasets/raw/moved/route.mp4'}
        history = (('datasets/raw/moved', 'Archive/Assets/capture'),)
        with self.assertRaisesRegex(ValueError, 'reserved for test'):
            package._check_reservations(row, 'train', self.group(), history)


if __name__ == '__main__':
    unittest.main()
