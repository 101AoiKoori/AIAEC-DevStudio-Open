"""Person split handoff preserves unclaimed files while selecting manifest pairs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from datasets.scripts.person import package_person_dataset as package

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'lab/training_common'))
from engine.data import records as training_records


class ManifestHandoffTest(unittest.TestCase):
    def test_only_manifest_pairs_are_handed_off_and_missing_pairs_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'train'
            (split / 'images').mkdir(parents=True)
            (split / 'labels').mkdir()
            (split / 'images/owned.jpg').write_bytes(b'owned')
            (split / 'labels/owned.txt').write_text('', encoding='utf-8')
            (split / 'images/manual.jpg').write_bytes(b'manual')
            (split / 'labels/manual.txt').write_text('', encoding='utf-8')
            entry = {'image': 'train/images/owned.jpg', 'label': 'train/labels/owned.txt'}
            (split / 'manifest.json').write_text(json.dumps({'records': [entry]}), encoding='utf-8')
            with patch.object(package, 'ROOT', root), patch.object(package, 'split_dir', return_value=split), \
                    patch.object(package, '_training_record', side_effect=lambda row, *_: row), \
                    patch.object(package, 'split_coverage', return_value={}):
                rows, coverage = package._inspect_split('train')
                self.assertEqual(rows, [entry])
                self.assertEqual(coverage['unowned_images_excluded'], ['train/images/manual.jpg'])
                self.assertEqual((split / 'images/manual.jpg').read_bytes(), b'manual')
                (root / 'data.yaml').write_text(json.dumps({'names': ['a', 'b'], 'train': 'train/images'}),
                                                encoding='utf-8')
                (root / 'workspace_sources.json').write_text(json.dumps({
                    'selection_mode': 'manifest_only', 'records': [dict(entry, split='train')]}), encoding='utf-8')
                selected = training_records(root, {'data': 'data.yaml', 'task': 'detect', 'version': 'v1'}, 'train')
                self.assertEqual([row['image'] for row in selected], ['train/images/owned.jpg'])
                (split / 'labels/owned.txt').unlink()
                with self.assertRaisesRegex(ValueError, 'manifest does not match'):
                    package._inspect_split('train')


if __name__ == '__main__':
    unittest.main()
