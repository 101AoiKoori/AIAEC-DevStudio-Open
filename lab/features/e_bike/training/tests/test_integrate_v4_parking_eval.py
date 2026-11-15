"""Test parking eval registration using only temporary data."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    'bike_eval_integration', ROOT / 'lab/features/e_bike/training/scripts/integrate_v4_parking_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IntegrationTest(unittest.TestCase):
    def test_registration_preserves_train_and_rejects_repeated_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source, dist = root / 'source.json', root / 'dist.json'
            configs = [root / 'one.yaml', root / 'two.yaml']
            batch = root / 'batch'
            (batch / 'images/val').mkdir(parents=True)
            (batch / 'labels/val').mkdir(parents=True)
            original_image, original_label = root / 'original.png', root / 'original.txt'
            original_image.write_bytes(b'generated original')
            original_label.write_text('1 .5 .5 .1 .1\n', encoding='utf8')
            image, label = batch / 'images/val/a.png', batch / 'labels/val/a.txt'
            image.write_bytes(original_image.read_bytes())
            label.write_bytes(original_label.read_bytes())
            (batch / 'manifest.json').write_text('{}', encoding='utf8')
            (root / 'qa.json').write_text('{}', encoding='utf8')
            originals = {'records': [{'image': 'original.png', 'label': 'original.txt',
                                       'image_size': [640, 360], 'objects': [{'class': 'fallen'}],
                                       'tone': 'mild', 'background_parent': 'background.png', 'background_image': 'background.png',
                                       'background_source_video': 'source.mp4', 'background_source_pts_seconds': 1.0}]}
            row = dict(originals['records'][0], source_image='original.png', source_label='original.txt',
                       source_record_index=0, source_manifest='old.json', image='batch/images/val/a.png',
                       label='batch/labels/val/a.txt', split='val')
            rows = [{'id': f'train_{i}', 'image': f'train_{i}.png', 'split': 'train',
                     'selected_tasks': ['e_bike']} for i in range(930)]
            rows += [{'id': f'val_{i}', 'image': f'val_{i}.png', 'split': 'val',
                      'selected_tasks': ['e_bike']} for i in range(4)]
            source.write_text(json.dumps({'samples': rows}), encoding='utf8')
            dist.write_text(json.dumps({'additional_records': []}), encoding='utf8')
            config_text = ('data: ' + module.BASE_DATA + '\ntest_unavailable: true\n'
                           'data_overrides:\n  train:\n  - images: train/images\n    labels: train/labels\n')
            for config in configs:
                config.write_text(config_text, encoding='utf8')
            with mock.patch.multiple(module, ROOT=root, SOURCE=source, DISTRIBUTION=dist, CONFIGS=configs):
                module._integrate(batch, [row], originals, root / 'qa.json', batch / 'manifest.json', [])
                result = json.loads(source.read_text(encoding='utf8'))
                self.assertEqual(len(result['samples']), 936)
                self.assertEqual(result['samples'][-1]['tasks']['e_bike']['state'], 'complete_positive')
                config = yaml.safe_load(configs[0].read_text(encoding='utf8'))
                self.assertFalse(config['test_unavailable'])
                self.assertEqual(config['data_overrides']['train'], [{'images': 'train/images', 'labels': 'train/labels'}])
                self.assertEqual(config['data_overrides']['val'][0]['split'], 'val')
                with self.assertRaises(ValueError):
                    module._integrate(batch, [row], originals, root / 'qa.json', batch / 'manifest.json', [])

    def test_exact_image_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            for name in ('source.png', 'selected.png', 'occupied.png'):
                (root / name).write_bytes(b'identical')
            for name in ('source.txt', 'selected.txt'):
                (root / name).write_text('1 .5 .5 .1 .1\n')
            parent = {'image': 'source.png', 'label': 'source.txt', 'image_size': [640, 360],
                      'objects': [{'class': 'fallen'}], 'tone': 'original', 'background_parent': 'bg.png', 'background_image': 'bg.png',
                      'background_source_video': 'source.mp4', 'background_source_pts_seconds': 0.0}
            row = dict(parent, image='selected.png', label='selected.txt', source_image='source.png',
                       source_label='source.txt', source_record_index=0)
            with mock.patch.object(module, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Exact output'):
                module._validate([row], {'records': [parent]}, [{'image': 'occupied.png'}])


if __name__ == '__main__':
    unittest.main()
