"""Exercise V4 candidate registration without touching workspace datasets."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    'bike_integration', ROOT / 'lab/features/e_bike/training/scripts/integrate_v4_masked_train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IntegrationTest(unittest.TestCase):
    def test_registers_candidates_and_refuses_duplicate(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            source, distribution = root / 'source.json', root / 'distribution.json'
            configs = [root / 'one.yaml', root / 'two.yaml']
            batch = root / 'batch'
            (batch / 'images/train').mkdir(parents=True)
            (batch / 'labels/train').mkdir(parents=True)
            image = batch / 'images/train/x.png'
            image.write_bytes(b'candidate')
            (batch / 'labels/train/x.txt').write_text('0 .5 .5 .1 .1\n')
            parent = root / 'parent.jpg'
            parent.write_bytes(b'parent')
            (batch / 'manifest.json').write_text('{}')
            (batch / 'qa.json').write_text('{}')
            vals = []
            for index in range(4):
                path = root / f'val_{index}.png'
                path.write_bytes(bytes([index]))
                vals.append({'id': f'val_{index}', 'image': path.name, 'split': 'val',
                             'selected_tasks': ['e_bike']})
            source.write_text(json.dumps({'samples': vals}))
            distribution.write_text(json.dumps({'additional_records': []}))
            for config in configs:
                config.write_text('data: existing.yaml\n', encoding='utf8')
            row = {'image': 'batch/images/train/x.png', 'label': 'batch/labels/train/x.txt',
                   'source_image': 'parent.jpg', 'image_size': [640, 480],
                   'crop_xyxy': [160, 0, 800, 480], 'tone_profile': 'mild',
                   'objects': [{'foreground_source': {'path': 'bike.png'}}],
                   'source_background': {'source_video': 'source.mp4', 'seconds': 1.0,
                                         'image': 'background.png'}}
            with mock.patch.multiple(module, ROOT=root, SOURCE=source, DISTRIBUTION=distribution,
                                     CONFIGS=configs):
                module._integrate(batch, [row], batch / 'qa.json', batch / 'manifest.json')
                result = json.loads(source.read_text(encoding='utf8'))
                self.assertEqual(len(result['samples']), 6)
                self.assertEqual(result['samples'][-1]['tasks']['e_bike']['state'], 'complete_positive')
                self.assertIn('data_overrides:', configs[0].read_text(encoding='utf8'))
                with self.assertRaises(ValueError):
                    module._integrate(batch, [row], batch / 'qa.json', batch / 'manifest.json')


if __name__ == '__main__':
    unittest.main()
