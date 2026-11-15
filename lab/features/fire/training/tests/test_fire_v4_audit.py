"""Check fire size statistics against a small synthetic, uncommitted fixture."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_fire_sticker_v4.py'
SPEC = importlib.util.spec_from_file_location('fire_v4', SCRIPT)
fire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fire)


class FireAuditTests(unittest.TestCase):
    def test_exported_pixel_sizes_and_empty_labels_are_counted(self):
        parent = fire.ROOT / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root = Path(temp)
            data = root / 'dataset'
            for split in ('train', 'val', 'test'):
                (data / 'images' / split).mkdir(parents=True)
                (data / 'labels' / split).mkdir(parents=True)
                Image.new('RGB', (640, 480)).save(data / 'images' / split / 'a.png')
                label = '0 0.5 0.5 0.01 0.02\n0 0.5 0.5 0.1 0.1\n' if split == 'train' else ''
                (data / 'labels' / split / 'a.txt').write_text(label, encoding='utf-8')
            Image.new('RGB', (1280, 720)).save(data / 'images/train/b.png')
            (data / 'labels/train/b.txt').write_text('0 0.5 0.5 0.02 0.1\n', encoding='utf-8')
            (data / 'data.yaml').write_text(yaml.safe_dump({'names': ['fire_sticker'],
                **{s: 'images/' + s for s in ('train', 'val', 'test')}}), encoding='utf-8')
            config = {'data': 'dataset/data.yaml', 'task': 'detect', 'model_key': 'fire_sticker', 'imgsz': 640}
            report = fire.audit_target_sizes(root, config)
            self.assertEqual(report['status'], 'descriptive_only')
            self.assertEqual(report['splits']['train']['instances'], 3)
            self.assertEqual(report['splits']['train']['exported_short_side_pixels']['0_8'], 1)
            self.assertEqual(report['splits']['train']['exported_short_side_pixels']['16_32'], 1)
            self.assertEqual(report['splits']['train']['estimated_input_short_side_pixels']['8_16'], 1)
            self.assertEqual(report['splits']['train']['exported_short_side_pixels']['over_32'], 1)
            self.assertEqual(report['splits']['val']['negative_images'], 1)
            self.assertIn('do not establish verified negative', report['scope'])
            self.assertIn('not verified source resolution', report['scope'])


if __name__ == '__main__':
    unittest.main()
