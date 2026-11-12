import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_fire_videos import ROOT, prepare, workspace_path
from generate_fire_dataset import generate, write_image
from export_fire_labels import export


class FireDatasetTest(unittest.TestCase):
    def test_synthetic_and_reviewed_export(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            base = Path(folder)
            image = base / 'frame.png'
            write_image(image, np.full((200, 300, 3), 180, dtype=np.uint8))
            relative = image.relative_to(ROOT).as_posix()
            regions = base / 'regions.json'
            regions.write_text(json.dumps([{'split': 'train', 'verified_empty': True, 'image': relative,
                                             'quad': [[40, 40], [100, 40], [100, 120], [40, 120]]}]), encoding='utf-8')
            out = base / 'synthetic'
            sticker = base / 'sticker.png'
            write_image(sticker, np.full((40, 40, 3), (10, 40, 240), dtype=np.uint8))
            self.assertEqual(generate(regions, sticker, out, 20, 12), 20)
            labels = list((out / 'labels/train').glob('*.txt'))
            self.assertTrue(any(not path.read_text().strip() for path in labels))
            box = next(path.read_text().split() for path in labels if path.read_text().strip())
            self.assertEqual(box[0], '0')
            self.assertTrue(all(0 < float(v) < 1 for v in box[1:]))
            with self.assertRaises(ValueError):
                generate(regions, sticker, out, 1, 1)
            source = base / 'annotations.json'
            data = [{'image': relative, 'split': 'val', 'reviewed': True,
                     'fires': [[40, 40, 80, 120]],
                     'names': [{'box': [110, 20, 200, 50], 'text': '示例楼宇'}]}]
            source.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            exported = base / 'export'
            self.assertEqual(export(source, exported), 1)
            self.assertIn('示例楼宇', (exported / 'ocr/val.txt').read_text(encoding='utf-8'))
            self.assertEqual(len((exported / 'fire/labels/val/frame_000001.txt').read_text().split()), 5)
            self.assertEqual(len((exported / 'name_sign/labels/val/frame_000001.txt').read_text().split()), 5)
            data[0]['reviewed'] = False
            source.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            with self.assertRaises(ValueError):
                export(source, base / 'unreviewed')

    def test_external_paths_rejected(self):
        with self.assertRaises(ValueError):
            workspace_path(str(ROOT.parent))

    def test_video_extraction_and_split(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            base = Path(folder)
            source = base / 'videos'
            (source / 'train').mkdir(parents=True)
            video = source / 'train' / 'session.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                            '-i', 'color=red:s=64x64:r=2:d=2', '-c:v', 'mpeg4', str(video)], check=True)
            output = base / 'frames'
            manifest = prepare(source, output, 1.0)
            self.assertEqual(manifest[0]['split'], 'train')
            self.assertEqual(manifest[0]['status'], 'pending_review')
            self.assertEqual(len(list((output / 'images/train/video_0001').glob('*.png'))), 2)
            with self.assertRaises(ValueError):
                prepare(source, output, 1.0)


if __name__ == '__main__':
    unittest.main()
