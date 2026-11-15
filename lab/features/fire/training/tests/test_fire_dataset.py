import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from prepare_fire_videos import ROOT, prepare, workspace_path
from generate_fire_dataset import generate, write_image
from export_fire_labels import export
import prepare_fire_videos as prepare_module
import generate_fire_dataset as generate_module
import export_fire_labels as export_module


class FireDatasetTest(unittest.TestCase):
    def setUp(self):
        # These tests cover image/label transformations; database behavior is
        # exercised separately by the production guard integration tests.
        @contextmanager
        def fixture_guard(operation, input_paths, output_paths, parameters, **kwargs):
            if any(Path(path).exists() for path in output_paths):
                raise ValueError('fixture output already exists')
            refs = {Path(path).resolve().relative_to(ROOT).as_posix(): [] for path in input_paths
                    if Path(path).is_file()}
            yield SimpleNamespace(reused=False, source_refs=refs)

        for module in (prepare_module, generate_module, export_module):
            self.enterContext(patch.object(module, 'generation_guard', fixture_guard))

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

    def test_explicit_plan_and_review_safety(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as folder:
            base = Path(folder)
            image, sticker, regions = base / 'frame.png', base / 'sticker.png', base / 'regions.json'
            write_image(image, np.full((200, 300, 3), 180, dtype=np.uint8))
            write_image(sticker, np.full((40, 40, 3), 120, dtype=np.uint8))
            common = {'split': 'train', 'image': image.relative_to(ROOT).as_posix(),
                      'review_state': 'ai_reviewed_empty_candidate'}
            quads = [[[20, 20], [60, 20], [60, 80], [20, 80]],
                     [[120, 20], [160, 20], [160, 80], [120, 80]]]
            plan = [dict(common, quad=q, sticker_count=n) for n in (0, 1, 2) for q in quads]
            regions.write_text(json.dumps(plan), encoding='utf-8')
            out = base / 'planned'
            generate(regions, sticker, out, 3, 42)
            rows = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual([len(row['projected']) for row in rows], [0, 1, 2])
            self.assertTrue(all(row['review'] == 'pending' and row['background_review'] == 'ai_reviewed_empty_candidate' for row in rows))
            for index, invalid in enumerate((True, 1.5, '1', None, 3)):
                regions.write_text(json.dumps([dict(common, quad=quads[0], sticker_count=invalid)]), encoding='utf-8')
                target = base / f'invalid_{index}'
                with self.assertRaises(ValueError):
                    generate(regions, sticker, target, 1, 42)
                self.assertFalse(target.exists())
            conflict = [dict(common, quad=quads[0]),
                        {'split': 'train', 'image': common['image'], 'quad': quads[1], 'verified_empty': True}]
            regions.write_text(json.dumps(conflict), encoding='utf-8')
            with self.assertRaises(ValueError):
                generate(regions, sticker, base / 'conflict', 1, 42)
            self.assertFalse((base / 'conflict').exists())
            regions.write_text(json.dumps(plan), encoding='utf-8')
            with self.assertRaises(ValueError):
                generate(regions, sticker, base / 'incomplete', 2, 42)
            self.assertFalse((base / 'incomplete').exists())

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
            self.assertEqual(manifest[0]['fps'], 1.0)
            self.assertEqual(manifest[0]['frames'], 2)
            self.assertEqual(len(list((output / 'images/train/video_0001').glob('*.png'))), 2)
            with self.assertRaises(ValueError):
                prepare(source, output, 1.0)


if __name__ == '__main__':
    unittest.main()
