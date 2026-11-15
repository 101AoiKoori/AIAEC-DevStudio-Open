import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/e_bike/training/scripts/prepare_e_bike_sources.py'
spec = importlib.util.spec_from_file_location('prepare_e_bike_sources', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareSourcesTest(unittest.TestCase):
    def test_frames_remain_grouped_and_are_not_assigned_a_split(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            videos = [root / 'a.mp4', root / 'b.mp4']
            for video in videos:
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                                '-i', 'color=c=red:s=64x64:d=2:r=5', '-pix_fmt', 'yuv420p',
                                str(video)], check=True)
            out = root / 'review'
            records = module._prepare(videos, out, 1)
            self.assertEqual([len(item['frames']) for item in records], [2, 2])
            self.assertTrue(all(item['split'] is None for item in records))
            self.assertTrue(all(item['status'] == 'pending_manual_review' for item in records))
            self.assertTrue(all((out / frame).is_file() for item in records for frame in item['frames']))
            self.assertTrue((out / 'review.html').is_file())
            with self.assertRaises(ValueError):
                module._prepare(videos, out, 1)

    def test_failed_video_is_not_treated_as_empty_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            video = root / 'bad.mp4'
            video.write_bytes(b'not a video')
            out = root / 'review'
            with self.assertRaises(RuntimeError):
                module._prepare([video], out, 1)
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['sources'][0]['status'], 'failed')
            self.assertNotEqual(manifest['sources'][0]['status'], 'pending_manual_review')


if __name__ == '__main__':
    unittest.main()
