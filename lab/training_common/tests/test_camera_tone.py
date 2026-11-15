"""Checks for deterministic, traceable camera-tone preview generation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from lab.training_common.camera_tone import apply_tone, _make_preview, PROFILES, statistics, write_image
from lab.training_common.production_guard import ProductionRun

# Public production entry/SQLite reuse checks are in lab/scripts/data/tests/test_shared_generation_entries.py.


def make_preview(root, image_paths, output, *, known, force_regression=False):
    states = []
    for image in image_paths:
        for profile, values in PROFILES.items():
            states.append({'id': image + ':' + profile, 'background': {'source': image},
                           'augmentation': {'profile': profile}, 'parameters': values,
                           'sources': [image], 'planned_count': 1})
    run = mock.Mock(spec=ProductionRun)
    run._token = 'fixture'
    run.reused = False
    run.active = True
    run.plan = {'strata': states}
    return _make_preview(root, image_paths, output, known=known, run=run,
                         force_regression=force_regression)


class CameraToneTest(unittest.TestCase):
    def setUp(self):
        self.workspace_tmp = Path(__file__).resolve().parents[3] / 'tmp'
        self.workspace_tmp.mkdir(exist_ok=True)

    def test_tone_reduces_value_and_saturation_without_geometry_or_mutation(self):
        image = np.empty((48, 64, 3), dtype=np.uint8)
        image[:] = (30, 100, 220)
        original = image.copy()
        first = apply_tone(image, value_gain=.70, saturation_gain=.60)
        second = apply_tone(image, value_gain=.70, saturation_gain=.60)
        self.assertTrue(np.array_equal(image, original))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, image.shape)
        before, after = statistics(image), statistics(first)
        self.assertLess(after['mean_gray_0_255'], before['mean_gray_0_255'])
        self.assertLess(after['mean_saturation_value_ge_16_0_255'], before['mean_saturation_value_ge_16_0_255'])

    def test_dark_saturation_statistics_and_invalid_inputs(self):
        self.assertIsNone(statistics(np.zeros((2, 2, 3), dtype=np.uint8))['mean_saturation_value_ge_16_0_255'])
        for invalid in (0, -0.1, 1.1, float('nan'), True):
            with self.assertRaises(ValueError):
                apply_tone(np.zeros((2, 2, 3), dtype=np.uint8), value_gain=invalid, saturation_gain=.8)
        with self.assertRaises(ValueError):
            apply_tone(np.zeros((2, 2), dtype=np.uint8), value_gain=.9, saturation_gain=.8)

    def test_preview_keeps_parent_and_refuses_reuse_or_over_720p(self):
        with tempfile.TemporaryDirectory(dir=self.workspace_tmp) as temporary:
            root = Path(temporary).resolve()
            source = root / 'source.png'
            write_image(source, np.full((480, 640, 3), (35, 100, 180), dtype=np.uint8))
            output = 'datasets/exports/mixed_scene/review'
            known = {'exposed_images': ['source.png'], 'exposed_video_frames': []}
            result = make_preview(root, ['source.png'], output, known=known)
            self.assertEqual(result['status'], 'complete_candidates')
            self.assertEqual(len(result['samples']), 2)
            manifest = json.loads((root / output / 'manifest.json').read_text(encoding='utf8'))
            for item in manifest['samples']:
                self.assertEqual(item['parent_ids'], ['source.png'])
                self.assertEqual(item['image_size'], [640, 480])
                self.assertEqual(item['split'], 'regression')
                self.assertEqual(item['task_state'], 'unreviewed')
                self.assertIsNone(item['labels'])
                decoded = cv2.imdecode(np.fromfile(str(root / output / item['image']), dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (480, 640, 3))
            with self.assertRaises(ValueError):
                make_preview(root, ['source.png'], output, known=known)
            too_large = root / 'too_large.png'
            write_image(too_large, np.zeros((721, 1280, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                make_preview(root, ['too_large.png'], 'datasets/exports/mixed_scene/invalid', known=known)
            with self.assertRaises(ValueError):
                make_preview(root, ['source.png'], '../outside', known=known)
            pending = make_preview(root, ['source.png'], 'datasets/exports/mixed_scene/pending',
                                   known={'exposed_images': [], 'exposed_video_frames': []})
            self.assertTrue(all(row['split'] == 'unassigned' and row['labels'] is None for row in pending['samples']))
            duplicate = root / 'duplicate.png'
            duplicate.write_bytes(source.read_bytes())
            reused = make_preview(root, ['duplicate.png'], 'datasets/exports/mixed_scene/reused', known=known)
            self.assertTrue(all(row['split'] == 'regression' for row in reused['samples']))

    def test_contact_buffer_uses_frame_pts_and_unexposed_frame_stays_unassigned(self):
        with tempfile.TemporaryDirectory(dir=self.workspace_tmp) as temporary:
            root = Path(temporary).resolve()
            seq = root / 'datasets/annotated/site_capture_20261112/sequence_640_v1/full_scene'
            seq.mkdir(parents=True)
            write_image(seq / 'frame_000001.png', np.full((16, 24, 3), 40, dtype=np.uint8))
            write_image(seq / 'frame_000002.png', np.full((16, 24, 3), 80, dtype=np.uint8))
            (seq / 'ffmpeg.log').write_text(
                '[showinfo] n: 0 pts: 0 pts_time:1.25\n'
                '[showinfo] n: 1 pts: 0 pts_time:1.252\n', encoding='utf8')
            video = 'datasets/raw/site_capture_20261112/videos/full_scene.mp4'
            known = {'exposed_images': [], 'exposed_video_frames': [{'video': video, 'seconds': 1.0}]}
            images = [(seq / f'frame_{n:06}.png').relative_to(root).as_posix() for n in (1, 2)]
            report = make_preview(root, images, 'datasets/exports/mixed_scene/buffer', known=known)
            self.assertEqual([s['split'] for s in report['samples']],
                             ['regression', 'regression', 'unassigned', 'unassigned'])
            self.assertEqual(report['samples'][0]['role_reason'], 'known_diagnostic_time_buffer')


if __name__ == '__main__':
    unittest.main()
