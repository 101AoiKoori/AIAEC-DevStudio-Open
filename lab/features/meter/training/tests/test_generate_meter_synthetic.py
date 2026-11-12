import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'generate_meter_synthetic.py'
spec = importlib.util.spec_from_file_location('meter_synthetic', SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class SyntheticMeterTests(unittest.TestCase):
    def test_projected_landmarks_and_label_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'batch'
            generator.generate(output, 8, 120)
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['samples']), 8)
            self.assertEqual(len(manifest['keypoint_order']), 8)
            for sample in manifest['samples']:
                name = sample['id']
                image = cv2.imread(str(output / 'images' / 'train' / (name + '.png')))
                self.assertEqual(image.shape, (640, 640, 3))
                tokens = (output / 'labels' / 'train' / (name + '.txt')).read_text().split()
                self.assertEqual(len(tokens), 29)  # class + box + 8*(x,y,visibility)
                self.assertEqual(tokens[0], '0')
                box = np.asarray([float(v) for v in tokens[1:5]])
                self.assertTrue(np.all(box >= 0) and np.all(box <= 1))
                self.assertGreater(box[2], 0)
                self.assertGreater(box[3], 0)
                for i, (point, vis) in enumerate(zip(sample['keypoints'], sample['visibility'])):
                    x, y, v = float(tokens[5 + i * 3]), float(tokens[6 + i * 3]), int(tokens[7 + i * 3])
                    self.assertEqual(v, vis)
                    if v:
                        self.assertTrue(0 <= x <= 1 and 0 <= y <= 1)
                        np.testing.assert_allclose([x * 640, y * 640], point, atol=.001)
                        self.assertLessEqual(abs(x - box[0]), box[2] / 2 + .002)
                        self.assertLessEqual(abs(y - box[1]), box[3] / 2 + .002)
                    else:
                        self.assertEqual((x, y), (0, 0))
                inverse = np.linalg.inv(np.asarray(sample['parameters']['homography']))
                original = generator.map_points(sample['keypoints'], inverse)
                for name, idx in (('temperature', 0), ('humidity', 2)):
                    scale = generator.SCALES[name]
                    np.testing.assert_allclose(original[idx], scale['center'], atol=.01)
                    t = (sample['values'][name] - scale['minimum']) / (scale['maximum'] - scale['minimum'])
                    angle = scale['angles'][0] + t * (scale['angles'][1] - scale['angles'][0])
                    np.testing.assert_allclose(original[idx + 1], generator.polar(scale['center'], scale['radius'] - 25, angle), atol=.01)
                self.assertEqual(len(sample['tick_lines']['temperature']), 81)
                self.assertEqual(len(sample['tick_lines']['humidity']), 51)
                self.assertIn('glare_ellipses', sample)

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'batch'
            generator.generate(output, 1, 1)
            marker = output / 'marker.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                generator.generate(output, 1, 2)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_generated_geometry_and_highlights_are_reproducible(self):
        first = generator.sample(np.random.default_rng(99))
        second = generator.sample(np.random.default_rng(99))
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])
        self.assertTrue(math.isfinite(first[3]['parameters']['yaw_deg']))

    def test_photo_augmentation_keeps_original_needle_positions(self):
        image, points, _, _, _ = generator.draw_dial(np.random.default_rng(42))
        reference = {'image_size': [640, 640], 'bbox_xyxy': [20, 20, 620, 620],
                     'status': 'unverified', 'source': 'test fixture',
                     'keypoints': [{'name': name, 'x': points[name][0], 'y': points[name][1], 'visibility': 2}
                                   for name in generator.KEYPOINTS]}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'photo'
            generator.generate(out, 3, 12, image, reference)
            manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['source_type'], 'annotated_single_image')
            for sample in manifest['samples']:
                inverse = np.linalg.inv(np.asarray(sample['parameters']['homography']))
                restored = generator.map_points(sample['keypoints'], inverse)
                np.testing.assert_allclose(restored, [points[k] for k in generator.KEYPOINTS], atol=.01)
                self.assertEqual(sample['values'], {'temperature': None, 'humidity': None})
                self.assertEqual(sample['reference_status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
