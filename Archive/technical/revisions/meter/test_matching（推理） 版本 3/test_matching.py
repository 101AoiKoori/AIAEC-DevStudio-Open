import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws' / 'src' / 'aec_vision' / 'src'))
from aec_vision.meter import match_template, project_points


class TemplateTests(unittest.TestCase):
    def test_transform_and_unrelated(self):
        rng = np.random.RandomState(11)
        template = np.full((320, 320, 3), 230, np.uint8)
        for _ in range(160):
            x, y = rng.randint(20, 300, size=2)
            cv2.circle(template, (int(x), int(y)), int(rng.randint(2, 7)),
                       tuple(int(v) for v in rng.randint(0, 200, size=3)), -1)
        transform = np.float32([[.85, 0, 35], [0, .85, 25], [0, 0, 1]])
        image = cv2.warpPerspective(template, transform, (400, 400), borderValue=(255, 255, 255))
        matrix, status = match_template(template, image)
        self.assertEqual(status['status'], 'matched')
        actual = project_points([[80, 100], [250, 240]], matrix)
        expected = project_points([[80, 100], [250, 240]], transform)
        self.assertLess(np.max(np.abs(actual - expected)), 3)
        matrix, status = match_template(template, np.full_like(image, 255))
        self.assertIsNone(matrix)
        self.assertEqual(status['status'], 'unknown')

    def test_partial_dial_is_not_reported_as_valid(self):
        rng = np.random.RandomState(125)
        template = np.full((320, 320, 3), 240, np.uint8)
        for _ in range(500):
            x, y = rng.randint(15, 305, size=2)
            cv2.circle(template, (int(x), int(y)), 2,
                       tuple(int(v) for v in rng.randint(0, 150, size=3)), -1)
        cropped = cv2.warpPerspective(template, np.float32([[1, 0, 100], [0, 1, 0], [0, 0, 1]]),
                                      (320, 320), borderValue=(255, 255, 255))
        matrix, status = match_template(template, cropped)
        self.assertIsNone(matrix)
        self.assertEqual(status['status'], 'unknown')

    def test_rotation_perspective_and_background_only(self):
        rng = np.random.RandomState(57)
        template = np.full((360, 360, 3), 245, np.uint8)
        # Distinct texture in the dial and in the surrounding corners.
        for _ in range(500):
            x, y = rng.randint(8, 352, size=2)
            cv2.circle(template, (int(x), int(y)), int(rng.randint(1, 4)),
                       tuple(int(v) for v in rng.randint(0, 190, size=3)), -1)
        source = np.float32([[0, 0], [359, 0], [359, 359], [0, 359]])
        target = np.float32([[105, 30], [397, 105], [360, 408], [33, 325]])
        transform = cv2.getPerspectiveTransform(source, target)
        image = cv2.warpPerspective(template, transform, (450, 450), borderValue=(255, 255, 255))
        matrix, status = match_template(template, image)
        self.assertEqual(status['status'], 'matched')
        actual = project_points([[100, 100], [255, 250], [180, 180]], matrix)
        expected = project_points([[100, 100], [255, 250], [180, 180]], transform)
        self.assertLess(np.max(np.abs(actual - expected)), 4)

        # The surrounding room survives but the dial was removed. The old
        # whole-frame matcher could falsely recover the original placement.
        no_dial = template.copy()
        cv2.ellipse(no_dial, (180, 180), (154, 154), 0, 0, 360, (245, 245, 245), -1)
        matrix, status = match_template(template, no_dial)
        self.assertIsNone(matrix)
        self.assertEqual(status['status'], 'unknown')

if __name__ == '__main__':
    unittest.main()
