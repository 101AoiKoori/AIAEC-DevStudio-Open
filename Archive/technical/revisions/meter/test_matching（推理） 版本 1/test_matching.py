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

if __name__ == '__main__':
    unittest.main()
