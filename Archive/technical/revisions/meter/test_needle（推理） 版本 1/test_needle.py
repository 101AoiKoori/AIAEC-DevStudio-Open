import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws' / 'src' / 'aec_vision' / 'src'))
from aec_vision.meter import detect_needle


class NeedleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'center': [100, 100], 'search_radius': [15, 60],
                    'angle_range_deg': [-90, 90], 'dark_threshold': 80,
                    'min_support': .6, 'max_tip_radius': 70}

    def test_synthetic_needle(self):
        image = np.full((200, 200, 3), 255, np.uint8)
        cv2.line(image, (100, 100), (160, 65), (0, 0, 0), 5)
        result = detect_needle(image, self.cfg)
        expected = math.degrees(math.atan2(-35, 60))
        self.assertEqual(result['status'], 'detected')
        self.assertLess(abs(result['angle_deg'] - expected), 3)

    def test_uniform_frames_unknown(self):
        for pixel in (0, 255):
            image = np.full((200, 200, 3), pixel, np.uint8)
            self.assertEqual(detect_needle(image, self.cfg)['status'], 'unknown')


if __name__ == '__main__':
    unittest.main()
