import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'ws/src/aec_vision/src'))
from aec_vision.meter_auto import analyze_meter

SAMPLE = ROOT / 'datasets/raw/meter/pdf/page15-005.jpg'
RANGES = {'temperature': {'minimum': -30, 'maximum': 50, 'direction': 'clockwise'},
          'humidity': {'minimum': 0, 'maximum': 100, 'direction': 'clockwise'}}


class AutoMeterTests(unittest.TestCase):
    def test_blank_returns_unknown(self):
        result = analyze_meter(np.full((672, 641, 3), 255, dtype=np.uint8), RANGES)
        self.assertEqual(result['status'], 'unknown')
        for dial in result['dials'].values():
            self.assertEqual(dial['reading']['status'], 'unknown')
            self.assertIsNone(dial['reading']['value'])

    @unittest.skipUnless(SAMPLE.exists(), 'Extract PDF sample first')
    def test_pdf_geometry_and_explicit_direction(self):
        image = cv2.imread(str(SAMPLE))
        for scale in (1.0, .75):
            sample = cv2.resize(image, None, fx=scale, fy=scale)
            result = analyze_meter(sample, RANGES)
            for name, expected_center, expected_angle in (
                    ('temperature', (320, 355), -62), ('humidity', (325, 497), 150)):
                dial = result['dials'][name]
                self.assertLess(np.linalg.norm(np.array(dial['center']) / scale - expected_center), 8)
                self.assertEqual(dial['needle']['status'], 'detected')
                self.assertLess(abs(dial['needle']['angle_deg'] - expected_angle), 4)
                self.assertEqual(dial['arc']['status'], 'candidate')
                self.assertEqual(dial['reading']['status'], 'estimated' if name == 'temperature' else 'unknown')
            self.assertEqual(result['status'], 'partial')
        unknown = analyze_meter(image, {'temperature': {'minimum': -30, 'maximum': 50},
                                        'humidity': {'minimum': 0, 'maximum': 100}})
        self.assertTrue(all(dial['reading']['value'] is None for dial in unknown['dials'].values()))

    @unittest.skipUnless(SAMPLE.exists(), 'Extract PDF sample first')
    def test_dark_image_does_not_fabricate_values(self):
        image = cv2.imread(str(SAMPLE))
        dark = cv2.convertScaleAbs(image, alpha=.5)
        result = analyze_meter(dark, RANGES)
        for dial in result['dials'].values():
            self.assertIsNone(dial['reading']['value'])


if __name__ == '__main__':
    unittest.main()
