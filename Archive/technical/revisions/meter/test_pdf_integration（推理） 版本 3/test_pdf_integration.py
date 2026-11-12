import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/meter/inference/scripts/read_meter.py'
CONFIG = ROOT / 'lab/features/meter/inference/config/pdf_page15.json'
TICKS = ROOT / 'lab/features/meter/inference/config/pdf_ticks.json'
SAMPLE = ROOT / 'datasets/raw/meter/pdf/page15-005.jpg'


@unittest.skipUnless(SAMPLE.exists(), 'Extract PDF sample first')
class PdfIntegrationTests(unittest.TestCase):
    def test_temperature_calibration_is_reproducible(self):
        script = ROOT / 'lab/features/meter/inference/scripts/calibrate_pdf_ticks.py'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'ticks.json'
            subprocess.run([sys.executable, str(script), '--image', str(SAMPLE),
                            '--ticks', str(TICKS), '--output', str(output)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(TICKS.read_text(encoding='utf-8-sig')),
                             json.loads(output.read_text(encoding='utf-8')))

    def run_cli(self, image, tmp):
        source = Path(tmp) / 'input.png'
        output = Path(tmp) / 'output.png'
        self.assertTrue(cv2.imwrite(str(source), image))
        run = subprocess.run([sys.executable, str(SCRIPT), str(source), '--template',
                              str(SAMPLE), '--config', str(CONFIG), '--ticks', str(TICKS),
                              '--output', str(output)], capture_output=True, text=True, check=True)
        return json.loads(run.stdout), cv2.imread(str(output))

    def test_original_transformed_and_failure(self):
        original = cv2.imread(str(SAMPLE))
        transform = np.float32([[.75, 0, 55], [0, .75, 65], [0, 0, 1]])
        moved = cv2.warpPerspective(original, transform, (700, 700), borderValue=(255, 255, 255))
        with tempfile.TemporaryDirectory() as tmp:
            for image, expected_center in ((original, (320, 355)), (moved, (295, 331.25))):
                data, output = self.run_cli(image, tmp)
                self.assertEqual(data['match']['status'], 'matched')
                self.assertEqual(data['needles']['temperature']['status'], 'detected')
                self.assertEqual(data['needles']['humidity']['status'], 'detected')
                self.assertLess(np.linalg.norm(np.array(data['needles']['temperature']['center']) - expected_center), 3)
                green = (output[:, :, 1] > 235) & (output[:, :, 0] < 50) & (output[:, :, 2] < 50)
                self.assertGreater(int(green.sum()), 500)
            blank = np.full((700, 700, 3), 255, np.uint8)
            data, output = self.run_cli(blank, tmp)
            self.assertEqual(data['match']['status'], 'unknown')
            self.assertEqual(data['needles']['temperature']['status'], 'unknown')
            np.testing.assert_array_equal(output, blank)

    def test_occluded_dial_and_changed_needle(self):
        original = cv2.imread(str(SAMPLE))
        hidden = original.copy()
        cv2.rectangle(hidden, (170, 405), (460, 625), (200, 200, 200), -1)
        mask = np.zeros(original.shape[:2], np.uint8)
        cv2.line(mask, (320, 355), (426, 154), 255, 15)
        changed = cv2.inpaint(original, mask, 5, cv2.INPAINT_TELEA)
        cv2.line(changed, (320, 355), (195, 185), (20, 20, 20), 5)
        with tempfile.TemporaryDirectory() as tmp:
            data, output = self.run_cli(hidden, tmp)
            self.assertEqual(data['match']['status'], 'unknown')
            self.assertEqual(data['needles']['humidity']['status'], 'unknown')
            np.testing.assert_array_equal(output, hidden)
            data, _ = self.run_cli(changed, tmp)
            self.assertEqual(data['match']['status'], 'matched')


if __name__ == '__main__':
    unittest.main()
