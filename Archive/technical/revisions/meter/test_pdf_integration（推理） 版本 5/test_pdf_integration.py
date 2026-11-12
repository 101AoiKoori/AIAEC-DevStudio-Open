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
sys.path.insert(0, str(ROOT / 'ws/src/aec_vision/src'))
from aec_vision.meter import annotate_ticks, project_points


@unittest.skipUnless(SAMPLE.exists(), 'Extract PDF sample first')
class PdfIntegrationTests(unittest.TestCase):
    def test_estimated_ticks_are_dashed_and_unknown_not_drawn(self):
        ticks = {'temperature': [{'index': i, 'start': [5, y], 'end': [45, y]}
                                 for i, y in enumerate((10, 25, 40))], 'humidity': []}
        visibility = {'temperature': {'visible_indices': [0], 'estimated_indices': [1]},
                      'humidity': {'visible_indices': [], 'estimated_indices': []}}
        output = annotate_ticks(np.zeros((60, 60, 3), np.uint8), ticks, np.eye(3), visibility)
        self.assertGreater(output[10, 25, 1], 200)
        self.assertGreater(output[25, 10, 1], 100)
        self.assertEqual(int(output[25, 22, 1]), 0)
        self.assertEqual(int(output[40, :, 1].max()), 0)

    def test_all_tick_segments_are_projected_and_drawn(self):
        ticks = json.loads(TICKS.read_text(encoding='utf-8-sig'))
        self.assertEqual([len(ticks[k]) for k in ('temperature', 'humidity')], [81, 51])
        matrix = np.float32([[.75, -.08, 100], [.08, .75, 25], [.0001, -.00005, 1]])
        rendered = annotate_ticks(np.zeros((700, 700, 3), np.uint8), ticks, matrix)
        for group in ('temperature', 'humidity'):
            for tick in ticks[group]:
                ends = project_points([tick['start'], tick['end']], matrix)
                point = np.rint(ends.mean(axis=0)).astype(int)
                self.assertGreater(int(rendered[point[1], point[0], 1]), 200,
                                   '{} tick {}'.format(group, tick['index']))

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
                self.assertEqual(data['ticks']['temperature']['estimated_indices'], [50])
                self.assertEqual(data['ticks']['humidity']['estimated_indices'], [33])
                self.assertEqual(data['ticks']['temperature']['unknown_count'], 0)
                self.assertEqual(data['ticks']['humidity']['unknown_count'], 0)
                self.assertLess(np.linalg.norm(np.array(data['needles']['temperature']['center']) - expected_center), 3)
                green = (output[:, :, 1] > 235) & (output[:, :, 0] < 50) & (output[:, :, 2] < 50)
                self.assertGreater(int(green.sum()), 500)
            blank = np.full((700, 700, 3), 255, np.uint8)
            data, output = self.run_cli(blank, tmp)
            self.assertEqual(data['match']['status'], 'unknown')
            self.assertEqual(data['needles']['temperature']['status'], 'unknown')
            np.testing.assert_array_equal(output, blank)

    def test_local_occlusion_excludes_affected_tick_segments(self):
        original = cv2.imread(str(SAMPLE))
        cases = [((230, 440, 275, 510), 'humidity', 40),
                 ((480, 240, 550, 315), 'temperature', 65)]
        with tempfile.TemporaryDirectory() as tmp:
            for (x0, y0, x1, y1), group, expected_index in cases:
                obscured = original.copy()
                cv2.rectangle(obscured, (x0, y0), (x1, y1), (200, 200, 200), -1)
                data, _ = self.run_cli(obscured, tmp)
                self.assertEqual(data['match']['status'], 'matched')
                self.assertIn(expected_index, data['ticks'][group]['unknown_indices'])
                self.assertNotIn(expected_index, data['ticks'][group]['visible_indices'])

    def test_dark_image_does_not_guess_humidity_direction(self):
        original = cv2.imread(str(SAMPLE))
        dark = cv2.convertScaleAbs(original, alpha=.5, beta=0)
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = self.run_cli(dark, tmp)
            self.assertEqual(data['match']['status'], 'matched')
            self.assertEqual(data['needles']['humidity']['status'], 'unknown')
            self.assertIsNone(data['needles']['humidity']['angle_deg'])

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
