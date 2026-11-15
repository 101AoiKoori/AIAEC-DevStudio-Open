"""Check named point summaries and unknown denominators."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_meter_v4.py'
SPEC = importlib.util.spec_from_file_location('meter_v4', SCRIPT)
meter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(meter)


class MeterPointTests(unittest.TestCase):
    def test_point_names_and_missing_values(self):
        result = meter.keypoint_errors([{'keypoint_index': '1', 'visibility': '2', 'error_pixels': '4'},
                                        {'keypoint_index': '1', 'visibility': '1', 'error_pixels': '6'}])
        self.assertEqual(result[1]['name'], 'temperature_tip')
        self.assertEqual(result[1]['mean_error_pixels'], 5)
        self.assertEqual(result[1]['visibility_1'], 1)
        self.assertIsNone(result[0]['mean_error_pixels'])

    def test_invalid_point_is_rejected(self):
        with self.assertRaises(ValueError):
            meter.keypoint_errors([{'keypoint_index': '8', 'visibility': '2', 'error_pixels': '1'}])
        with self.assertRaises(ValueError):
            meter.keypoint_errors([{'keypoint_index': '0', 'visibility': '0', 'error_pixels': '1'}])


if __name__ == '__main__':
    unittest.main()
