"""Verify upright/fallen pairing without treating duplicate boxes as vehicles."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_e_bike_v4.py'
SPEC = importlib.util.spec_from_file_location('bike_v4', SCRIPT)
bike = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bike)


class BikeStateTests(unittest.TestCase):
    def test_opposite_state_pairs_are_one_to_one(self):
        box = [0, 0, 10, 10]
        rows = [{'ground_truth': [{'class_id': 0, 'xyxy': box}, {'class_id': 1, 'xyxy': [20, 20, 30, 30]}],
                 'predictions': [{'class_id': 1, 'xyxy': box}, {'class_id': 1, 'xyxy': box},
                                 {'class_id': 1, 'xyxy': [20, 20, 30, 30]}],
                 'matches': [[1, 2, 1.0]], 'missed_gt_indices': [0], 'false_prediction_indices': [0, 1]}]
        result = bike.state_confusions(rows, 0.5)
        self.assertEqual(result['opposite_state_pairs'], 1)
        self.assertEqual(result['states']['upright']['missed'], 1)
        self.assertEqual(result['states']['upright']['opposite_state_prediction'], 1)
        self.assertEqual(result['states']['fallen']['matched_same_state'], 1)
        self.assertEqual(result['states']['fallen']['opposite_state_prediction'], 0)

    def test_invalid_state_and_iou_fail(self):
        with self.assertRaises(ValueError):
            bike.state_confusions([], 0)
        with self.assertRaises(ValueError):
            bike.state_confusions([{'ground_truth': [{'class_id': 2}], 'predictions': []}], 0.5)


if __name__ == '__main__':
    unittest.main()
