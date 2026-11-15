import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('bin_v4', Path(__file__).resolve().parents[1] / 'scripts/train_waste_bin_state_v4.py')
bin_state = importlib.util.module_from_spec(spec); spec.loader.exec_module(bin_state)


class BinStateReportTests(unittest.TestCase):
    def test_opposite_state_by_reviewed_color(self):
        box = [0, 0, 10, 10]
        rows = [{'source_image': 'x', 'ground_truth': [{'class_id': 0, 'xyxy': box}],
                 'predictions': [{'class_id': 1, 'xyxy': box}, {'class_id': 1, 'xyxy': box}],
                 'matches': [], 'missed_gt_indices': [0], 'false_prediction_indices': [0, 1]}]
        review = [{'source_image': 'x', 'gt_index': 0, 'color': 'red', 'review_status': 'confirmed', 'review_evidence': 'a'}]
        result = bin_state.reviewed_color_states(rows, review, .5)
        self.assertEqual(result['colors']['red']['open']['missed'], 1)
        self.assertEqual(result['colors']['red']['open']['opposite_state'], 1)
        self.assertIsNone(result['colors']['gray']['closed']['recall'])

    def test_no_review_fails(self):
        with self.assertRaises(ValueError): bin_state.reviewed_color_states([], [], .5)


if __name__ == '__main__':
    unittest.main()
