import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('paper_v4', Path(__file__).resolve().parents[1] / 'scripts/train_building_name_paper_v4.py')
paper = importlib.util.module_from_spec(spec); spec.loader.exec_module(paper)


class PaperSizesTests(unittest.TestCase):
    def test_input_pixels_and_negative_image(self):
        rows = [{'shape_hw': [720, 1280], 'ground_truth': [{'class_id': 0, 'xyxy': [0, 0, 100, 20]}],
                 'matches': [[0, 0, .7]], 'false_prediction_indices': []},
                {'shape_hw': [480, 640], 'ground_truth': [], 'matches': [], 'false_prediction_indices': [0]}]
        report = paper.paper_size_report(rows, 640)
        self.assertEqual(report['short_side_pixels']['8_16']['matched'], 1)
        self.assertIsNone(report['short_side_pixels']['0_8']['recall'])
        self.assertEqual(report['empty_label_images_with_predictions'], 1)

    def test_invalid_class_fails(self):
        with self.assertRaises(ValueError):
            paper.paper_size_report([{'shape_hw': [10, 10], 'ground_truth': [{'class_id': 1}], 'matches': []}], 640)


if __name__ == '__main__':
    unittest.main()
