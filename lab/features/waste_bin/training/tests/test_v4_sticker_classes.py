import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('sticker_v4', Path(__file__).resolve().parents[1] / 'scripts/train_waste_sticker_v4.py')
sticker = importlib.util.module_from_spec(spec); spec.loader.exec_module(sticker)


class StickerClassReportTests(unittest.TestCase):
    def test_macro_and_multi_image_preserve_missing_classes(self):
        rows = [{'ground_truth': [{'class_id': 0}, {'class_id': 1}],
                 'predictions': [{'class_id': 0}, {'class_id': 1}, {'class_id': 1}],
                 'matches': [[0, 0, .9], [1, 1, .9]], 'false_prediction_indices': [2]}]
        result = sticker.sticker_class_report(rows, [str(i) for i in range(24)])
        self.assertEqual(result['all_stickers_matched_images'], 1)
        self.assertEqual(result['all_stickers_matched_without_extra_predictions_images'], 0)
        self.assertEqual(result['classes'][1]['false_predictions'], 1)
        self.assertEqual(len(result['absent_class_ids']), 22)
        self.assertIsNone(result['classes'][2]['recall'])

    def test_bad_matching_fails(self):
        row = {'ground_truth': [{'class_id': 0}], 'predictions': [{'class_id': 1}],
               'matches': [[0, 0, .9]], 'false_prediction_indices': []}
        with self.assertRaises(ValueError): sticker.sticker_class_report([row], [str(i) for i in range(24)])


if __name__ == '__main__':
    unittest.main()
