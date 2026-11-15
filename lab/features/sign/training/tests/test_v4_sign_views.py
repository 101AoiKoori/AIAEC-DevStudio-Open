import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('sign_v4', Path(__file__).resolve().parents[1] / 'scripts/train_sign_v4.py')
sign = importlib.util.module_from_spec(spec); spec.loader.exec_module(sign)


class SignViewsTests(unittest.TestCase):
    def test_back_and_unknown_group_denominators(self):
        rows = [{'source_image': 'x.png', 'ground_truth': [{'class_id': 0}, {'class_id': 0}], 'matches': [[0, 0, .8]]}]
        reviews = [{'source_image': 'x.png', 'gt_index': 0, 'view': 'back', 'review_status': 'confirmed', 'review_evidence': 'a'},
                   {'source_image': 'x.png', 'gt_index': 1, 'view': 'unknown', 'review_status': 'confirmed', 'review_evidence': 'a'}]
        result = sign.reviewed_view_report(rows, reviews)
        self.assertEqual(result['views']['back']['recall'], 1)
        self.assertEqual(result['views']['unknown']['recall'], 0)
        self.assertIsNone(result['views']['front']['recall'])

    def test_unreviewed_and_duplicate_fails(self):
        rows = [{'source_image': 'x', 'ground_truth': [{'class_id': 0}], 'matches': []}]
        item = {'source_image': 'x', 'gt_index': 0, 'view': 'back', 'review_status': 'confirmed', 'review_evidence': 'a'}
        with self.assertRaises(ValueError): sign.reviewed_view_report(rows, [item, item])
        with self.assertRaises(ValueError): sign.reviewed_view_report(rows, [])


if __name__ == '__main__':
    unittest.main()
