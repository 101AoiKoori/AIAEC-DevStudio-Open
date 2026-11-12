import json
import sys
import unittest
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from aec_vision.waste_bin import WasteCategory, associate_sticker, classify_bin_color, judge_disposal


class WasteBinTest(unittest.TestCase):
    def setUp(self):
        self.items = json.loads((PACKAGE / 'config/waste_bin/items.json').read_text(encoding='utf-8'))['items']

    def test_four_body_colors_and_unknown(self):
        for bgr, expected in [((255, 0, 0), WasteCategory.RECYCLABLE),
                              ((0, 200, 0), WasteCategory.FOOD),
                              ((0, 0, 220), WasteCategory.HAZARDOUS),
                              ((140, 140, 140), WasteCategory.RESIDUAL),
                              ((255, 255, 255), WasteCategory.UNKNOWN)]:
            image = np.full((30, 30, 3), bgr, dtype=np.uint8)
            self.assertEqual(classify_bin_color(image)['category'], expected)
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        image[:, :15] = (255, 0, 0)
        image[:, 15:] = (0, 200, 0)
        self.assertEqual(classify_bin_color(image)['category'], WasteCategory.UNKNOWN)
        with self.assertRaises(ValueError):
            classify_bin_color(np.empty((0, 0, 3), dtype=np.uint8))

    def test_all_rules_and_untrusted_inputs(self):
        self.assertEqual(len(self.items), 24)
        for key, rule in self.items.items():
            result = judge_disposal(rule['category'], key, self.items, True, True)
            self.assertEqual(result['status'], 'correct')
            other = next(c.value for c in WasteCategory if c not in (WasteCategory.UNKNOWN, WasteCategory(rule['category'])))
            self.assertEqual(judge_disposal(other, key, self.items, True, True)['status'], 'incorrect')
        self.assertEqual(judge_disposal('food', '苹果', self.items, True, True)['name'], '果核')
        for args in [('food', '未知物', True, True), ('unknown', '苹果', True, True),
                     ('food', '苹果', False, True), ('food', '苹果', True, False)]:
            self.assertEqual(judge_disposal(args[0], args[1], self.items, args[2], args[3])['status'], 'unknown')

    def test_association_requires_single_matching_bin(self):
        self.assertEqual(associate_sticker((10, 10, 20, 20), [(0, 0, 30, 30), (40, 0, 70, 30)]), 0)
        self.assertEqual(associate_sticker(np.array([10, 10, 20, 20], dtype=np.float32), [np.array([0, 0, 30, 30])]), 0)
        self.assertIsNone(associate_sticker((10, 10, 20, 20), [(0, 0, 30, 30), (5, 5, 25, 25)]))
        self.assertIsNone(associate_sticker((30, 10, 40, 20), [(0, 0, 25, 25)]))
        with self.assertRaises(ValueError):
            associate_sticker((0, 0, 0, 1), [])


if __name__ == '__main__':
    unittest.main()
