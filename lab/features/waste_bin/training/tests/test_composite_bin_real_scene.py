import importlib.util
import unittest
from collections import Counter
from pathlib import Path
import random

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/composite_bin_real_scene.py'
spec = importlib.util.spec_from_file_location('composite_bin_real_scene', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
art_spec = importlib.util.spec_from_file_location('draw_independent_sticker_eval',
                                                   SCRIPT.with_name('draw_independent_sticker_eval.py'))
art = importlib.util.module_from_spec(art_spec)
art_spec.loader.exec_module(art)


class BinCompositeTests(unittest.TestCase):
    def test_sticker_counts_match_requested_strata(self):
        self.assertEqual(Counter(module.sticker_count(i) for i in range(20)),
                         {0: 3, 1: 5, 2: 3, 3: 4, 4: 5})

    def test_paper_and_icon_stay_in_bin_mask(self):
        image = np.zeros((120, 100, 3), dtype=np.uint8)
        mask = np.zeros((120, 100), dtype=bool)
        mask[10:110, 10:90] = True
        icon = (np.zeros((20, 20, 3), dtype=np.uint8), np.ones((20, 20), dtype=np.float32))
        box = module.paper_on_bin(image, mask, icon, random.Random(2))
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        self.assertTrue(mask[y1:y2, x1:x2].all())
        self.assertTrue(np.any(image[y1:y2, x1:x2]))
        self.assertFalse(np.any(image[~mask]))

    def test_paper_rejects_inadequate_mask(self):
        image = np.zeros((120, 100, 3), dtype=np.uint8)
        mask = np.zeros((120, 100), dtype=bool)
        mask[10:110, 10:90:2] = True
        icon = (np.zeros((20, 20, 3), dtype=np.uint8), np.ones((20, 20), dtype=np.float32))
        self.assertIsNone(module.paper_on_bin(image, mask, icon, random.Random(2)))
        self.assertFalse(np.any(image))

    def test_cached_icon_keeps_opencv_bgr_order(self):
        names, icons, _, _ = module.load_icons()
        with np.load(module.CACHE / '苹果.npz') as stored:
            self.assertTrue(np.array_equal(icons[names.index('苹果')][0], stored['rgb']))

    def test_independent_vector_artwork_uses_distinct_designs(self):
        self.assertEqual(len(art.CANDIDATE_CLASSES), 12)
        for cls in art.CANDIDATE_CLASSES:
            val = np.asarray(art.draw_icon(cls, 'val'))
            test = np.asarray(art.draw_test_independent(cls))
            self.assertFalse(np.array_equal(val, test), cls)
            self.assertTrue(np.any(val != 255), cls)
            self.assertTrue(np.any(test != 255), cls)


if __name__ == '__main__':
    unittest.main()
