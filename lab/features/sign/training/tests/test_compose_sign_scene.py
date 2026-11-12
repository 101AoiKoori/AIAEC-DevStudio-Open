import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compose_sign_scene.py'
spec = importlib.util.spec_from_file_location('compose_sign_scene', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestSignComposition(unittest.TestCase):
    def test_yolo_full_frame_and_invalid_boxes(self):
        self.assertEqual(module.yolo([0, 0, 960, 480]), '0 0.500000 0.500000 1.000000 1.000000')
        for box in ([-1, 0, 1, 1], [1, 1, 1, 2], [0, 0, 961, 480]):
            with self.assertRaises(ValueError):
                module.yolo(box)

    def test_negative_has_no_box(self):
        background = np.full((480, 960, 3), 64, dtype=np.uint8)
        result, box = module.make_image(background, None, random.Random(1), 'train')
        self.assertIsNone(box)
        np.testing.assert_array_equal(result, background)

    def test_positive_box_fits_visible_pixels(self):
        background = np.zeros((480, 960, 3), dtype=np.uint8)
        sprite = np.full((100, 30, 4), 255, dtype=np.uint8)
        for split in ('train', 'val', 'test'):
            result, box = module.make_image(background, sprite, random.Random(42), split)
            self.assertTrue(np.any(result))
            module.yolo(box)
            if split == 'val':
                self.assertEqual(box[1], 0)


if __name__ == '__main__':
    unittest.main()
