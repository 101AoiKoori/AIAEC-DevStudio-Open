import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / 'lab/features/e_bike/training/scripts/preview_person_occlusions.py'
spec = importlib.util.spec_from_file_location('preview_person_occlusions', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OcclusionTest(unittest.TestCase):
    def test_holes_leave_person_visible_and_box_uses_visible_pixels(self):
        canvas = np.zeros((module.HEIGHT, module.WIDTH, 3), np.uint8)
        person = np.full((4, 4, 4), 255, np.uint8)
        person_mask = module.mask_and_place(canvas, person, 10, 10)
        bike = np.full((4, 4, 4), 255, np.uint8)
        bike[:, :, :3] = 0
        bike[:, :, 3] = 0
        bike[:, :2, 3] = 255
        bike[1, 0, 3] = 0
        bike_mask = module.mask_and_place(canvas, bike, 10, 10)
        visible = np.logical_and(person_mask, np.logical_not(bike_mask))
        self.assertEqual(int(visible.sum()), 9)
        self.assertTrue(visible[11, 10])
        self.assertFalse(visible[10, 10])
        self.assertEqual(module.box(visible), [10, 10, 14, 14])

    def test_rejects_out_of_bounds_placement(self):
        canvas = np.zeros((module.HEIGHT, module.WIDTH, 3), np.uint8)
        with self.assertRaises(ValueError):
            module.mask_and_place(canvas, np.ones((4, 4, 4), np.uint8), 1023, 767)


if __name__ == '__main__':
    unittest.main()
