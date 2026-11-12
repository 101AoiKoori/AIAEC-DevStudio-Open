import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SCRIPTS = ROOT / 'lab/features/e_bike/training/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('refine_selected_bike_masks', SCRIPTS / 'refine_selected_bike_masks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RefineSelectedTest(unittest.TestCase):
    def test_alpha_changes_only_in_selected_roi_and_never_fills_holes(self):
        rgba = np.full((2048, 1152, 4), 255, np.uint8)
        rgba[:, :, :3] = (128, 128, 128)
        rgba[1080, 450, :3] = (0, 220, 220)
        rgba[1080, 451, :3] = (20, 20, 20)
        rgba[980, 370, :3] = (0, 220, 220)
        before = rgba.copy()
        for index, bounds in [(0, (425, 485, 1073, 1105)), (2, (330, 460, 960, 1015))]:
            alpha, _ = module.local_background_cleanup(rgba, index)
            left, right, top, bottom = bounds
            changed = alpha != rgba[:, :, 3]
            self.assertTrue(changed.any())
            ys, xs = np.nonzero(changed)
            self.assertGreaterEqual(xs.min(), left)
            self.assertLess(xs.max(), right)
            self.assertGreaterEqual(ys.min(), top)
            self.assertLess(ys.max(), bottom)
            self.assertFalse(((rgba[:, :, 3] == 0) & (alpha > 0)).any())
        self.assertEqual(module.local_background_cleanup(rgba, 0)[0][1080, 450], 255)
        self.assertEqual(module.local_background_cleanup(rgba, 0)[0][1080, 451], 255)
        self.assertTrue(np.array_equal(module.local_background_cleanup(rgba, 1)[0], rgba[:, :, 3]))
        self.assertTrue(np.array_equal(rgba, before))

    def test_rejects_wrong_views_and_wrong_dimensions(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            names = ['sam2_01_fallen.png', 'sam2_03_fallen.png', 'sam2_04_upright.png', 'sam2_06_upright.png']
            for name in names:
                cv2.imencode('.png', np.full((12, 12, 4), 255, np.uint8))[1].tofile(str(root / name))
            selection = root / 'selection.json'
            plan = {'vehicles': [{'path': str(root / name)} for name in names]}
            plan['vehicles'][0]['path'] = str(root / 'wrong.png')
            selection.write_text(json.dumps(plan), encoding='utf-8')
            with self.assertRaises(ValueError):
                module.refine(selection, root / 'wrong_views')
            self.assertFalse((root / 'wrong_views').exists())
            plan['vehicles'][0]['path'] = str(root / names[0])
            selection.write_text(json.dumps(plan), encoding='utf-8')
            out = root / 'wrong_dimensions'
            with self.assertRaises(ValueError):
                module.refine(selection, out)
            manifest = json.loads((out / 'refinement_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'failed')
            self.assertIn('2160×3840', manifest['error'])


if __name__ == '__main__':
    unittest.main()
