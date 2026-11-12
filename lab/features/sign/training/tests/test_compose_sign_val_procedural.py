import importlib.util
import random
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('compose_sign_val_procedural', SCRIPTS / 'compose_sign_val_procedural.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestProceduralSign(unittest.TestCase):
    def test_fixture_has_face_pole_and_base(self):
        sprite = module.draw_fixture(random.Random(41))
        self.assertEqual(sprite.shape[2], 4)
        alpha = sprite[:, :, 3] > 127
        self.assertTrue(np.any(alpha[50:150]))
        self.assertTrue(np.any(alpha[250:350]))
        self.assertTrue(np.any(alpha[450:490]))

    def test_output_rejects_wrong_location(self):
        with self.assertRaises(ValueError):
            module.generate(Path('tmp/outside-sign-exports'), 10, 1)


if __name__ == '__main__':
    unittest.main()
