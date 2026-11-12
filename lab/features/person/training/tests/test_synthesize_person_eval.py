import sys
import random
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import synthesize_person_eval as evaluator


class PersonEvaluationTest(unittest.TestCase):
    def test_mask_visibility_and_requested_occlusion(self):
        rgba = np.full((150, 45, 4), 255, dtype=np.uint8)
        person = ({'sprite': 'candidate_person'}, rgba)
        bike = ({'path': 'candidate_bike'}, rgba)
        people = {'community': person, 'noncommunity': person}
        bikes = {'upright': [bike], 'fallen': [bike]}
        background = np.full((480, 960, 3), 120, dtype=np.uint8)
        rng = random.Random(29)
        for _ in range(100):
            try:
                _, full, visible, objects, plan, _ = evaluator.make_sample(2, rng, people, bikes, background)
                break
            except ValueError:
                continue
        else:
            self.fail('unable to produce an occlusion sample')
        self.assertEqual(plan['kind'], 'occlusion')
        self.assertEqual(len(objects), 2)
        self.assertTrue(np.any(full[0] & full[1]))
        for f, v, obj in zip(full, visible, objects):
            self.assertTrue(np.all(v <= f))
            self.assertGreaterEqual(obj['visible_fraction'], .55)

    def test_source_must_be_inside_workspace(self):
        with self.assertRaises(ValueError):
            evaluator.workspace_file('../missing_source.png')


if __name__ == '__main__':
    unittest.main()
