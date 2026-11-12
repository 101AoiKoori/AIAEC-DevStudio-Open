import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from generate_meter_from_renders import LIGHTING, sample_gain


class LightSamplingTest(unittest.TestCase):
    def test_bounded_levels_are_distinct_and_reproducible(self):
        first, second = np.random.default_rng(420), np.random.default_rng(420)
        for level in ('dim', 'normal', 'bright'):
            samples = [sample_gain(first, level) for _ in range(100)]
            other = [sample_gain(second, level) for _ in range(100)]
            self.assertEqual(samples, other)
            _, _, low, high = LIGHTING[level]
            self.assertTrue(all(low <= x <= high for x in samples))
        self.assertLess(LIGHTING['dim'][3], LIGHTING['normal'][2])
        self.assertLess(LIGHTING['normal'][3], LIGHTING['bright'][2])


if __name__ == '__main__':
    unittest.main()
