import sys
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from synthesize_vehicle_scene import scenario


class VehicleScenarioTest(unittest.TestCase):
    def test_twenty_sample_cycle_has_requested_scenes(self):
        plans = [scenario(i) for i in range(20)]
        self.assertEqual(Counter(kind for kind, _ in plans),
                         {'upright_only': 8, 'fallen_present': 8, 'empty': 4})
        for kind, classes in plans:
            if kind == 'upright_only':
                self.assertTrue(classes and set(classes) == {'upright'})
            elif kind == 'fallen_present':
                self.assertIn('fallen', classes)
            else:
                self.assertEqual(classes, [])
        self.assertIn(7, [len(classes) for _, classes in plans])
        self.assertTrue(any(set(classes) == {'upright', 'fallen'} for _, classes in plans))


if __name__ == '__main__':
    unittest.main()
