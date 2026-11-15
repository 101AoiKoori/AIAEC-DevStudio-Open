"""Reject placements that leave reviewed ground or overlap recorded obstacles."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import compose_v4_grounded_train as ground  # noqa: E402


class SafePlacementTest(unittest.TestCase):
    def setUp(self):
        self.background = {'safe_regions': [[[10, 10], [200, 10], [200, 200], [10, 200]]],
                           'forbidden_boxes': [[100, 100, 140, 160]]}
        self.actor = {'sprite': np.zeros((30, 40, 4), dtype=np.uint8), 'foot_x': 60, 'foot_y': 80}

    def test_complete_canvas_inside_and_clear(self):
        ground.validate_safe_placement(self.actor, self.background)

    def test_rejects_partial_region_exit(self):
        self.actor['foot_x'] = 15
        with self.assertRaisesRegex(ValueError, 'outside reviewed safe region'):
            ground.validate_safe_placement(self.actor, self.background)

    def test_rejects_obstacle_intersection(self):
        self.actor.update({'foot_x': 110, 'foot_y': 130})
        with self.assertRaisesRegex(ValueError, 'overlaps a recorded obstacle'):
            ground.validate_safe_placement(self.actor, self.background)


if __name__ == '__main__':
    unittest.main()
