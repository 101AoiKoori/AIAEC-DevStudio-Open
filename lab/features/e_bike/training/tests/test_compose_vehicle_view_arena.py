"""Check inferred contact alignment and low-camera ground projection."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location(
    'bike_arena_composite', ROOT / 'lab/features/e_bike/training/scripts/compose_vehicle_view_arena.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class ContactGeometryTest(unittest.TestCase):
    def test_similarity_maps_both_contact_points(self):
        source = np.array([[10, 30], [100, 35]], dtype=np.float32)
        target = np.array([[180, 340], [280, 347]], dtype=np.float32)
        matrix = subject.similarity(source, target)
        actual = np.column_stack((source, np.ones(2))) @ matrix.T
        np.testing.assert_allclose(actual, target, atol=1e-4)

    def test_ground_projection_shrinks_with_distance(self):
        camera = {'camera_location_m': [0, -1.5, .16],
                  'camera_target_m': [0, .15, .08], 'camera_lens_mm': 22}
        near = subject.project(camera, [-.1, -.6, -.02]), subject.project(camera, [.1, -.6, -.02])
        far = subject.project(camera, [-.1, .0, -.02]), subject.project(camera, [.1, .0, -.02])
        self.assertGreater(np.linalg.norm(near[1] - near[0]), np.linalg.norm(far[1] - far[0]))


if __name__ == '__main__':
    unittest.main()
