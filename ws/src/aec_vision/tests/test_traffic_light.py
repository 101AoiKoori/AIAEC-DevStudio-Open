import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aec_vision.traffic_light import TrafficLightState, detect_traffic_light


class TrafficLightTest(unittest.TestCase):
    def setUp(self):
        self.regions = {'red': (0, 0, 1, .3), 'yellow': (0, .35, 1, .65),
                        'green': (0, .7, 1, 1)}
        self.ranges = {'red': [((0, 180, 180), (10, 255, 255)),
                               ((170, 180, 180), (179, 255, 255))],
                       'yellow': [((20, 180, 180), (35, 255, 255))],
                       'green': [((45, 180, 180), (85, 255, 255))]}
        self.areas = dict.fromkeys(self.regions, 16)

    def detect(self, image):
        return detect_traffic_light(image, self.regions, self.ranges, self.areas)

    def test_three_colors_and_positions_along_roi(self):
        for y, bgr, color in [(10, (0, 0, 255), 'red'), (45, (0, 255, 255), 'yellow'),
                              (80, (0, 255, 0), 'green')]:
            for x in (1, 45, 90):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[y:y + 5, x:x + 5] = bgr
                self.assertEqual(self.detect(image)['color'], color)
                self.assertEqual(self.detect(image)['status'], 'candidate')

    def test_missing_wrong_region_small_noise_or_conflict_are_unknown(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(self.detect(image)['status'], 'unknown')
        image[10:15, 10:15] = (0, 255, 0)
        self.assertEqual(self.detect(image)['status'], 'unknown')
        image[:] = 0
        image[80:83, 10:13] = (0, 255, 0)
        self.assertEqual(self.detect(image)['status'], 'unknown')
        image[80:85, 30:35] = (0, 255, 0)
        image[10:15, 30:35] = (0, 0, 255)
        self.assertEqual(self.detect(image)['status'], 'unknown')

    def test_hue_wrap_is_supported(self):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[10:15, 10:15] = (175, 255, 255)
        self.assertEqual(self.detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))['color'], 'red')

    def test_vertical_strips_cover_different_heights(self):
        self.regions = {'red': (0, 0, .3, 1), 'yellow': (.35, 0, .65, 1),
                        'green': (.7, 0, 1, 1)}
        for x, bgr, color in [(10, (0, 0, 255), 'red'), (45, (0, 255, 255), 'yellow'),
                              (80, (0, 255, 0), 'green')]:
            for y in (1, 45, 90):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[y:y + 5, x:x + 5] = bgr
                self.assertEqual(self.detect(image)['color'], color)

    def test_invalid_frame_and_configuration(self):
        self.assertEqual(self.detect(None)['status'], 'unknown')
        self.assertEqual(self.detect(np.zeros((10, 10), dtype=np.uint8))['status'], 'unknown')
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for invalid in ({}, {'red': (0, 0, 1, 0), 'yellow': (0, 0, 1, 1), 'green': (0, 0, 1, 1)}):
            with self.assertRaises(ValueError):
                detect_traffic_light(image, invalid, self.ranges, self.areas)
        bad_ranges = dict(self.ranges, red=[((170, 0, 0), (10, 255, 255))])
        with self.assertRaises(ValueError):
            detect_traffic_light(image, self.regions, bad_ranges, self.areas)

    def test_stability_change_unknown_and_expiry_revoke_green(self):
        state = TrafficLightState(required_frames=2, max_age=.5)
        green = {'status': 'candidate', 'color': 'green'}
        red = {'status': 'candidate', 'color': 'red'}
        self.assertEqual(state.update(green, 1., 1., True)['status'], 'unknown')
        self.assertEqual(state.update(green, 1.1, 1.1, True)['color'], 'green')
        self.assertEqual(state.update(red, 1.2, 1.2, True)['status'], 'unknown')
        self.assertEqual(state.update(red, 1.3, 1.3, True)['color'], 'red')
        self.assertEqual(state.update({'status': 'unknown'}, 1.4, 1.4, True)['status'], 'unknown')
        state.update(green, 2., 2., True)
        self.assertEqual(state.update(green, 2.1, 2.1, True)['color'], 'green')
        self.assertEqual(state.read(2.7, True)['status'], 'unknown')
        self.assertEqual(state.read(2.2, True)['status'], 'unknown')

    def test_untrusted_position_stale_and_repeated_frame(self):
        state = TrafficLightState(required_frames=1, max_age=.5)
        green = {'status': 'candidate', 'color': 'green'}
        self.assertEqual(state.update(green, 1., 1., False)['status'], 'unknown')
        self.assertEqual(state.update(green, 1., 1.6, True)['status'], 'unknown')
        self.assertEqual(state.update(green, 2., 2., True)['color'], 'green')
        self.assertEqual(state.update(green, 2., 2.1, True)['status'], 'unknown')
        self.assertEqual(state.update(green, 3., 2.9, True)['status'], 'unknown')
        with self.assertRaises(ValueError):
            TrafficLightState(required_frames=0, max_age=.5)


if __name__ == '__main__':
    unittest.main()
