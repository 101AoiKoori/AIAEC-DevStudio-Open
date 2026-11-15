"""Synthetic checks for the mask geometry module.

Masks are drawn with OpenCV into zero arrays; no file is read.  Expected values
are analytic (a ring's hole ratio, a strip flush against the left border) so the
tests stay independent of the implementation.
"""

import unittest

import cv2
import numpy as np

from datasets.scripts.common.review import masks
from datasets.scripts.common.review.schema import REASON_CODES

SHAPE = (120, 120)


def ring_mask(outer: int = 40, inner: int = 15, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    centre = (shape[1] // 2, shape[0] // 2)
    cv2.circle(mask, centre, outer, 255, -1)
    cv2.circle(mask, centre, inner, 0, -1)
    return mask


def disk_mask(radius: int = 40, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.circle(mask, (shape[1] // 2, shape[0] // 2), radius, 255, -1)
    return mask


def blocks_mask(origins, side: int = 10, shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for x, y in origins:
        mask[y:y + side, x:x + side] = 255
    return mask


def left_strip_mask(shape: tuple[int, int] = (80, 80), width: int = 6) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    mask[20:60, :width] = 255
    return mask


class HoleFractionTest(unittest.TestCase):
    def test_ring_hole_matches_the_drawn_ratio(self):
        mask = ring_mask(outer=40, inner=15)
        # hole area / foreground area = r_in**2 / (r_out**2 - r_in**2)
        expected = (15.0 ** 2) / (40.0 ** 2 - 15.0 ** 2)
        self.assertAlmostEqual(masks.hole_area_fraction(mask), expected, delta=0.02)

    def test_filled_disk_has_no_hole(self):
        self.assertAlmostEqual(masks.hole_area_fraction(disk_mask()), 0.0, places=6)

    def test_empty_mask_reports_zero(self):
        self.assertEqual(masks.hole_area_fraction(np.zeros((40, 40), np.uint8)), 0.0)

    def test_full_frame_has_no_enclosed_background(self):
        self.assertEqual(masks.hole_area_fraction(np.full((40, 40), 255, np.uint8)), 0.0)


class BorderFractionTest(unittest.TestCase):
    def test_strip_on_the_left_edge_is_border(self):
        mask = left_strip_mask()
        value = masks.border_fraction(mask, margin_px=6)
        self.assertGreater(value, 0.9)
        self.assertAlmostEqual(value, 1.0)

    def test_interior_shape_does_not_touch_the_border(self):
        self.assertAlmostEqual(masks.border_fraction(disk_mask()), 0.0, places=6)

    def test_zero_margin_and_empty_mask(self):
        mask = left_strip_mask()
        self.assertEqual(masks.border_fraction(mask, margin_px=0), 0.0)
        self.assertEqual(masks.border_fraction(np.zeros((40, 40), np.uint8)), 0.0)

    def test_margin_is_clamped_to_the_frame(self):
        mask = left_strip_mask()
        self.assertAlmostEqual(masks.border_fraction(mask, margin_px=500), 1.0)


class ComponentCountTest(unittest.TestCase):
    def test_two_separate_blocks(self):
        self.assertEqual(masks.component_count(blocks_mask([(10, 10), (40, 40)])), 2)

    def test_specks_below_sixteen_pixels_are_ignored(self):
        mask = blocks_mask([(10, 10)], side=10)
        mask[10:13, 70:73] = 255  # 3×3 == 9 pixels
        self.assertEqual(masks.component_count(mask), 1)
        self.assertLess(3 * 3, masks.MIN_COMPONENT_AREA)

    def test_empty_and_ring(self):
        self.assertEqual(masks.component_count(np.zeros((40, 40), np.uint8)), 0)
        self.assertEqual(masks.component_count(ring_mask()), 1)

    def test_eight_connectivity_joins_diagonals(self):
        mask = np.zeros((40, 40), np.uint8)
        mask[10:18, 10:18] = 255
        mask[18:26, 18:26] = 255  # touches the first block on a diagonal
        self.assertEqual(masks.component_count(mask), 1)


class AlphaQualityTest(unittest.TestCase):
    def test_holed_mask_is_flagged_as_mask_hole(self):
        quality = masks.alpha_quality(ring_mask())
        self.assertEqual(quality["components"], 1)
        self.assertGreater(quality["hole_fraction"], 0.02)
        self.assertEqual(quality["reason_codes"], ["mask_hole"])

    def test_border_shape_is_flagged_as_mask_border(self):
        quality = masks.alpha_quality(left_strip_mask(), params={"border_margin_px": 6})
        self.assertAlmostEqual(quality["border_fraction"], 1.0)
        self.assertEqual(quality["reason_codes"], ["mask_border"])

    def test_clean_disk_has_no_reason_codes(self):
        quality = masks.alpha_quality(disk_mask())
        self.assertEqual(quality["reason_codes"], [])
        self.assertEqual(quality["components"], 1)
        self.assertAlmostEqual(quality["hole_fraction"], 0.0, places=6)
        self.assertAlmostEqual(quality["border_fraction"], 0.0, places=6)

    def test_empty_mask_is_rejected_as_geometry_not_hole(self):
        quality = masks.alpha_quality(np.zeros((60, 60), np.uint8))
        self.assertEqual(quality["components"], 0)
        self.assertEqual(quality["hole_fraction"], 0.0)
        self.assertEqual(quality["border_fraction"], 0.0)
        self.assertEqual(quality["foreground_fraction"], 0.0)
        self.assertEqual(quality["reason_codes"], ["geometry_rejected"])

    def test_two_blocks_are_within_the_component_budget(self):
        quality = masks.alpha_quality(blocks_mask([(10, 10), (40, 40)]))
        self.assertEqual(quality["components"], 2)
        self.assertEqual(quality["reason_codes"], [])

    def test_four_components_are_rejected(self):
        quality = masks.alpha_quality(blocks_mask([(10, 10), (40, 10), (10, 40), (40, 40)]))
        self.assertEqual(quality["components"], 4)
        self.assertEqual(quality["reason_codes"], ["geometry_rejected"])

    def test_result_exposes_the_documented_keys(self):
        quality = masks.alpha_quality(disk_mask())
        self.assertEqual(
            sorted(quality),
            ["border_fraction", "components", "foreground_fraction", "hole_fraction", "reason_codes"],
        )
        self.assertIsInstance(quality["components"], int)

    def test_foreground_fraction_is_the_nonzero_share(self):
        mask = np.zeros((50, 50), np.uint8)
        mask[:10, :] = 255  # 500 of 2500 pixels
        self.assertAlmostEqual(masks.alpha_quality(mask)["foreground_fraction"], 0.2)

    def test_full_frame_mask(self):
        quality = masks.alpha_quality(np.full((40, 40), 255, np.uint8))
        self.assertAlmostEqual(quality["foreground_fraction"], 1.0)
        self.assertEqual(quality["hole_fraction"], 0.0)
        # 40×40 frame with a 2 px band: (1600 - 36×36) / 1600 pixels are border.
        self.assertAlmostEqual(quality["border_fraction"], (1600 - 36 * 36) / 1600)
        self.assertGreater(quality["border_fraction"], 0.15)
        self.assertEqual(quality["reason_codes"], ["mask_border"])

    def test_thresholds_come_from_params(self):
        holed = masks.alpha_quality(ring_mask(), params={"hole_max": 0.5})
        self.assertEqual(holed["reason_codes"], [])
        bordered = masks.alpha_quality(left_strip_mask(), params={"border_margin_px": 0})
        self.assertEqual(bordered["reason_codes"], [])
        budget = masks.alpha_quality(
            blocks_mask([(10, 10), (40, 40)]),
            params={"components_max": 1},
        )
        self.assertEqual(budget["reason_codes"], ["geometry_rejected"])
        unknown_keys_are_ignored = masks.alpha_quality(disk_mask(), params={"unused": 1})
        self.assertEqual(unknown_keys_are_ignored["reason_codes"], [])

    def test_reason_codes_stay_inside_the_shared_vocabulary(self):
        cases = [
            masks.alpha_quality(ring_mask()),
            masks.alpha_quality(left_strip_mask(), params={"border_margin_px": 6}),
            masks.alpha_quality(np.zeros((30, 30), np.uint8)),
            masks.alpha_quality(blocks_mask([(10, 10), (40, 10), (10, 40), (40, 40)])),
        ]
        for quality in cases:
            for code in quality["reason_codes"]:
                self.assertIn(code, REASON_CODES)


class InputToleranceTest(unittest.TestCase):
    def test_single_channel_axis_is_accepted(self):
        mask = ring_mask()
        quality = masks.alpha_quality(mask[:, :, None])
        self.assertEqual(quality["components"], 1)
        self.assertGreater(quality["hole_fraction"], 0.02)

    def test_float_zero_one_mask_matches_uint8_mask(self):
        mask = ring_mask()
        as_float = (mask > 0).astype(np.float32)
        from_uint8 = masks.alpha_quality(mask)
        from_float = masks.alpha_quality(as_float)
        self.assertEqual(from_float["components"], from_uint8["components"])
        self.assertAlmostEqual(from_float["hole_fraction"], from_uint8["hole_fraction"])
        self.assertAlmostEqual(from_float["border_fraction"], from_uint8["border_fraction"])
        self.assertAlmostEqual(from_float["foreground_fraction"], from_uint8["foreground_fraction"])

    def test_any_nonzero_value_is_foreground(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[5:15, 5:15] = 3  # not 255
        self.assertAlmostEqual(masks.alpha_quality(mask)["foreground_fraction"], 0.25)

    def test_multichannel_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            masks.alpha_quality(np.zeros((10, 10, 3), np.uint8))
        with self.assertRaises(ValueError):
            masks.component_count(np.zeros((10, 10, 3), np.uint8))

    def test_empty_array_is_rejected(self):
        with self.assertRaises(ValueError):
            masks.alpha_quality(np.zeros((0, 0), np.uint8))

    def test_negative_margin_is_rejected(self):
        with self.assertRaises(ValueError):
            masks.border_fraction(left_strip_mask(), margin_px=-1)
        with self.assertRaises(ValueError):
            masks.alpha_quality(left_strip_mask(), params={"border_margin_px": -2})

    def test_matches_are_independent_of_repeated_calls(self):
        mask = ring_mask()
        first = masks.alpha_quality(mask)
        second = masks.alpha_quality(mask)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
