"""Checks for the Gabor/FFT texture fallback and patch-level fallbacks.

The synthetic scenes test the contract: the texture profile must find a real
wall/floor step, reject a uniform scene, and the geometry fallbacks must return
a legal patch only when the normal geometry checks pass.
"""

import unittest
from unittest import mock

import cv2
import numpy as np

from datasets.scripts.common.review import geometry, texture

WIDTH = 640
HEIGHT = 360


def wall_floor_scene(boundary=140, floor_lines=True) -> np.ndarray:
    rng = np.random.default_rng(4)
    image = np.full((HEIGHT, WIDTH, 3), 20, np.uint8)
    floor = np.full((HEIGHT - boundary, WIDTH, 3), 105, np.uint8)
    floor = np.clip(floor.astype(np.float64) + rng.normal(0.0, 3.0, floor.shape), 0, 255).astype(np.uint8)
    image[boundary:] = floor
    if floor_lines:
        for row in range(boundary + 20, HEIGHT, 30):
            cv2.line(image, (0, row), (WIDTH - 1, row), (235, 235, 235), 3, cv2.LINE_AA)
    return image


class TextureBoundaryTest(unittest.TestCase):
    def test_texture_finds_the_wall_floor_step(self):
        result = texture.texture_floor_top(wall_floor_scene(), horizon_y=140)
        self.assertIsNotNone(result["row"], result)
        self.assertLessEqual(abs(int(result["row"]) - 140), 20, result)
        self.assertGreater(result["confidence"], 0.0, result)

    def test_uniform_image_has_no_texture_boundary(self):
        image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
        result = texture.texture_floor_top(image, horizon_y=100)
        self.assertIsNone(result["row"], result)

    def test_short_floor_strip_is_rejected_by_patch_depth(self):
        image = wall_floor_scene(boundary=HEIGHT - 12)
        result = texture.texture_floor_top(image, horizon_y=100)
        self.assertIsNotNone(result["row"], result)
        self.assertLess(result["floor_fraction"], 0.10, result)
        self.assertIsNone(
            geometry._texture_floor_fallback(
                image, (HEIGHT, WIDTH), 100.0, geometry._resolve_params(None)
            )
        )


class GeometryFallbackTest(unittest.TestCase):
    def options(self):
        return geometry._resolve_params(None)

    def test_level_camera_patch_without_homography(self):
        floor = [[0, 200], [WIDTH - 1, 200], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]]
        line = np.array([0.0, 1.0, -180.0])
        patch, detail, mode = geometry._placement_patch_with_fallback(
            None, floor, line, (HEIGHT, WIDTH), self.options(), ()
        )
        self.assertEqual(mode, "level_camera", detail)
        self.assertEqual(len(patch), 4)
        self.assertTrue(geometry._patch_depth_ok(patch, (HEIGHT, WIDTH), self.options()))

    def test_vp_failure_retries_level_camera(self):
        floor = [[0, 200], [WIDTH - 1, 200], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]]
        line = np.array([0.0, 1.0, -180.0])
        good = [[100, HEIGHT - 3], [WIDTH - 100, HEIGHT - 3], [WIDTH - 200, 220], [200, 220]]
        with mock.patch.object(geometry, "placement_patch_report", side_effect=[
            ([], "patch_thin"),
            (good, ""),
        ]):
            patch, _, mode = geometry._placement_patch_with_fallback(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                floor, line, (HEIGHT, WIDTH), self.options(),
                [[100.0, 100.0], [400.0, 100.0]],
            )
        self.assertEqual(mode, "level_camera")
        self.assertEqual(patch, good)

    def test_texture_fallback_rejects_shallow_patch(self):
        image = wall_floor_scene(boundary=HEIGHT - 20)
        result = geometry._texture_floor_fallback(
            image, (HEIGHT, WIDTH), 120.0, self.options()
        )
        self.assertIsNone(result)

    def test_patch_depth_threshold(self):
        shallow = [[10, 350], [WIDTH - 10, 350], [WIDTH - 20, 330], [20, 330]]
        deep = [[10, 350], [WIDTH - 10, 350], [WIDTH - 20, 200], [20, 200]]
        options = self.options()
        self.assertFalse(geometry._patch_depth_ok(shallow, (HEIGHT, WIDTH), options))
        self.assertTrue(geometry._patch_depth_ok(deep, (HEIGHT, WIDTH), options))


if __name__ == "__main__":
    unittest.main()
