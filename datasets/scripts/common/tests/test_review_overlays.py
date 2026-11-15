"""Contract tests for the review overlay drawing layer."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from datasets.scripts.common.review import overlays
from datasets.scripts.common.review.schema import (
    DuplicateReport,
    EmptinessReport,
    GroundObservation,
)


def make_ground():
    return GroundObservation(
        ok=True,
        polygon=[[20, 20], [180, 20], [180, 130], [20, 130]],
        foot_xy=[100, 110],
        horizon_y=60.0,
        residual=1.2,
    )


def make_emptiness():
    return EmptinessReport(
        clean=False,
        suspects=[
            {"xyxy": [30, 30, 80, 90], "score": 0.81},
            {"box": [120, 40, 170, 100], "confidence": 0.66},
        ],
        covered_area_fraction=0.05,
    )


class DrawOverlayTest(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((150, 200, 3), np.uint8)

    def test_draws_expected_layers_without_mutating_input(self):
        result = overlays.draw_overlay(
            self.image,
            ground=make_ground(),
            emptiness=make_emptiness(),
            duplicates=DuplicateReport(group="g1"),
            caption="caption",
        )
        self.assertEqual(result.shape, self.image.shape)
        self.assertEqual(result.dtype, self.image.dtype)
        self.assertTrue(np.array_equal(self.image, np.zeros((150, 200, 3), np.uint8)))
        self.assertFalse(np.array_equal(result, self.image))

        green = int(np.all(result == overlays.GREEN, axis=2).sum())
        red = int(np.all(result == overlays.RED, axis=2).sum())
        yellow = int(np.all(result == overlays.YELLOW, axis=2).sum())
        white = int(np.all(result > 240, axis=2).sum())
        self.assertGreater(green, 300)
        self.assertGreater(red, 100)
        self.assertGreater(yellow, 20)
        self.assertGreater(white, 5)

    def test_every_argument_combination_is_tolerated(self):
        layers = {
            "ground": make_ground(),
            "emptiness": make_emptiness(),
            "duplicates": DuplicateReport(group="g1"),
        }
        for ground in (None, layers["ground"]):
            for emptiness in (None, layers["emptiness"]):
                for duplicates in (None, layers["duplicates"]):
                    with self.subTest(ground=ground is not None,
                                      emptiness=emptiness is not None,
                                      duplicates=duplicates is not None):
                        result = overlays.draw_overlay(
                            self.image,
                            ground=ground,
                            emptiness=emptiness,
                            duplicates=duplicates,
                        )
                        self.assertEqual(result.shape, self.image.shape)
                        self.assertTrue(np.array_equal(self.image, np.zeros((150, 200, 3), np.uint8)))

    def test_empty_reports_draw_nothing(self):
        result = overlays.draw_overlay(
            self.image,
            ground=GroundObservation(ok=False),
            emptiness=EmptinessReport(clean=True, suspects=[]),
            duplicates=None,
        )
        self.assertTrue(np.array_equal(result, self.image))

    def test_accepts_grayscale_input(self):
        gray = np.zeros((50, 60), np.uint8)
        result = overlays.draw_overlay(gray, ground=GroundObservation(
            ok=True, polygon=[[5, 5], [50, 5], [50, 40], [5, 40]], foot_xy=[20, 30]))
        self.assertEqual(result.shape, gray.shape)


class WriteOverlayTest(unittest.TestCase):
    def test_write_creates_parent_and_returns_path(self):
        image = np.zeros((60, 80, 3), np.uint8)
        image[10:20, 10:20] = (0, 255, 0)
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "batch" / "overlays" / "frame_0001.png"
            returned = overlays.write_overlay(image, target)
            self.assertEqual(returned, str(target))
            self.assertTrue(target.exists())
            loaded = cv2.imread(str(target))
            self.assertEqual(loaded.shape, image.shape)

    def test_write_failure_raises_runtime_error(self):
        image = np.zeros((20, 20, 3), np.uint8)
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                overlays.write_overlay(image, Path(tmp) / "overlay.unknown_extension")


if __name__ == "__main__":
    unittest.main()
