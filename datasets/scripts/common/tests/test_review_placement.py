"""Deterministic checks for placement-patch clutter assessment.

The synthetic scenes are drawn in memory: a plain floor, one large object, one
thin ground marking and one labelled task target. They test the measurement and
decision contract, not real-scene accuracy. The b3 regression frames remain a
manual/visual check because they depend on data under ``datasets/``.
"""

import unittest

import cv2
import numpy as np

from datasets.scripts.common.review import judge, placement
from datasets.scripts.common.review.schema import EmptinessReport, FrameRef, GroundObservation, MetricRecord

WIDTH = 640
HEIGHT = 360
PATCH = [[100, 300], [540, 300], [460, 200], [180, 200]]
TARGET = {"height_fraction": 0.30, "aspect": 0.42}


def floor_scene(seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((HEIGHT, WIDTH, 3), 120, np.uint8)
    noise = rng.normal(0.0, 3.0, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def with_object(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (260, 220), (380, 300), (25, 25, 25), -1)
    cv2.rectangle(result, (260, 220), (380, 300), (240, 240, 240), 2)
    return result


def with_marking(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    cv2.line(result, (120, 280), (520, 280), (245, 245, 245), 3, cv2.LINE_AA)
    return result


def with_target(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (270, 200), (370, 245), (35, 35, 35), -1)
    cv2.rectangle(result, (270, 200), (370, 245), (230, 230, 230), 2)
    return result


def metrics(**overrides) -> dict:
    base = {
        "object_ratio": 0.0,
        "object_max_component_ratio": 0.0,
        "vertical_structure_ratio": 0.0,
        "largest_free_rect_ratio": 0.9,
        "valid_footpoint_ratio": 0.9,
        "best_body_obstacle_ratio": 0.0,
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def metric_record(clearance: dict, *, extra=None) -> MetricRecord:
    ground = GroundObservation(
        ok=True,
        polygon=[[0, 150], [WIDTH - 1, 150], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]],
        foot_xy=[320, 300],
        horizon_y=150.0,
        residual=0.10,
        reason_codes=["ground_visible", "ground_polygon_fit", "placement_patch_fit"],
        placement_patch=[list(point) for point in PATCH],
        clearance=dict(clearance),
    )
    emptiness = EmptinessReport(
        clean=True,
        suspects=[],
        covered_area_fraction=0.0,
        reason_codes=["no_unlabelled_target"],
        target_spec={"min_height_fraction": 0.28},
    )
    return MetricRecord(
        frame=FrameRef(path="synthetic.png", parent_video=None, timestamp_seconds=None),
        task="person",
        ground=ground,
        emptiness=emptiness,
        duplicates=None,
        extra={"image_shape": [HEIGHT, WIDTH], "parent_link_verified": True, **(extra or {})},
    )


class PlacementMeasurementTest(unittest.TestCase):
    def test_empty_floor_is_usable(self):
        assessment = placement.assess_placement(floor_scene(), PATCH, target=TARGET)
        self.assertTrue(assessment.metrics["object_ratio"] < 0.02, assessment.metrics)
        self.assertTrue(assessment.metrics["largest_free_rect_ratio"] >= 0.8, assessment.metrics)
        decision, evidence = placement.placement_decision(assessment.metrics)
        self.assertEqual(decision, "usable", evidence)

    def test_large_object_is_clutter(self):
        assessment = placement.assess_placement(with_object(floor_scene()), PATCH, target=TARGET)
        self.assertGreater(assessment.metrics["object_ratio"], 0.15, assessment.metrics)
        self.assertGreater(assessment.metrics["object_max_component_ratio"], 0.08, assessment.metrics)
        decision, evidence = placement.placement_decision(assessment.metrics)
        self.assertEqual(decision, "cluttered", evidence)

    def test_thin_marking_is_reported_not_counted_as_object(self):
        assessment = placement.assess_placement(with_marking(floor_scene()), PATCH, target=TARGET)
        self.assertGreater(assessment.metrics["marking_ratio"], 0.0, assessment.metrics)
        self.assertTrue(assessment.metrics["object_ratio"] < 0.05, assessment.metrics)
        decision, evidence = placement.placement_decision(assessment.metrics)
        self.assertEqual(decision, "usable", evidence)

    def test_labelled_target_is_not_clutter(self):
        assessment = placement.assess_placement(
            with_target(floor_scene()),
            PATCH,
            target=TARGET,
            boxes=[{"class_name": "person", "xyxy": [270, 200, 370, 245]}],
        )
        self.assertGreater(assessment.metrics["task_target_ratio"], 0.0, assessment.metrics)
        self.assertTrue(assessment.metrics["object_ratio"] < 0.05, assessment.metrics)
        decision, evidence = placement.placement_decision(assessment.metrics)
        self.assertNotEqual(decision, "cluttered", evidence)
        failures = {item["check"] for item in evidence.get("failures", [])}
        self.assertNotIn("object_ratio_exceeded", failures)

    def test_missing_metrics_is_unsure(self):
        decision, evidence = placement.placement_decision({"object_ratio": 0.0})
        self.assertEqual(decision, "unsure")
        self.assertEqual(evidence["judge"], "placement_metrics_missing")

    def test_low_confidence_is_unsure(self):
        decision, evidence = placement.placement_decision(metrics(confidence=0.2))
        self.assertEqual(decision, "unsure")
        self.assertEqual(evidence["judge"], "placement_confidence_low")

    def test_borderline_over_threshold_is_unsure(self):
        decision, evidence = placement.placement_decision(
            metrics(object_ratio=0.12, object_max_component_ratio=0.09)
        )
        self.assertEqual(decision, "unsure")
        self.assertEqual(evidence["judge"], "placement_patch_unsure")

    def test_near_limit_is_unsure(self):
        decision, evidence = placement.placement_decision(metrics(object_ratio=0.095))
        self.assertEqual(decision, "unsure")
        self.assertEqual(evidence["judge"], "placement_patch_borderline")

    def test_clear_over_threshold_is_cluttered(self):
        decision, evidence = placement.placement_decision(metrics(object_ratio=0.2))
        self.assertEqual(decision, "cluttered")
        self.assertEqual(evidence["judge"], "placement_patch_cluttered")


class JudgePlacementGateTest(unittest.TestCase):
    def plan(self, required=True):
        return {
            "thresholds": {
                "ground_min_area_ratio": 0.02,
                "ground_min_plane_support": 0.30,
                "ground_max_horizon_fraction": 0.85,
                "emptiness_required": True,
                "suspect_min_score": 0.62,
                "near_duplicate_phash_max": 8,
                "near_duplicate_ssim_min": 0.95,
                "require_two_duplicate_signals": True,
                "min_ground_area_ratio": 0.0,
                "min_patch_free_ratio": 0.0,
                "max_patch_obstacle_components": 0,
            },
            "placement": {
                "required": required,
                "thresholds": dict(placement.DEFAULT_THRESHOLDS),
            },
        }

    def test_usable_placement_passes_to_ready(self):
        verdict = judge.decide(metric_record(metrics()), self.plan())
        self.assertEqual(verdict.verdict, "ready", verdict)
        self.assertNotIn("placement_patch_cluttered", verdict.reason_codes)

    def test_cluttered_placement_is_excluded(self):
        verdict = judge.decide(
            metric_record(
                metrics(object_ratio=0.2, object_max_component_ratio=0.2)
            ),
            self.plan(),
        )
        self.assertEqual(verdict.verdict, "excluded", verdict)
        self.assertIn("placement_patch_cluttered", verdict.reason_codes)

    def test_unsure_placement_stays_unknown(self):
        verdict = judge.decide(
            metric_record(metrics(confidence=0.2)),
            self.plan(),
        )
        self.assertEqual(verdict.verdict, "unknown", verdict)
        self.assertIn("placement_patch_unsure", verdict.reason_codes)

    def test_disabled_gate_keeps_previous_behaviour(self):
        verdict = judge.decide(
            metric_record(metrics(object_ratio=0.2, object_max_component_ratio=0.2)),
            self.plan(required=False),
        )
        self.assertEqual(verdict.verdict, "ready", verdict)


if __name__ == "__main__":
    unittest.main()
