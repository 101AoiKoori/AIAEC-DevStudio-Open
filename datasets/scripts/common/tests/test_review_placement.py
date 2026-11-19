"""Deterministic checks for placement-patch clutter assessment.

The synthetic scenes are drawn in memory: a plain floor, one large object, one
thin ground marking and one labelled task target. They test the measurement and
decision contract, not real-scene accuracy. The b3 regression frames remain a
manual/visual check because they depend on data under ``datasets/``.
"""

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from datasets.scripts.common.review import geometry, judge, placement, placement_record
from datasets.scripts.common.audit_background_space import _refine_sparse_ground
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


class GroundPlacementTest(unittest.TestCase):
    def test_prompt_refinement_preserves_obstacles_and_leaves_mask_holes_unknown(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        image[72:89, 39:45] = 220
        candidate = np.zeros(image.shape[:2], dtype=bool)
        candidate[78:91, 12:26] = True
        obstacle = np.zeros_like(candidate)
        obstacle[72:89, 95:110] = True
        floor = np.zeros_like(candidate, dtype=np.uint8)
        floor[60:, :] = 1
        floor[72:89, 65:81] = 0
        old = placement.GroundPlacement("candidate", "fixture", candidate, obstacle)
        with patch("datasets.scripts.common.audit_background_space._prompt_mask", return_value=floor):
            refined, evidence, outcome = _refine_sparse_ground(image, old, object(), source_kind="blender_render")
        self.assertEqual(outcome, "supported")
        self.assertEqual(evidence["prompt_basis"], "inset_contact_seeds_with_upper_background_prior")
        self.assertEqual(refined.contact_status((72, 80)), "unknown")
        self.assertEqual(refined.contact_status((41, 80)), "unknown")
        self.assertTrue(evidence["bright_render_markers_excluded"])
        self.assertEqual(refined.contact_status((100, 80)), "blocked")
        self.assertEqual(refined.contact_status((47, 80)), "candidate")
        np.testing.assert_array_equal(refined.obstacle_mask, obstacle)

    def test_failed_prompt_refinement_keeps_previous_contact_evidence(self):
        image = np.full((100, 120, 3), 40, dtype=np.uint8)
        candidate = np.zeros(image.shape[:2], dtype=bool)
        candidate[78:91, 12:26] = True
        old = placement.GroundPlacement("candidate", "fixture", candidate, np.zeros_like(candidate))
        with patch("datasets.scripts.common.audit_background_space._prompt_mask", return_value=None):
            result, evidence, outcome = _refine_sparse_ground(image, old, object())
        self.assertEqual(outcome, "prompt_unresolved")
        self.assertTrue(evidence["prompt_attempts"])
        self.assertIs(result, old)

    def test_reference_quad_fits_only_supported_pixels(self):
        candidate = np.zeros((60, 80), dtype=bool)
        candidate[18:55, 8:70] = True
        candidate[18:55, 36:42] = False
        reference = geometry.contact_reference_quad(candidate)
        self.assertIsNotNone(reference)
        (x0, y0), _, (x1, y1), _ = reference
        self.assertTrue(candidate[y0:y1 + 1, x0:x1 + 1].all())
        self.assertIsNone(geometry.contact_reference_quad(np.zeros_like(candidate)))

    def test_obstacle_mask_blocks_even_where_floor_appearance_agrees(self):
        obstacle = np.zeros((HEIGHT, WIDTH), dtype=bool)
        obstacle[242:281, 304:324] = True
        floor = np.ones_like(obstacle)
        result = placement.assess_placeable_ground(floor_scene(), PATCH, floor_mask=floor, obstacle_mask=obstacle)
        self.assertEqual(result.contact_status((314, 270)), "blocked")
        self.assertEqual(result.contact_status((260, 270)), "candidate")
        result = placement.assess_placeable_ground(floor_scene(), [], obstacle_mask=obstacle)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.contact_status((314, 270)), "blocked")

    def test_local_floor_does_not_cross_an_obstacle_or_fill_appearance_holes(self):
        support = np.ones((180, 320), dtype=bool)
        patch = np.ones_like(support)
        obstacle = np.zeros_like(support)
        obstacle[154:157, :160] = True
        support[170, 240] = False
        contact = placement._local_floor_contacts(support, patch, obstacle)
        self.assertFalse(contact[140, 100])
        self.assertFalse(contact[170, 240])
        self.assertTrue(contact[140, 220])
        self.assertTrue(contact[170, 100])
        self.assertFalse((contact & obstacle).any())

    def test_geometry_and_low_clutter_alone_leave_ground_unknown(self):
        result = placement.assess_placeable_ground(floor_scene(), PATCH)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.basis, "near_floor_appearance_unresolved")
        self.assertEqual(result.contact_status((320, 270)), "unknown")

    def test_clutter_vetoes_auto_floor_and_marks_suspected_obstacle(self):
        result = placement.assess_placeable_ground(with_object(floor_scene()), PATCH)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.basis, "placement_cluttered")
        self.assertEqual(result.contact_status((320, 270)), "blocked")
        self.assertFalse(result.candidate_mask.any())

    def test_supplied_positive_mask_cannot_override_target_or_extend_contact(self):
        floor = np.zeros((HEIGHT, WIDTH), dtype=bool)
        floor[245:290, 210:430] = True
        result = placement.assess_placeable_ground(
            floor_scene(), PATCH, floor_mask=floor,
            boxes=[{"class_name": "person", "xyxy": [300, 250, 330, 280]}],
        )
        self.assertEqual(result.status, "candidate")
        self.assertEqual(result.basis, "supplied_floor_mask")
        self.assertEqual(result.contact_status((250, 260), contact_width=16), "candidate")
        self.assertEqual(result.contact_status((215, 260), contact_width=16), "unknown")
        self.assertEqual(result.contact_status((292, 260), contact_width=20), "blocked")
        self.assertEqual(result.contact_status((250, 240)), "unknown")


class PlacementRecordTest(unittest.TestCase):
    def test_record_preserves_unknown_and_blocked_holes_inside_reference_quad(self):
        candidate = np.ones((20, 30), dtype=bool)
        candidate[:, 14:16] = False
        blocked = np.zeros_like(candidate)
        blocked[8:13, 14:16] = True
        assessment = placement.GroundPlacement("candidate", "fixture", candidate, blocked)
        quad = ((1, 1), (28, 1), (28, 18), (1, 18))
        record = placement_record.encode_placement_record(assessment, reference_quad_px=quad, horizon_y=0)
        restored = placement_record.decode_placement_record(json.loads(json.dumps(record)))
        self.assertEqual(record["reference_quad_px"], [list(point) for point in quad])
        np.testing.assert_array_equal(restored.candidate_mask, candidate)
        np.testing.assert_array_equal(restored.obstacle_mask, blocked)
        self.assertEqual(restored.contact_status((15, 10), contact_width=8), "blocked")
        self.assertEqual(restored.contact_status((15, 5), contact_width=8), "unknown")

    def test_output_contact_uses_crop_scale_and_padding(self):
        candidate = np.ones((20, 30), dtype=bool)
        candidate[:, 13] = False
        blocked = ~candidate
        assessment = placement.GroundPlacement("candidate", "fixture", candidate, blocked)
        bounds = {"crop_xyxy": (10, 4, 24, 16), "content_xyxy": (0, 6, 28, 30)}
        query = placement_record.contact_status_in_view
        self.assertEqual(query(assessment, (7, 14), contact_width=1, **bounds), "blocked")
        self.assertEqual(query(assessment, (16, 14), contact_width=4, **bounds), "candidate")
        self.assertEqual(query(assessment, (16, 5), contact_width=4, **bounds), "unknown")
        self.assertEqual(query(assessment, (0, 14), contact_width=2, **bounds), "unknown")

    def test_record_rejects_mask_declared_for_another_image_shape(self):
        candidate = np.zeros((20, 30), dtype=bool)
        assessment = placement.GroundPlacement("unknown", "fixture", candidate, candidate)
        record = placement_record.encode_placement_record(assessment)
        record["image_shape"] = [30, 20]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            placement_record.decode_placement_record(record)


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
