"""Deterministic tests for the review judge: no media, no IO."""

import unittest

from datasets.scripts.common.review import judge
from datasets.scripts.common.review.schema import (
    DuplicateReport,
    EmptinessReport,
    FrameRef,
    GroundObservation,
    MetricRecord,
    VERDICT_EXCLUDED,
    VERDICT_READY,
    VERDICT_UNKNOWN,
)

SHAPE = [480, 640]


def ground(ok=True, polygon=None, residual=0.3, placement_patch=None):
    patch = [[140, 380], [500, 380], [520, 470], [120, 470]] if placement_patch is None else placement_patch
    return GroundObservation(
        ok=ok,
        polygon=polygon if polygon is not None else [[120, 300], [520, 300], [620, 470], [20, 470]],
        foot_xy=[320, 460],
        horizon_y=250.0,
        residual=residual,
        reason_codes=["ground_visible", "ground_polygon_fit", "horizon_estimable"] if ok else ["geometry_rejected"],
        placement_patch=patch,
    )


def clean_emptiness():
    return EmptinessReport(clean=True, suspects=[], covered_area_fraction=0.31, reason_codes=["no_unlabelled_target"])


def metric(**kwargs):
    frame = kwargs.pop("frame", FrameRef(path="datasets/raw/x/frame_0001.png", parent_video="datasets/raw/x/v.mp4", timestamp_seconds=1.0))
    record = MetricRecord(frame=frame, task=kwargs.pop("task", "person"))
    record.ground = kwargs.pop("ground", ground())
    record.emptiness = kwargs.pop("emptiness", clean_emptiness())
    record.duplicates = kwargs.pop("duplicates", None)
    record.extra = kwargs.pop("extra", {"image_shape": SHAPE})
    return record


class JudgeTests(unittest.TestCase):
    def test_ready_requires_ground_emptiness_and_no_duplicate(self):
        verdict = judge.decide(metric(), {"thresholds": {}})
        self.assertEqual(verdict.verdict, VERDICT_READY)
        self.assertIn("no_unlabelled_target", verdict.reason_codes)
        self.assertEqual(verdict.ground_polygon, ground().polygon)
        self.assertEqual(verdict.foot_xy, [320, 460])

    def test_missing_ground_is_unknown_not_excluded(self):
        verdict = judge.decide(metric(ground=ground(ok=False, polygon=[])), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertIn("geometry_rejected", verdict.reason_codes)

    def test_small_polygon_is_excluded_with_scale_reason(self):
        tiny = ground(polygon=[[0, 0], [4, 0], [4, 4], [0, 4]])
        verdict = judge.decide(metric(ground=tiny), None)
        self.assertEqual(verdict.verdict, VERDICT_EXCLUDED)
        self.assertIn("scale_out_of_range", verdict.reason_codes)

    def test_low_edge_support_qualifies_but_does_not_block(self):
        verdict = judge.decide(metric(ground=ground(residual=0.9)), None)
        self.assertEqual(verdict.verdict, VERDICT_READY)
        self.assertTrue(verdict.evidence.get("ground_support_low"))

    def test_missing_emptiness_is_unknown_not_ready(self):
        verdict = judge.decide(metric(emptiness=None), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertIn("emptiness_unproven", verdict.reason_codes)

    def test_emptiness_without_templates_is_unknown(self):
        unproven = EmptinessReport(clean=False, suspects=[], covered_area_fraction=0.0, reason_codes=["emptiness_unproven"])
        verdict = judge.decide(metric(emptiness=unproven), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertIn("emptiness_unproven", verdict.reason_codes)

    def test_suspected_unlabelled_target_is_unknown(self):
        suspects = EmptinessReport(
            clean=False,
            suspects=[{"xyxy": [10, 10, 40, 90], "score": 0.81, "method": "template"}],
            covered_area_fraction=0.2,
            reason_codes=["unlabelled_target_suspected"],
        )
        verdict = judge.decide(metric(emptiness=suspects), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertIn("unlabelled_target_suspected", verdict.reason_codes)
        self.assertEqual(verdict.evidence.get("count"), 1)

    def test_near_duplicate_is_excluded(self):
        dup = DuplicateReport(group="dup_0", duplicate_of="datasets/raw/x/frame_0000.png", distances={"phash": 2, "ssim": 0.99, "orb": 0.8})
        verdict = judge.decide(metric(duplicates=dup), None)
        self.assertEqual(verdict.verdict, VERDICT_EXCLUDED)
        self.assertIn("near_duplicate", verdict.reason_codes)
        self.assertEqual(verdict.near_duplicate_group, "dup_0")

    def test_single_duplicate_signal_is_not_enough(self):
        dup = DuplicateReport(group="dup_0", duplicate_of="other.png", distances={"phash": 2, "ssim": 0.5, "orb": 0.1})
        verdict = judge.decide(metric(duplicates=dup), None)
        self.assertEqual(verdict.verdict, VERDICT_READY)

    def test_unverified_parent_keeps_verdict_but_records_source_unresolved(self):
        record = metric(extra={"image_shape": SHAPE, "parent_link_verified": False})
        verdict = judge.decide(record, None)
        self.assertEqual(verdict.verdict, VERDICT_READY)
        self.assertIn("source_unresolved", verdict.reason_codes)

    def test_non_person_hard_negatives_and_below_scale_keep_ready_with_reasons(self):
        report = EmptinessReport(
            clean=True,
            suspects=[],
            covered_area_fraction=0.4,
            reason_codes=["no_unlabelled_target", "below_target_scale", "hard_negative_present"],
            below_scale=[{"xyxy": [10, 10, 40, 30], "score": 0.7, "height_fraction": 0.1}],
            hard_negatives=[{"xyxy": [50, 10, 80, 30], "score": 0.6, "side": "back"}],
        )
        verdict = judge.decide(metric(task="e_bike", emptiness=report), None)
        self.assertEqual(verdict.verdict, VERDICT_READY)
        self.assertIn("below_target_scale", verdict.reason_codes)
        self.assertIn("hard_negative_present", verdict.reason_codes)
        self.assertEqual(verdict.evidence.get("hard_negative_count"), 1)

    def test_person_small_matches_precede_placement_unsure(self):
        report = EmptinessReport(
            clean=True,
            below_scale=[{"xyxy": [10, 10, 20, 32], "score": 0.7}],
        )
        plan = {"placement": {"required": True}}
        verdict = judge.decide(metric(emptiness=report), plan)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertEqual(verdict.reason_codes, ["below_target_scale", "unlabelled_target_suspected"])
        self.assertEqual(verdict.evidence.get("below_scale_count"), 1)

    def test_person_old_emptiness_scope_is_unknown_even_without_hits(self):
        report = EmptinessReport(clean=True, target_spec={"below_min_scale": "not_a_target"})
        verdict = judge.decide(metric(emptiness=report), {"placement": {"required": True}})
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertEqual(verdict.reason_codes, ["emptiness_unproven"])
        self.assertEqual(verdict.evidence.get("judge"), "stale_person_emptiness_scope")

    def test_scaled_search_without_dropped_hits_can_be_ready(self):
        report = EmptinessReport(clean=True, target_spec={"below_min_scale": "not_a_target"})
        record = metric(emptiness=report, extra={"image_shape": SHAPE,
                                                 "emptiness_search": {"reference_frame_height": 360}})
        verdict = judge.decide(record)
        self.assertEqual(verdict.verdict, VERDICT_READY)

    def test_fresh_person_suspect_precedes_placement_even_if_emptiness_optional(self):
        report = EmptinessReport(clean=False, suspects=[{"xyxy": [10, 10, 20, 32], "score": 0.7}])
        plan = {"placement": {"required": True}, "thresholds": {"emptiness_required": False}}
        verdict = judge.decide(metric(emptiness=report), plan)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertEqual(verdict.reason_codes, ["unlabelled_target_suspected"])
        self.assertEqual(verdict.evidence.get("count"), 1)

    def test_known_person_box_excludes_empty_background_and_is_preserved(self):
        targets = [{"class_name": "community", "xyxy": [10, 10, 20, 32]}]
        record = metric(extra={"image_shape": SHAPE, "existing_targets": targets})
        verdict = judge.decide(record, {"placement": {"required": True}})
        self.assertEqual(verdict.verdict, VERDICT_EXCLUDED)
        self.assertEqual(verdict.reason_codes, ["target_annotation_present"])
        self.assertEqual(verdict.existing_targets, targets)
        self.assertEqual(judge.decide(record, role="foreground").verdict, VERDICT_READY)

    def test_background_without_placement_patch_is_unknown(self):
        bare = GroundObservation(
            ok=True,
            polygon=[[120, 300], [520, 300], [620, 470], [20, 470]],
            foot_xy=None,
            horizon_y=250.0,
            residual=0.2,
            reason_codes=["ground_visible", "ground_polygon_fit", "horizon_estimable", "geometry_rejected"],
            placement_patch=[],
        )
        verdict = judge.decide(metric(ground=bare), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertEqual(verdict.evidence.get("judge"), "placement_patch_missing")

    def test_self_intersecting_patch_is_unknown(self):
        bowtie = [[140, 380], [500, 470], [500, 380], [140, 470]]
        verdict = judge.decide(metric(ground=ground(placement_patch=bowtie)), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertEqual(verdict.evidence.get("judge"), "patch_invalid")

    def test_tiny_patch_is_unknown(self):
        tiny = [[100, 400], [110, 400], [110, 404], [100, 404]]
        verdict = judge.decide(metric(ground=ground(placement_patch=tiny)), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)

    def test_patch_covering_a_third_of_the_frame_is_unknown(self):
        band = [[5, 5], [635, 5], [635, 475], [5, 475]]
        verdict = judge.decide(metric(ground=ground(placement_patch=band)), None)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertEqual(verdict.evidence.get("detail"), "patch_too_large")

    def test_cluttered_patch_is_unknown_when_a_free_ratio_is_required(self):
        record = metric()
        record.ground.clearance = {"free_ratio": 0.35, "components": 4}
        plan = {"thresholds": {"min_patch_free_ratio": 0.6}}
        verdict = judge.decide(record, plan)
        self.assertEqual(verdict.verdict, VERDICT_UNKNOWN)
        self.assertEqual(verdict.evidence.get("judge"), "patch_too_cluttered")

    def test_clear_patch_passes_the_free_ratio_gate(self):
        record = metric()
        record.ground.clearance = {"free_ratio": 0.85, "components": 1}
        plan = {"thresholds": {"min_patch_free_ratio": 0.6, "max_patch_obstacle_components": 2}}
        verdict = judge.decide(record, plan)
        self.assertEqual(verdict.verdict, VERDICT_READY)

    def test_threshold_override_changes_verdict(self):
        record = metric(emptiness=None)
        plan = {"thresholds": {"emptiness_required": False}}
        verdict = judge.decide(record, plan)
        self.assertEqual(verdict.verdict, VERDICT_READY)

    def test_summarize_counts_verdicts_reasons_and_groups(self):
        a = judge.decide(metric(), None)
        b = judge.decide(metric(ground=ground(ok=False, polygon=[])), None)
        summary = judge.summarize([a, b])
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["by_verdict"][VERDICT_READY], 1)
        self.assertEqual(summary["by_verdict"][VERDICT_UNKNOWN], 1)
        self.assertIn("datasets/raw/x/v.mp4", summary["by_group"])


if __name__ == "__main__":
    unittest.main()
