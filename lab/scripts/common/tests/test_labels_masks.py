from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from lab.scripts.common import labels, mask_quality


class LabelMaskTest(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[3]
        (workspace / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=workspace / "tmp")
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        cv2.imwrite(
            str(self.image),
            np.zeros((100, 120, 3), np.uint8),
        )

    def tearDown(self):
        self.temp.cleanup()

    def record(
        self,
        state="complete",
        *,
        instances=None,
        accepted_by=None,
        checked=True,
        box_accepted_by=None,
        box_acceptance=None,
    ):
        if instances is None:
            instances = [
                {
                    "id": "p1",
                    "class_id": 0,
                    "visible_bbox_xyxy": [10, 20, 50, 90],
                    "visibility": "full",
                }
            ]
        return {
            "id": "sample",
            "image": self.image.relative_to(self.root).as_posix(),
            "source_kind": "direct_camera_capture",
            "parent_sources": [
                {
                    "path": "datasets/raw/site/video.mp4",
                    "role": "camera_source",
                    "evidence": "datasets/annotated/review.json",
                }
            ],
            "class_names": ["community", "noncommunity"],
            "label_state": state,
            "review": {
                "full_frame_checked": checked,
                "accepted_by": accepted_by,
                "box_accepted_by": box_accepted_by,
                "box_acceptance": box_acceptance,
                "method": "full_frame_semantic_review",
            },
            "instances": instances,
            "partial_reason": "occluded boundary" if state == "partial" else None,
        }

    def test_complete_label_rebuilds_yolo_from_visible_box(self):
        report = labels.audit_manifest(
            {
                "records": [
                    self.record(
                        accepted_by="reviewer",
                        box_accepted_by="reviewer",
                        box_acceptance="human",
                    )
                ]
            },
            root=self.root,
        )
        record = report["records"][0]
        self.assertEqual(record["label_lines"], ["0 0.250000 0.550000 0.333333 0.700000"])
        self.assertEqual(report["summary"]["human_complete_frames"], 1)
        self.assertEqual(report["summary"]["human_complete_instances"], 1)
        output = labels.rebuild_labels(report, "rebuilt", root=self.root)
        self.assertEqual(output, ["rebuilt/sample.txt"])
        self.assertTrue((self.root / "rebuilt/sample.txt").is_file())

    def test_empty_unknown_is_not_negative(self):
        record = self.record(state="unknown", instances=[])
        report = labels.audit_manifest({"records": [record]}, root=self.root)
        self.assertEqual(report["records"][0]["status"], "unknown")
        self.assertEqual(report["records"][0]["label_text"], "")
        self.assertEqual(report["summary"]["unknown_not_negative"], 1)
        self.assertEqual(report["summary"]["confirmed_negative"], 0)

    def test_full_frame_negative_does_not_require_human_reviewer(self):
        record = self.record(state="confirmed_negative", instances=[], accepted_by=None)
        report = labels.audit_manifest({"records": [record]}, root=self.root)
        self.assertEqual(report["records"][0]["status"], "confirmed_negative")
        self.assertEqual(report["summary"]["status_layers"]["human_confirmed_records"], 0)
        record["review"]["accepted_by"] = "reviewer"
        report = labels.audit_manifest({"records": [record]}, root=self.root)
        self.assertEqual(report["records"][0]["status"], "confirmed_negative")

    def test_partial_label_is_retained_but_requires_review(self):
        record = self.record(state="partial")
        report = labels.audit_manifest({"records": [record]}, root=self.root)
        self.assertEqual(report["records"][0]["status"], "review")
        self.assertIn("partial_label", report["records"][0]["reason_codes"])

    def test_semantic_review_does_not_count_as_human_box_acceptance(self):
        record = self.record(accepted_by="reviewer")
        report = labels.audit_manifest({"records": [record]}, root=self.root)
        self.assertEqual(report["records"][0]["status"], "review")
        self.assertIn("box_acceptance_pending", report["records"][0]["reason_codes"])
        self.assertEqual(report["summary"]["human_complete_frames"], 0)
        self.assertEqual(report["summary"]["complete_without_box_acceptance"], 1)
        self.assertEqual(report["summary"]["status_layers"]["pending_records"], 1)
        self.assertEqual(report["summary"]["status_layers"]["human_confirmed_records"], 0)

    def test_candidate_boxes_remain_candidate_and_pending(self):
        record = self.record(
            accepted_by="reviewer",
            box_accepted_by="Codex visual review",
            box_acceptance="candidate",
        )
        report = labels.audit_manifest({"records": [record]}, root=self.root)
        self.assertEqual(report["summary"]["candidate_complete_frames"], 1)
        self.assertEqual(report["summary"]["status_layers"]["candidate_records"], 1)
        self.assertEqual(report["summary"]["status_layers"]["pending_records"], 1)
        self.assertEqual(report["records"][0]["status"], "review")

    def test_programmatic_boxes_are_counted_separately(self):
        record = self.record(
            accepted_by="programmatic_blender_generator",
            box_accepted_by="programmatic_blender_generator",
            box_acceptance="programmatic",
        )
        report = labels.audit_manifest({"records": [record]}, root=self.root)
        self.assertEqual(report["records"][0]["status"], "pass_candidate")
        self.assertEqual(report["summary"]["human_complete_frames"], 0)
        self.assertEqual(report["summary"]["programmatic_complete_frames"], 1)
        self.assertEqual(report["summary"]["status_layers"]["auto_validated_records"], 1)

    def test_mask_layers_and_contact_are_checked(self):
        full = np.zeros((100, 120), np.uint8)
        full[20:90, 20:70] = 255
        visible = full.copy()
        visible[70:90, 35:55] = 0
        occluder = np.zeros_like(full)
        occluder[70:90, 35:55] = 255
        result = mask_quality.audit_mask_arrays(
            full,
            visible,
            occlusion_masks=[occluder],
            ground_polygon=[[0, 80], [119, 80], [119, 99], [0, 99]],
            contact_point=[45, 85],
        )
        self.assertNotIn("hidden_region_unexplained", result["reason_codes"])
        self.assertTrue(result["ground"]["inside"])
        self.assertGreater(result["masks"]["hidden_pixels"], 0)

    def test_visible_mask_outside_full_is_rejected(self):
        full = np.zeros((50, 50), np.uint8)
        full[5:30, 5:30] = 255
        visible = full.copy()
        visible[30:40, 30:40] = 255
        result = mask_quality.audit_mask_arrays(
            full,
            visible,
            ground_polygon=[[0, 0], [49, 0], [49, 49], [0, 49]],
            contact_point=[10, 20],
        )
        self.assertEqual(result["status"], "reject")
        self.assertIn("visible_not_subset", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
