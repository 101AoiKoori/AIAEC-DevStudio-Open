from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from lab.scripts.common import blender_cameras, branch_sample, cli


class BlenderCameraPlanTest(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[3]
        (workspace / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=workspace / "tmp")
        self.root = Path(self.temp.name)
        self.scene = self.root / "scene.blend"
        self.scene.write_bytes(b"BLENDER-test-fixture")

    def tearDown(self):
        self.temp.cleanup()

    def plan(self):
        return {
            "task": "person",
            "id": "camera-test",
            "master_scene": "scene.blend",
            "scene_truth": "test_fixture",
            "target": {
                "height_m": 0.15,
                "anchor_m": [0.0, 0.03],
                "front_axis_xy": [0.0, -1.0],
            },
            "required_views": ["front", "oblique", "side"],
            "required_coverage_tags": ["single", "multi", "interference"],
            "required_scale_bins": [[0.04, 0.30]],
            "cameras": [
                {
                    "id": "front",
                    "view": "front",
                    "location_m": [0.0, -0.40, 0.16],
                    "target_m": [0.0, 0.03, 0.085],
                    "lens_mm": 22,
                    "resolution": [640, 480],
                    "coverage_tags": ["single", "multi", "interference"],
                },
                {
                    "id": "oblique",
                    "view": "oblique",
                    "location_m": [0.28, -0.28, 0.18],
                    "target_m": [0.0, 0.03, 0.085],
                    "lens_mm": 22,
                    "resolution": [640, 480],
                },
                {
                    "id": "side",
                    "view": "side",
                    "location_m": [0.40, 0.03, 0.16],
                    "target_m": [0.0, 0.03, 0.085],
                    "lens_mm": 22,
                    "resolution": [640, 480],
                },
            ],
        }

    def test_camera_projection_and_coverage_pass(self):
        report = blender_cameras.validate_camera_plan(
            self.plan(),
            root=self.root,
            registry_lookup=lambda path, root: {"status": "found", "asset": {"state": "active_in_use"}},
        )
        self.assertNotEqual(report["status"], "fail")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["coverage"]["views"], {"front": 1, "oblique": 1, "side": 1})
        self.assertEqual(report["coverage"]["coverage_tags"]["interference"], 1)

    def test_missing_required_view_fails(self):
        plan = self.plan()
        plan["required_views"].append("back")
        report = blender_cameras.validate_camera_plan(
            plan,
            root=self.root,
            registry_lookup=lambda path, root: {"status": "found", "asset": {"state": "active_in_use"}},
        )
        self.assertEqual(report["status"], "fail")
        self.assertIn("view_not_covered:back", report["errors"])

    def test_view_label_mismatch_fails(self):
        plan = self.plan()
        plan["cameras"][1]["view"] = "front"
        report = blender_cameras.validate_camera_plan(
            plan,
            root=self.root,
            registry_lookup=lambda path, root: {"status": "found", "asset": {"state": "active_in_use"}},
        )
        self.assertTrue(any("view_class_mismatch" in error for error in report["errors"]))

    def test_reference_registration_is_measured(self):
        plan = self.plan()
        plan["reference_registration"] = {
            "world_ground_quad_m": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            "image_ground_quad_px": [[80, 300], [560, 300], [600, 470], [40, 470]],
            "max_reprojection_error_px": 1.0,
            "source_image": "reference.png",
        }
        report = blender_cameras.validate_camera_plan(
            plan,
            root=self.root,
            registry_lookup=lambda path, root: {"status": "found", "asset": {"state": "active_in_use"}},
        )
        registration = report["coverage"]["reference_registration"]
        self.assertEqual(registration["status"], "pass")
        self.assertLess(registration["max_error_px"], 1e-6)


class BranchSampleTest(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[3]
        (workspace / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=workspace / "tmp")
        self.root = Path(self.temp.name)
        for name in ("direct.png", "composite.png", "render.png", "video.mp4", "background.png", "art.png", "scene.blend", "mask.png"):
            path = self.root / name
            if path.suffix == ".png":
                cv2.imwrite(str(path), np.zeros((20, 20, 3), np.uint8))
            else:
                path.write_bytes(b"fixture")

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self):
        base = {
            "content_branches": ["empty"],
            "instance_count": 0,
            "label_state": "unknown",
            "background_state": "candidate_empty",
            "review": {"accepted_by": None, "full_frame_checked": False},
            "evidence": {},
        }
        return {
            "task": "person",
            "batch": "test",
            "required_source_kinds": [
                "direct_camera_capture",
                "real_image_composite",
                "blender_render",
            ],
            "required_content_branches": ["empty", "single", "multi", "interference"],
            "records": [
                {
                    **base,
                    "id": "empty",
                    "image": "direct.png",
                    "source_kind": "direct_camera_capture",
                    "parent_sources": [{"path": "video.mp4", "role": "camera_source"}],
                },
                {
                    **base,
                    "id": "single",
                    "image": "composite.png",
                    "source_kind": "real_image_composite",
                    "content_branches": ["single"],
                    "instance_count": 1,
                    "label_state": "complete",
                    "parent_sources": [
                        {"path": "background.png", "role": "real_background"},
                        {"path": "art.png", "role": "target_source"},
                    ],
                },
                {
                    **base,
                    "id": "multi",
                    "image": "direct.png",
                    "source_kind": "direct_camera_capture",
                    "content_branches": ["multi", "interference"],
                    "instance_count": 2,
                    "label_state": "partial",
                    "parent_sources": [{"path": "video.mp4", "role": "camera_source"}],
                    "interference_evidence": {"occlusion_layer": "mask.png"},
                },
                {
                    **base,
                    "id": "render",
                    "image": "render.png",
                    "source_kind": "blender_render",
                    "content_branches": ["multi"],
                    "instance_count": 3,
                    "label_state": "unknown",
                    "parent_sources": [{"path": "scene.blend", "role": "blender_scene"}],
                },
            ],
        }

    def test_branch_complete_sample_reports_pending_review(self):
        report = branch_sample.validate_sample(self.manifest(), root=self.root)
        self.assertEqual(report["status"], "review")
        self.assertEqual(report["errors"], [])
        self.assertEqual(
            report["summary"]["source_branches"],
            {
                "blender_render": 1,
                "direct_camera_capture": 2,
                "real_image_composite": 1,
            },
        )
        self.assertEqual(
            set(report["summary"]["content_branches"]),
            {"empty", "single", "multi", "interference"},
        )
        self.assertEqual(len(report["manual_review"]), 4)
        matrices = {
            item["id"]: item["status_matrix"] for item in report["manual_review"]
        }
        self.assertEqual(matrices["empty"]["objects"], [])
        self.assertEqual(len(matrices["single"]["objects"]), 1)
        self.assertEqual(len(matrices["multi"]["objects"]), 2)
        self.assertEqual(len(matrices["render"]["objects"]), 3)
        self.assertEqual(
            set(matrices["single"]["objects"][0]),
            set(branch_sample.OBJECT_MATRIX_COLUMNS),
        )
        self.assertIsNone(matrices["multi"]["presence"])
        self.assertFalse(matrices["multi"]["full_frame_checked"])

    def test_full_frame_negative_without_reviewer_passes(self):
        manifest = self.manifest()
        manifest["records"][0].update(
            {
                "label_state": "confirmed_negative",
                "review": {"accepted_by": None, "full_frame_checked": True},
            }
        )
        report = branch_sample.validate_sample(manifest, root=self.root)
        self.assertEqual(report["errors"], [])

    def test_missing_source_branch_fails(self):
        manifest = self.manifest()
        manifest["required_source_kinds"].append("render_real_mix")
        report = branch_sample.validate_sample(manifest, root=self.root)
        self.assertIn("source_branch_missing:render_real_mix", report["errors"])

    def test_overlay_only_does_not_count_as_raw_render_branch(self):
        manifest = self.manifest()
        manifest["records"][3]["image_role"] = "review_overlay_not_raw_render"
        report = branch_sample.validate_sample(manifest, root=self.root)
        self.assertIn("blender_render:raw_render_branch_missing", report["warnings"])

    def test_programmatic_acceptance_is_not_reported_as_human(self):
        manifest = self.manifest()
        manifest["records"][3]["review"] = {
            "accepted_by": "programmatic_blender_generator",
            "full_frame_checked": True,
        }
        report = branch_sample.validate_sample(manifest, root=self.root)
        self.assertNotIn("render:human_acceptance_missing", report["warnings"])
        self.assertEqual(report["summary"]["human_accepted"], 0)
        self.assertEqual(report["summary"]["programmatic_accepted"], 1)
        render_result = next(item for item in report["records"] if item["id"] == "render")
        self.assertEqual(render_result["acceptance_method"], "programmatic")

    def test_manual_review_accepts_single_and_contact_sheet_shapes(self):
        review = {
            "task": "person",
            "batch": "test",
            "class_names": ["community", "noncommunity"],
            "items": [
                {
                    "id": "single",
                    "status_matrix": {
                        "image_role": "single_frame",
                        "presence": "single",
                        "task_target_count": 1,
                        "background_state": "contains_target",
                        "objects": [
                            {
                                "row_id": "obj_001",
                                "left_to_right_order": 1,
                                "category": "community_person",
                                "view": "front",
                                "visibility": "full",
                                "occluder": "none",
                                "is_task_target": True,
                                "annotation_action": "label_visible_box",
                                "visible_bbox_xyxy": [2, 2, 12, 18],
                                "full_bbox_xyxy": None,
                                "box_accepted_by": "reviewer",
                                "notes": None,
                            },
                            {
                                "row_id": "obj_002",
                                "left_to_right_order": 2,
                                "category": "person_standee_nonfront",
                                "view": "side",
                                "visibility": "full",
                                "occluder": "none",
                                "is_task_target": False,
                                "annotation_action": "hard_negative_no_box",
                                "visible_bbox_xyxy": None,
                                "full_bbox_xyxy": None,
                                "box_accepted_by": None,
                                "notes": "edge-on standee",
                            }
                        ],
                        "other_objects": [],
                        "reviewer": "reviewer",
                        "full_frame_checked": True,
                        "notes": None,
                    },
                },
                {
                    "id": "sheet",
                    "status_matrix": {
                        "image_role": "contact_sheet",
                        "reviewer": "reviewer",
                        "notes": None,
                        "panels": [
                            {
                                "panel_id": "row1_col1",
                                "source_name": "panel-a.png",
                                "presence": "none",
                                "task_target_count": 0,
                                "background_state": "confirmed_empty",
                                "objects": [],
                                "other_objects": [],
                                "reviewer": "reviewer",
                                "full_frame_checked": True,
                                "notes": None,
                            }
                        ],
                    },
                },
            ],
        }
        report = branch_sample.validate_manual_review(review)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["errors"], [])
        self.assertEqual([item["image_role"] for item in report["items"]], ["single_frame", "contact_sheet"])

    def test_manual_review_rejects_legacy_panel_keys_and_string_booleans(self):
        review = {
            "items": [
                {
                    "id": "bad",
                    "status_matrix": {
                        "presence": "multi",
                        "task_target_count": 1,
                        "background_state": "contains_target",
                        "objects1,1": [],
                        "objects": [
                            {
                                "row_id": "obj_001",
                                "category": "community_person",
                                "view": "front",
                                "visibility": "full",
                                "occluder": "person",
                                "is_task_target": "true",
                                "annotation_action": "label_visible_box",
                            }
                        ],
                        "reviewer": "reviewer",
                        "full_frame_checked": True,
                    },
                }
            ]
        }
        report = branch_sample.validate_manual_review(review)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("legacy_panel_keys" in error for error in report["errors"]))
        self.assertTrue(any("is_task_target_not_boolean" in error for error in report["errors"]))

    def test_accepted_review_preserves_source_digest(self):
        review = {
            "task": "person",
            "batch": "test",
            "class_names": ["community", "noncommunity"],
            "items": [
                {
                    "id": "empty",
                    "status_matrix": {
                        "image_role": "single_frame",
                        "presence": "none",
                        "task_target_count": 0,
                        "background_state": "confirmed_empty",
                        "objects": [],
                        "other_objects": [],
                        "reviewer": "reviewer",
                        "full_frame_checked": True,
                        "notes": None,
                    },
                }
            ],
        }
        source = self.root / "manual_review.json"
        source.write_text("{}", encoding="utf-8")
        accepted = branch_sample.review_acceptance_document(review, source_review=source, root=self.root)
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(len(accepted["source_sha256"]), 64)
        self.assertEqual(accepted["items"][0]["status_matrix"]["reviewer"], "reviewer")

    def test_review_template_does_not_replace_existing_edits(self):
        target = self.root / "manual_review.json"
        target.write_text('{"edited": true}\n', encoding="utf-8")
        created = branch_sample.write_review_template_if_missing(
            target,
            {"items": []},
            root=self.root,
        )
        self.assertFalse(created)
        self.assertEqual(target.read_text(encoding="utf-8"), '{"edited": true}\n')

    def test_default_validate_merges_review_and_is_stable_on_rerun(self):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(self.manifest(), ensure_ascii=False),
            encoding="utf-8",
        )
        review_path = self.root / "manual_review.json"
        review_path.write_text(
            json.dumps(
                {
                    "task": "person",
                    "batch": "test",
                    "items": [
                        {
                            "id": "single",
                            "status_matrix": {
                                "presence": None,
                                "background_state": "contains_target",
                                "accepted_by": "human",
                                "objects": [{"row_id": "obj_001", "category": "community_person"}],
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        args = type(
            "Args",
            (),
            {"manifest": "manifest.json", "output": "sample_validation.json"},
        )()
        with patch.object(cli, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli._sample_validate(args), 0)
            first = review_path.read_bytes()
            self.assertEqual(cli._sample_validate(args), 0)
            second = review_path.read_bytes()
        self.assertEqual(first, second)
        merged = json.loads(second)
        matrix = merged["items"][0]["status_matrix"]
        self.assertIsNone(matrix["presence"])
        self.assertEqual(matrix["background_state"], "contains_target")
        self.assertEqual(matrix["accepted_by"], "human")
        self.assertEqual(matrix["objects"][0]["category"], "community_person")
        self.assertTrue(list(self.root.glob("manual_review.pre_rebuild_*.json")))

    def test_review_template_is_created_once_and_rows_include_box_fields(self):
        target = self.root / "manual_review.json"
        payload = {"items": [{"id": "single", "status_matrix": {"objects": [{"row_id": "obj_001"}]}}]}
        self.assertTrue(branch_sample.write_review_template_if_missing(target, payload, root=self.root))
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        before = target.read_text(encoding="utf-8")
        self.assertFalse(
            branch_sample.write_review_template_if_missing(
                target,
                {"items": [{"id": "replacement"}]},
                root=self.root,
            )
        )
        self.assertEqual(target.read_text(encoding="utf-8"), before)
        row = branch_sample._object_rows(1)[0]
        self.assertIn("visible_bbox_xyxy", row)
        self.assertIn("full_bbox_xyxy", row)
        self.assertIn("box_accepted_by", row)

    def test_merge_preserves_null_and_nonnull_manual_values_on_repeat(self):
        existing = {
            "items": [
                {
                    "id": "single",
                    "status_matrix": {
                        "presence": None,
                        "background_state": "contains_target",
                        "accepted_by": "human",
                        "objects": [
                            {
                                "row_id": "obj_001",
                                "category": "community_person",
                                "visible_bbox_xyxy": None,
                            }
                        ],
                    },
                }
            ]
        }
        generated = {
            "instructions": "new instructions",
            "items": [
                {
                    "id": "single",
                    "status_matrix": {
                        "presence": "single",
                        "background_state": None,
                        "accepted_by": None,
                        "full_frame_checked": False,
                        "objects": [
                            {
                                "row_id": "obj_001",
                                "category": None,
                                "visible_bbox_xyxy": [1, 2, 8, 9],
                                "full_bbox_xyxy": None,
                                "box_accepted_by": None,
                            }
                        ],
                    },
                }
            ],
        }
        merged = branch_sample.merge_review_preserving_existing(existing, generated)
        matrix = merged["items"][0]["status_matrix"]
        row = matrix["objects"][0]
        self.assertIsNone(matrix["presence"])
        self.assertEqual(matrix["background_state"], "contains_target")
        self.assertEqual(matrix["accepted_by"], "human")
        self.assertEqual(row["category"], "community_person")
        self.assertIsNone(row["visible_bbox_xyxy"])
        self.assertIn("full_bbox_xyxy", row)
        again = branch_sample.merge_review_preserving_existing(merged, generated)
        self.assertEqual(again, merged)

    def test_rebuild_review_template_backs_up_manual_content(self):
        target = self.root / "manual_review.json"
        manual = {
            "items": [
                {
                    "id": "single",
                    "status_matrix": {
                        "presence": "none",
                        "task_target_count": 0,
                        "background_state": "confirmed_empty",
                        "objects": [],
                        "reviewer": "reviewer",
                        "full_frame_checked": True,
                    },
                }
            ]
        }
        target.write_text(json.dumps(manual, ensure_ascii=False), encoding="utf-8")
        result = branch_sample.rebuild_review_template(
            target,
            {"items": [{"id": "replacement", "status_matrix": {"objects": []}}]},
            root=self.root,
        )
        self.assertIsNotNone(result["backup"])
        backup = self.root / result["backup"]
        self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), manual)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["items"][0]["id"], "replacement")

    def test_normalize_review_accepts_box_correction_and_separates_confirmations(self):
        review = {
            "task": "person",
            "batch": "test",
            "class_names": ["community", "noncommunity"],
            "items": [
                {
                    "id": "single",
                    "image": "direct.png",
                    "source_kind": "direct_camera_capture",
                    "class_names": ["community", "noncommunity"],
                    "parent_sources": [{"path": "video.mp4", "role": "camera_source"}],
                    "status_matrix": {
                        "image_role": "single_frame",
                        "presence": None,
                        "task_target_count": None,
                        "background_state": None,
                        "objects": [
                            {
                                "row_id": "obj_001",
                                "left_to_right_order": 1,
                                "category": None,
                                "view": None,
                                "visibility": None,
                                "occluder": None,
                                "is_task_target": None,
                                "annotation_action": None,
                                "visible_bbox_xyxy": None,
                                "full_bbox_xyxy": None,
                                "box_accepted_by": None,
                                "notes": None,
                            }
                        ],
                        "other_objects": [],
                        "reviewer": None,
                        "accepted_by": None,
                        "full_frame_checked": False,
                        "review_state": "pending",
                        "notes": None,
                    },
                }
            ],
        }
        answers = {
            "accepted_by": "human",
            "items": [
                {
                    "id": "single",
                    "full_frame_checked": True,
                    "background_state": "contains_target",
                    "task_target_count": 1,
                    "objects": [
                        {
                            "row_id": "obj_001",
                            "category": "community_person",
                            "view": "front",
                            "visibility": "full",
                            "occluder": "none",
                            "is_task_target": True,
                            "annotation_action": "label_visible_box",
                            "visible_bbox_xyxy": [5, 6, 15, 18],
                            "full_bbox_xyxy": [4, 5, 16, 19],
                            "box_accepted_by": "programmatic_blender_generator",
                        }
                    ],
                }
            ],
        }
        normalized = branch_sample.normalize_review_answers(review, answers)
        report = branch_sample.validate_manual_review(normalized)
        self.assertEqual(report["status"], "pass")
        layers = report["status_layers"]
        self.assertEqual(layers["human_confirmed"]["frames"], 1)
        self.assertEqual(layers["human_confirmed"]["box_instances"], 0)
        self.assertEqual(layers["auto_validated"]["box_instances"], 1)
        manifest = branch_sample.label_manifest_from_review(normalized)
        self.assertEqual(manifest["records"][0]["label_state"], "complete")
        self.assertIsNone(manifest["records"][0]["review"]["box_accepted_by"])
        self.assertEqual(manifest["records"][0]["review"]["box_acceptance"], "candidate")

        normalized["items"][0]["status_matrix"]["objects"][0]["box_accepted_by"] = "human"
        manifest = branch_sample.label_manifest_from_review(normalized)
        self.assertEqual(manifest["records"][0]["label_state"], "complete")
        self.assertEqual(manifest["records"][0]["review"]["box_accepted_by"], "human")

    def test_full_frame_negative_and_nonfront_no_box_remain_pending_semantics(self):
        review = branch_sample._matrix_template({"instance_count": 0})
        review.update(
            {
                "presence": "none",
                "task_target_count": 0,
                "background_state": "confirmed_empty",
                "reviewer": "human",
                "accepted_by": "human",
                "full_frame_checked": True,
                "review_state": "confirmed",
            }
        )
        document = {
            "task": "person",
            "batch": "test",
            "items": [{"id": "empty", "status_matrix": review}],
        }
        report = branch_sample.validate_manual_review(document)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["status_layers"]["human_confirmed"]["negative_frames"], 1)
        self.assertEqual(report["status_layers"]["pending"]["frames"], 0)

        nonfront = {
            "task": "person",
            "batch": "test",
            "items": [
                {
                    "id": "nonfront",
                    "status_matrix": {
                        "image_role": "single_frame",
                        "presence": "none",
                        "task_target_count": 0,
                        "background_state": "confirmed_empty",
                        "objects": [
                            {
                                "row_id": "obj_001",
                                "left_to_right_order": 1,
                                "category": "person_standee_nonfront",
                                "view": "side",
                                "visibility": "full",
                                "occluder": "none",
                                "is_task_target": False,
                                "annotation_action": "hard_negative_no_box",
                                "visible_bbox_xyxy": None,
                                "full_bbox_xyxy": None,
                                "box_accepted_by": None,
                                "notes": "edge-on standee",
                            }
                        ],
                        "other_objects": [],
                        "reviewer": "human",
                        "accepted_by": "human",
                        "full_frame_checked": True,
                        "review_state": "confirmed",
                        "notes": None,
                    },
                }
            ],
        }
        report = branch_sample.validate_manual_review(nonfront)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["status_layers"]["human_confirmed"]["box_instances"], 0)

    def test_semantic_acceptance_can_persist_while_boxes_are_pending(self):
        review = {
            "task": "person",
            "batch": "test",
            "items": [
                {
                    "id": "single",
                    "status_matrix": {
                        "image_role": "single_frame",
                        "presence": "single",
                        "task_target_count": 1,
                        "background_state": "contains_target",
                        "objects": [
                            {
                                "row_id": "obj_001",
                                "left_to_right_order": 1,
                                "category": "community_person",
                                "view": "front",
                                "visibility": "full",
                                "occluder": "none",
                                "is_task_target": True,
                                "annotation_action": "label_visible_box",
                                "visible_bbox_xyxy": None,
                                "full_bbox_xyxy": None,
                                "box_accepted_by": None,
                                "notes": None,
                            }
                        ],
                        "other_objects": [],
                        "reviewer": "human",
                        "accepted_by": "human",
                        "full_frame_checked": True,
                        "review_state": "confirmed",
                        "notes": None,
                    },
                }
            ],
        }
        validation = branch_sample.validate_manual_review(review)
        self.assertEqual(validation["status"], "review")
        source = self.root / "manual_review.json"
        source.write_text(json.dumps(review), encoding="utf-8")
        accepted = branch_sample.review_acceptance_document(
            review,
            source_review=source,
            validation=validation,
            root=self.root,
        )
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["status_layers"]["human_confirmed"]["box_instances"], 0)
        self.assertEqual(accepted["status_layers"]["pending"]["label_action_boxes"], 1)

    def test_csv_review_answers_are_normalized_without_filling_box_acceptance(self):
        answers = self.root / "answers.csv"
        answers.write_text(
            "item_id,panel_id,row_id,presence,background_state,task_target_count,"
            "category,view,visibility,occluder,is_task_target,annotation_action,"
            "visible_bbox_xyxy,full_bbox_xyxy,box_accepted_by,accepted_by,"
            "full_frame_checked,review_state,notes\n"
            'single,,obj_001,multi,contains_target,1,community_person,front,full,'
            'none,true,label_visible_box,"[1,2,9,10]","[0,1,10,11]",,human,'
            'true,confirmed,\n',
            encoding="utf-8",
        )
        loaded = branch_sample.load_review_answers(answers, root=self.root)
        row = loaded["items"][0]["objects"][0]
        self.assertEqual(row["visible_bbox_xyxy"], [1, 2, 9, 10])
        self.assertNotIn("box_accepted_by", row)


if __name__ == "__main__":
    unittest.main()
